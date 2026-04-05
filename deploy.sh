#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename -- "$0")"

NGINX_USER_EXPLICIT=false
WEB_GROUP_EXPLICIT=false

if [[ -n "${NGINX_USER:-}" ]]; then
  NGINX_USER_EXPLICIT=true
fi

if [[ -n "${WEB_GROUP:-}" ]]; then
  WEB_GROUP_EXPLICIT=true
fi

HUGO_DIR="${HUGO_DIR:-$SCRIPT_DIR}"
TARGET_DIR="${TARGET_DIR:-/var/www/lanshi.space}"
NGINX_USER="${NGINX_USER:-}"
WEB_GROUP="${WEB_GROUP:-}"
HUGO_CONFIG="${HUGO_CONFIG:-config.yaml}"
LOCK_FILE="${LOCK_FILE:-/tmp/hugo_blog_deploy.lock}"
BUILD_LOG="${BUILD_LOG:-}"
AI_PROXY_SERVICE_NAME="${AI_PROXY_SERVICE_NAME:-hugo-blog-ai-proxy}"
AI_PROXY_USER="${AI_PROXY_USER:-$(id -un)}"
AI_PROXY_GROUP="${AI_PROXY_GROUP:-}"
AI_PROXY_SHELL_RC="${AI_PROXY_SHELL_RC:-}"
AI_PROXY_ENV_FILE="${AI_PROXY_ENV_FILE:-}"

DRY_RUN=false
PULL_UPDATES=false
SKIP_PDFINFO=false
SKIP_PERMISSION_FIX=false
SKIP_NGINX_RELOAD=false
SKIP_AI_PROXY=false

BUILD_OUTPUT_DIR=""
LOCK_FD=""
TARGET_EXISTS=false

log() {
  local level="$1"
  shift
  printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$*"
}

log_info() {
  log "INFO" "$*"
}

log_warn() {
  log "WARN" "$*"
}

log_error() {
  log "ERROR" "$*"
}

usage() {
  cat <<EOF
用法: ./$SCRIPT_NAME [选项]

选项:
  --dry-run                仅演练流程，不写入目标目录、不改权限、不重载 Nginx
  --pull                   部署前执行 git pull --ff-only
  --skip-pdfinfo           跳过 gen-pdfinfo.sh
  --skip-permission-fix    跳过 find + chmod 权限修正
  --skip-nginx-reload      跳过 Nginx 配置检查与重载
  --skip-ai-proxy          跳过 AI 代理服务安装/启动
  --hugo-dir <path>        Hugo 项目目录（默认脚本所在目录）
  --target-dir <path>      网站目标目录（默认: $TARGET_DIR）
  --config <file>          Hugo 配置文件（相对 hugo-dir，默认: $HUGO_CONFIG）
  --nginx-user <user>      目标文件所属用户（默认: 自动继承目标目录/父目录属主，否则回退当前用户）
  --web-group <group>      目标文件所属组（默认: 自动继承目标目录/父目录属组，否则回退当前主组）
  --lock-file <path>       锁文件路径（默认: $LOCK_FILE）
  --build-log <path>       构建日志路径（默认: /tmp/hugo_build_时间戳.log）
  -h, --help               显示帮助

环境变量同名参数可覆盖默认值:
  HUGO_DIR TARGET_DIR NGINX_USER WEB_GROUP HUGO_CONFIG LOCK_FILE BUILD_LOG
  AI_PROXY_SERVICE_NAME AI_PROXY_USER AI_PROXY_GROUP AI_PROXY_SHELL_RC AI_PROXY_ENV_FILE
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        DRY_RUN=true
        ;;
      --pull)
        PULL_UPDATES=true
        ;;
      --skip-pdfinfo)
        SKIP_PDFINFO=true
        ;;
      --skip-permission-fix)
        SKIP_PERMISSION_FIX=true
        ;;
      --skip-nginx-reload)
        SKIP_NGINX_RELOAD=true
        ;;
      --skip-ai-proxy)
        SKIP_AI_PROXY=true
        ;;
      --hugo-dir)
        [[ $# -ge 2 ]] || { log_error "--hugo-dir 需要参数"; exit 1; }
        HUGO_DIR="$2"
        shift
        ;;
      --target-dir)
        [[ $# -ge 2 ]] || { log_error "--target-dir 需要参数"; exit 1; }
        TARGET_DIR="$2"
        shift
        ;;
      --config)
        [[ $# -ge 2 ]] || { log_error "--config 需要参数"; exit 1; }
        HUGO_CONFIG="$2"
        shift
        ;;
      --nginx-user)
        [[ $# -ge 2 ]] || { log_error "--nginx-user 需要参数"; exit 1; }
        NGINX_USER="$2"
        NGINX_USER_EXPLICIT=true
        shift
        ;;
      --web-group)
        [[ $# -ge 2 ]] || { log_error "--web-group 需要参数"; exit 1; }
        WEB_GROUP="$2"
        WEB_GROUP_EXPLICIT=true
        shift
        ;;
      --lock-file)
        [[ $# -ge 2 ]] || { log_error "--lock-file 需要参数"; exit 1; }
        LOCK_FILE="$2"
        shift
        ;;
      --build-log)
        [[ $# -ge 2 ]] || { log_error "--build-log 需要参数"; exit 1; }
        BUILD_LOG="$2"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        log_error "未知参数: $1"
        usage
        exit 1
        ;;
    esac
    shift
  done
}

cleanup() {
  local exit_code=$?
  set +e

  if [[ -n "$BUILD_OUTPUT_DIR" && -d "$BUILD_OUTPUT_DIR" ]]; then
    rm -rf "$BUILD_OUTPUT_DIR"
  fi

  if [[ $exit_code -ne 0 ]]; then
    if [[ -n "$BUILD_LOG" ]]; then
      log_error "部署失败，请检查日志: $BUILD_LOG"
    else
      log_error "部署失败。"
    fi
  fi
}

trap cleanup EXIT

check_dependencies() {
  local deps=(hugo rsync flock find node)

  if [[ "$PULL_UPDATES" == true ]]; then
    deps+=(git)
  fi

  if [[ "$DRY_RUN" != true ]]; then
    deps+=(sudo)
  fi

  if [[ "$DRY_RUN" != true && "$SKIP_NGINX_RELOAD" != true ]]; then
    deps+=(systemctl)
  fi

  if [[ "$SKIP_AI_PROXY" != true ]]; then
    deps+=(node npm systemctl)
    if [[ "$AI_PROXY_SHELL_RC" == *.zshrc ]]; then
      deps+=(zsh)
    fi
  fi

  local cmd
  for cmd in "${deps[@]}"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      log_error "未找到依赖命令: $cmd"
      exit 1
    fi
  done
}

find_existing_reference_path() {
  local path="$1"

  while [[ ! -e "$path" && "$path" != "/" ]]; do
    path="$(dirname "$path")"
  done

  if [[ -e "$path" ]]; then
    printf '%s\n' "$path"
  fi
}

resolve_ai_proxy_defaults() {
  if [[ -z "$AI_PROXY_GROUP" ]]; then
    AI_PROXY_GROUP="$(id -gn "$AI_PROXY_USER")"
    log_info "AI 代理服务用户组默认设置为: $AI_PROXY_GROUP"
  fi

  if [[ -z "$AI_PROXY_SHELL_RC" ]]; then
    local ai_proxy_home=""

    if command -v getent >/dev/null 2>&1; then
      ai_proxy_home="$(getent passwd "$AI_PROXY_USER" | cut -d: -f6)"
    fi

    if [[ -z "$ai_proxy_home" ]]; then
      ai_proxy_home="$(eval printf '%s' "~$AI_PROXY_USER")"
    fi

    if [[ -n "$ai_proxy_home" && -f "$ai_proxy_home/.zshrc" ]]; then
      AI_PROXY_SHELL_RC="$ai_proxy_home/.zshrc"
      log_info "自动检测 AI 代理环境文件: $AI_PROXY_SHELL_RC"
    elif [[ -n "$ai_proxy_home" && -f "$ai_proxy_home/.bashrc" ]]; then
      AI_PROXY_SHELL_RC="$ai_proxy_home/.bashrc"
      log_info "自动检测 AI 代理环境文件: $AI_PROXY_SHELL_RC"
    fi
  fi
}

resolve_owner_group_defaults() {
  local reference_path=""
  local inferred_user=""
  local inferred_group=""

  reference_path="$(find_existing_reference_path "$TARGET_DIR")"

  if [[ -n "$reference_path" ]]; then
    if [[ "$NGINX_USER_EXPLICIT" != true ]]; then
      inferred_user="$(stat -Lc '%U' "$reference_path" 2>/dev/null || true)"
      if [[ -n "$inferred_user" && "$inferred_user" != "UNKNOWN" ]]; then
        NGINX_USER="$inferred_user"
        log_info "自动解析部署用户: $NGINX_USER (来源: $reference_path)"
      fi
    fi

    if [[ "$WEB_GROUP_EXPLICIT" != true ]]; then
      inferred_group="$(stat -Lc '%G' "$reference_path" 2>/dev/null || true)"
      if [[ -n "$inferred_group" && "$inferred_group" != "UNKNOWN" ]]; then
        WEB_GROUP="$inferred_group"
        log_info "自动解析部署用户组: $WEB_GROUP (来源: $reference_path)"
      fi
    fi
  fi

  if [[ "$NGINX_USER_EXPLICIT" != true && -z "$NGINX_USER" ]]; then
    NGINX_USER="$(id -un)"
    log_info "目标路径无法推断属主，回退部署用户为当前用户: $NGINX_USER"
  fi

  if [[ "$WEB_GROUP_EXPLICIT" != true && -z "$WEB_GROUP" ]]; then
    WEB_GROUP="$(id -gn)"
    log_info "目标路径无法推断属组，回退部署用户组为当前主组: $WEB_GROUP"
  fi
}

run_with_optional_sudo() {
  if [[ "$DRY_RUN" == true ]]; then
    "$@"
  else
    if ! sudo -n true >/dev/null 2>&1; then
      ensure_sudo_ready
    fi
    sudo -n "$@"
  fi
}

ensure_sudo_ready() {
  if [[ "$DRY_RUN" == true ]]; then
    return
  fi

  if sudo -n true >/dev/null 2>&1; then
    return
  fi

  log_info "需要 sudo 权限，正在请求认证..."
  if ! sudo -v; then
    log_error "需要 sudo 权限，但当前环境无法交互输入密码；请先执行 sudo -v 或配置无密码 sudo"
    exit 1
  fi

  if ! sudo -n true >/dev/null 2>&1; then
    log_error "sudo 凭据校验失败，无法继续部署"
    exit 1
  fi
}

validate_settings() {
  [[ -n "$HUGO_DIR" ]] || { log_error "HUGO_DIR 不能为空"; exit 1; }
  [[ -n "$TARGET_DIR" ]] || { log_error "TARGET_DIR 不能为空"; exit 1; }
  [[ -n "$NGINX_USER" ]] || { log_error "NGINX_USER 不能为空"; exit 1; }
  [[ -n "$WEB_GROUP" ]] || { log_error "WEB_GROUP 不能为空"; exit 1; }

  if [[ "$TARGET_DIR" == "/" ]]; then
    log_error "TARGET_DIR 不能是根目录 /"
    exit 1
  fi

  if [[ ! -d "$HUGO_DIR" ]]; then
    log_error "Hugo 目录不存在: $HUGO_DIR"
    exit 1
  fi

  if [[ ! -f "$HUGO_DIR/$HUGO_CONFIG" ]]; then
    log_error "在 $HUGO_DIR 中未找到配置文件: $HUGO_CONFIG"
    exit 1
  fi

  local lock_dir
  lock_dir="$(dirname "$LOCK_FILE")"
  if [[ ! -d "$lock_dir" ]]; then
    log_error "锁文件目录不存在: $lock_dir"
    exit 1
  fi

  if [[ ! -w "$lock_dir" ]]; then
    log_error "锁文件目录不可写: $lock_dir"
    exit 1
  fi

  if [[ "$DRY_RUN" != true ]]; then
    if ! id -u "$NGINX_USER" >/dev/null 2>&1; then
      log_error "用户不存在: $NGINX_USER"
      exit 1
    fi

    if command -v getent >/dev/null 2>&1; then
      if ! getent group "$WEB_GROUP" >/dev/null 2>&1; then
        log_error "用户组不存在: $WEB_GROUP"
        exit 1
      fi
    else
      log_warn "未找到 getent，跳过用户组存在性校验"
    fi

    ensure_sudo_ready
  else
    log_info "dry-run 模式下跳过用户/用户组与 sudo 校验"
  fi

  if [[ "$SKIP_AI_PROXY" != true ]]; then
    if [[ ! -f "$HUGO_DIR/server/ai-assistant-proxy.mjs" ]]; then
      log_error "未找到 AI 代理入口: $HUGO_DIR/server/ai-assistant-proxy.mjs"
      exit 1
    fi

    if [[ ! -f "$HUGO_DIR/server/run-ai-assistant-proxy.sh" ]]; then
      log_error "未找到 AI 代理启动脚本: $HUGO_DIR/server/run-ai-assistant-proxy.sh"
      exit 1
    fi

    if [[ "$DRY_RUN" != true ]]; then
      if ! id -u "$AI_PROXY_USER" >/dev/null 2>&1; then
        log_error "AI 代理服务用户不存在: $AI_PROXY_USER"
        exit 1
      fi

      if command -v getent >/dev/null 2>&1; then
        if ! getent group "$AI_PROXY_GROUP" >/dev/null 2>&1; then
          log_error "AI 代理服务用户组不存在: $AI_PROXY_GROUP"
          exit 1
        fi
      fi
    fi
  fi
}

acquire_lock() {
  if ! exec {LOCK_FD}>"$LOCK_FILE"; then
    log_error "无法创建或打开锁文件: $LOCK_FILE"
    exit 1
  fi

  if ! flock -n "$LOCK_FD"; then
    log_error "已有部署任务在运行，锁文件: $LOCK_FILE"
    exit 1
  fi
}

prepare_target() {
  if [[ -d "$TARGET_DIR" ]]; then
    TARGET_EXISTS=true
    return
  fi

  if [[ "$DRY_RUN" == true ]]; then
    TARGET_EXISTS=false
    log_warn "dry-run 模式下目标目录不存在，将跳过 rsync 演练: $TARGET_DIR"
  else
    log_info "目标目录不存在，准备创建: $TARGET_DIR"
    run_with_optional_sudo install -d -m 2775 -o "$NGINX_USER" -g "$WEB_GROUP" "$TARGET_DIR"
    TARGET_EXISTS=true
  fi
}

pull_updates() {
  if [[ "$PULL_UPDATES" != true ]]; then
    return
  fi

  if ! git -C "$HUGO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log_error "--pull 已启用，但 $HUGO_DIR 不是 git 仓库"
    exit 1
  fi

  if [[ -n "$(git -C "$HUGO_DIR" status --porcelain)" ]]; then
    log_warn "检测到未提交改动，git pull --ff-only 可能失败"
  fi

  log_info "拉取最新仓库内容..."
  git -C "$HUGO_DIR" pull --ff-only
}

run_pdfinfo_generation() {
  if [[ "$SKIP_PDFINFO" == true ]]; then
    log_info "已跳过 PDF 元数据生成"
    return
  fi

  if [[ ! -f "$HUGO_DIR/gen-pdfinfo.sh" ]]; then
    log_info "未找到 gen-pdfinfo.sh，跳过 PDF 元数据生成"
    return
  fi

  log_info "更新 PDF 元数据..."
  if ! (cd "$HUGO_DIR" && bash "./gen-pdfinfo.sh"); then
    log_warn "gen-pdfinfo.sh 执行失败，继续部署"
  fi
}

build_site() {
  BUILD_OUTPUT_DIR="$(mktemp -d /tmp/hugo_public.XXXXXX)"

  if [[ -z "$BUILD_LOG" ]]; then
    BUILD_LOG="/tmp/hugo_build_$(date '+%Y%m%d_%H%M%S').log"
  fi

  log_info "开始 Hugo 构建，日志: $BUILD_LOG"

  if ! (
    cd "$HUGO_DIR"
    hugo \
      --config "$HUGO_CONFIG" \
      --cleanDestinationDir \
      --gc \
      --minify \
      --destination "$BUILD_OUTPUT_DIR" \
      >"$BUILD_LOG" 2>&1
  ); then
    log_error "Hugo 构建失败"
    return 1
  fi

  log_info "Hugo 构建完成"
}

run_search_embedding_generation() {
  local generator="$HUGO_DIR/server/generate-search-embeddings.mjs"

  if [[ ! -f "$generator" ]]; then
    log_info "未找到 generate-search-embeddings.mjs，跳过向量索引生成"
    return
  fi

  log_info "生成搜索向量索引..."
  if ! (
    cd "$HUGO_DIR"
    SEARCH_OUTPUT_DIR="$BUILD_OUTPUT_DIR" node "$generator" >>"$BUILD_LOG" 2>&1
  ); then
    log_warn "搜索向量索引生成失败，搜索页将自动降级到关键词搜索"
  fi
}

sync_site() {
  if [[ "$DRY_RUN" == true && "$TARGET_EXISTS" != true ]]; then
    log_info "跳过同步：dry-run 模式且目标目录不存在"
    return
  fi

  local rsync_args=(
    -a
    -h
    --delete
    --delete-delay
    --delay-updates
    "--chmod=D2775,F664"
  )

  if [[ "$DRY_RUN" == true ]]; then
    rsync_args+=(--dry-run --itemize-changes)
  else
    rsync_args+=("--chown=${NGINX_USER}:${WEB_GROUP}")
  fi

  log_info "同步文件到: $TARGET_DIR"
  run_with_optional_sudo rsync "${rsync_args[@]}" "$BUILD_OUTPUT_DIR/" "$TARGET_DIR/"
}

fix_permissions() {
  if [[ "$DRY_RUN" == true ]]; then
    log_info "dry-run 模式下跳过权限修正"
    return
  fi

  if [[ "$SKIP_PERMISSION_FIX" == true ]]; then
    log_info "已跳过权限修正"
    return
  fi

  log_info "修正目标目录权限..."
  run_with_optional_sudo find "$TARGET_DIR" -type d -exec chmod 2775 {} +
  run_with_optional_sudo find "$TARGET_DIR" -type f -exec chmod 664 {} +
}

reload_nginx() {
  if [[ "$DRY_RUN" == true ]]; then
    log_info "dry-run 模式下跳过 Nginx 重载"
    return
  fi

  if [[ "$SKIP_NGINX_RELOAD" == true ]]; then
    log_info "已跳过 Nginx 重载"
    return
  fi

  if command -v nginx >/dev/null 2>&1; then
    log_info "检查 Nginx 配置..."
    run_with_optional_sudo nginx -t
  else
    log_warn "未找到 nginx 命令，跳过 nginx -t 校验"
  fi

  log_info "重载 Nginx..."
  run_with_optional_sudo systemctl reload nginx
}

manage_ai_proxy() {
  if [[ "$SKIP_AI_PROXY" == true ]]; then
    log_info "已跳过 AI 代理服务安装/启动"
    return
  fi

  local service_path="/etc/systemd/system/${AI_PROXY_SERVICE_NAME}.service"
  local unit_tmp
  unit_tmp="$(mktemp /tmp/${AI_PROXY_SERVICE_NAME}.XXXXXX.service)"

  cat >"$unit_tmp" <<EOF
[Unit]
Description=Hugo Blog AI Assistant Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$AI_PROXY_USER
Group=$AI_PROXY_GROUP
WorkingDirectory=$HUGO_DIR
Environment=HUGO_DIR=$HUGO_DIR
Environment=AI_PROXY_SHELL_RC=$AI_PROXY_SHELL_RC
Environment=AI_PROXY_ENV_FILE=$AI_PROXY_ENV_FILE
ExecStart=/usr/bin/env bash $HUGO_DIR/server/run-ai-assistant-proxy.sh
Restart=always
RestartSec=5
TimeoutStartSec=30
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

  if [[ "$DRY_RUN" == true ]]; then
    log_info "dry-run 模式下跳过 AI 代理服务安装"
    rm -f "$unit_tmp"
    return
  fi

  log_info "安装/更新 AI 代理 systemd 服务: $AI_PROXY_SERVICE_NAME"
  run_with_optional_sudo install -m 644 "$unit_tmp" "$service_path"
  rm -f "$unit_tmp"

  log_info "重新加载 systemd 配置..."
  run_with_optional_sudo systemctl daemon-reload

  log_info "启用并重启 AI 代理服务..."
  run_with_optional_sudo systemctl enable "$AI_PROXY_SERVICE_NAME"
  run_with_optional_sudo systemctl restart "$AI_PROXY_SERVICE_NAME"
  run_with_optional_sudo systemctl --no-pager --full status "$AI_PROXY_SERVICE_NAME"
}

main() {
  parse_args "$@"
  resolve_owner_group_defaults
  resolve_ai_proxy_defaults

  log_info "==== 开始部署流程 ===="
  log_info "HUGO_DIR=$HUGO_DIR"
  log_info "TARGET_DIR=$TARGET_DIR"
  log_info "DEPLOY_OWNER=${NGINX_USER}:${WEB_GROUP}"
  log_info "DRY_RUN=$DRY_RUN"
  log_info "AI_PROXY_SERVICE_NAME=$AI_PROXY_SERVICE_NAME"
  log_info "AI_PROXY_RUNTIME=${AI_PROXY_USER}:${AI_PROXY_GROUP}"

  check_dependencies
  validate_settings
  acquire_lock
  prepare_target
  pull_updates
  run_pdfinfo_generation
  build_site
  run_search_embedding_generation
  sync_site
  fix_permissions
  manage_ai_proxy
  reload_nginx

  if [[ "$DRY_RUN" == true ]]; then
    log_info "==== 演练完成（未实际部署）===="
  else
    log_info "==== 部署成功 ===="
  fi
}

main "$@"
