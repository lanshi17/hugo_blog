#!/usr/bin/env bash
set -eo pipefail

# ----------------------
# 配置变量区（按需修改）
# ----------------------
HUGO_DIR="/home/$USER/hugo_blog"
TARGET_DIR="/var/www/lanshi.space"
NGINX_USER="hugo"
WEB_GROUP="webadmin"
HUGO_CONFIG="config.yaml"

# ----------------------
# 功能函数区
# ----------------------
log() {
    local timestamp=$(date +"%Y-%m-%d %T")
    echo -e "[${timestamp}] $1"
}

check_dependencies() {
    local deps=("hugo" "sudo" "rsync")
    for cmd in "${deps[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            log "\033[31m错误：未找到 $cmd 命令\033[0m"
            exit 1
        fi
    done
}

# ----------------------
# 主执行流程
# ----------------------
log "\033[34m==== 开始部署流程 ====\033[0m"

# 1. 检查前置依赖
check_dependencies

# 2. 验证 Hugo 项目目录
if [[ ! -f "$HUGO_DIR/$HUGO_CONFIG" ]]; then
    log "\033[31m错误：在 $HUGO_DIR 中未找到配置文件 $HUGO_CONFIG\033[0m"
    exit 1
fi

# 3. 进入项目目录
log "进入 Hugo 目录: $HUGO_DIR"
cd "$HUGO_DIR" || exit 1

# 4. 清理旧生成文件
log "清理旧生成文件..."
rm -rf public resources .hugo_build.lock

# 5. 构建静态网站
log "开始生成静态文件 (详细日志见 /tmp/hugo_build.log)"
hugo \
    --config "$HUGO_CONFIG" \
    --cleanDestinationDir \
    --gc \
    --minify \
    --logLevel debug \
    --destination "$HUGO_DIR/public" \
    > /tmp/hugo_build.log 2>&1

if [ $? -ne 0 ]; then
    log "\033[31mHugo 构建失败，请检查 /tmp/hugo_build.log\033[0m"
    exit 1
fi

# 6. 同步文件到目标目录
log "同步文件到 $TARGET_DIR"
sudo rsync -avh --delete --chown=${NGINX_USER}:${WEB_GROUP} \
    "$HUGO_DIR/public/" \
    "$TARGET_DIR/"

# 7. 设置权限（继承父目录权限）
log "设置目录权限..."
sudo find "$TARGET_DIR" -type d -exec chmod 2775 {} \;
sudo find "$TARGET_DIR" -type f -exec chmod 664 {} \;

# 8. 重载 Nginx（而非重启）
log "重载 Nginx 配置..."
sudo systemctl reload nginx

# ----------------------
# 完成提示
# ----------------------
log "\033[32m==== 部署成功！ ====\033[0m"
