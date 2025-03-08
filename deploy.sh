#!/usr/bin/env bash
#!/usr/bin/env bash
set -e

# 1) 进入您的 Hugo 博客本地目录(假设为 ~/hugo_blog)
cd ~/hugo_blog

# 2) 生成静态文件，--minify 会压缩文件体积(可选)
hugo  --config config.yaml   --cleanDestinationDir --gc --minify 

# 3) 清空 Nginx 指向的目标目录(假设为 /var/www/lanshi.xyz)
sudo rm -rf /var/www/lanshi.xyz/*

# 4) 拷贝 Hugo 编译后的 public 文件夹至目标目录
sudo cp -r public/* /var/www/lanshi.xyz/

# 5) 重新加载或重启 Nginx，以应用可能的配置更改(可视需求选择 reload/restart)
sudo systemctl restart nginx

echo "Blog updated and Nginx reloaded successfully!"
#
# set -e

# BLOG_DIR="$HOME/hugo_blog"
# TARGET_DIR="/var/www/lanshi.xyz"
# LOG_FILE="$BLOG_DIR/deploy_blog.log"
# TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# # 记录日志的函数
# log() {
#   echo "$TIMESTAMP: $@" >> "$LOG_FILE"
# }

# error_log() {
#   echo "$TIMESTAMP: ERROR: $@" >> "$LOG_FILE"
#   echo "ERROR: $@" >&2
# }

# # 清空日志文件 (可选，每次部署都从新的日志开始)
# > "$LOG_FILE"

# # 1) 进入 Hugo 博客本地目录
# cd "$BLOG_DIR" || {
#   error_log "Could not change directory to $BLOG_DIR"
#   exit 1
# }
# log "Changed directory to: $(pwd)"

# # 2) 生成静态文件 (清除缓存并压缩) and capture Hugo's output to commandline & log
# log "Generating Hugo site..."
# hugo --config config.yaml --minify --ignoreCache --cleanDestinationDir 2>&1 | tee -a "$LOG_FILE"
# HUGO_RESULT=$?
# if [ "$HUGO_RESULT" -ne 0 ]; then
#   error_log "Hugo site generation failed! (Exit code: $HUGO_RESULT)"
#   exit 1
# fi
# log "Hugo site generation completed successfully."

# # 3) 使用 rsync 同步文件到目标目录 (更高效且安全)
# log "Syncing files to target directory using rsync..."
# rsync -avz public/ "$TARGET_DIR/" --delete >> "$LOG_FILE" 2>&1
# RSYNC_RESULT=$?
# if [ "$RSYNC_RESULT" -ne 0 ]; then
#   error_log "File synchronization with rsync failed! (Exit code: $RSYNC_RESULT)"
#   exit 1
# fi
# log "Files synced to target directory successfully."

# # 4) 重新加载 Nginx
# log "Reloading Nginx..."
# sudo systemctl reload nginx >> "$LOG_FILE" 2>&1
# NGINX_RELOAD_RESULT=$?
# if [ "$NGINX_RELOAD_RESULT" -ne 0 ]; then
#   error_log "Nginx reload failed! (Exit code: $NGINX_RELOAD_RESULT)"
#   exit 1
# fi
# log "Nginx reloaded successfully."

# echo "Blog updated and Nginx reloaded successfully!"
#