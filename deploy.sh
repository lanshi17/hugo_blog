#!/usr/bin/env bash
set -e

# 1) 进入您的 Hugo 博客本地目录(假设为 ~/hugo_blog)
cd ~/hugo_blog

# 2) 生成静态文件，--minify 会压缩文件体积(可选)
hugo  --config config.yaml --minify --ignoreCache

# 3) 清空 Nginx 指向的目标目录(假设为 /var/www/lanshi.xyz)
sudo rm -rf /var/www/lanshi.xyz/*

# 4) 拷贝 Hugo 编译后的 public 文件夹至目标目录
sudo cp -r public/* /var/www/lanshi.xyz/

# 5) 重新加载或重启 Nginx，以应用可能的配置更改(可视需求选择 reload/restart)
sudo systemctl reload nginx

echo "Blog updated and Nginx reloaded successfully!"