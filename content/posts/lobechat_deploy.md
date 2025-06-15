---
# 核心元数据
author: lanshi
date: "2025-06-15T19:00:00+08:00"
lastmod:
title: "LobeChat 部署 - 服务端"

# 内容控制
draft: false
showToc: true
tocOpen: false
showFullContent: true
summary: 本文介绍了如何在服务器上部署 LobeChat 服务端，包括 SSH 连接、Docker 安装、Postgres 数据库部署、LobeChat 部署以及 Nginx 反向代理配置。
# 内容分类
series:
tags: ["LobeChat", "部署", "Docker", "Postgres", "Nginx"]
categories: ["教程"]

# SEO优化
description: 详细步骤和命令，帮助你在服务器上部署 LobeChat 服务端。
keywords: ["LobeChat", "部署", "Docker", "Postgres", "Nginx"]

# 主题集成
math: true
comment: true
hiddenFromSearch: false
hiddenFromHomePage: false

# 视觉配置
cover:
  image: "img/database-cover.png"
  alt: "LobeChat 部署封面"
  caption: "LobeChat 服务端部署"
  relative: true

# 版权声明
copyright: true
---
## ssh连接

```bash
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

按提示操作，通常会生成两个文件：`id_rsa`（私钥）和`id_rsa.pub`（公钥）。

- **将公钥添加到云服务器**：
  - 登录到你的云服务提供商的控制台。
  - 找到与你的服务器相关的SSH密钥管理部分。
  - 将生成的公钥（`id_rsa.pub`的内容）添加到服务器的授权密钥列表中。
  - 打开终端（或命令提示符）。
  - 运行以下命令，将`username`替换为你的用户名，`server_ip`替换为你的服务器IP地址：

    ```bash
    ssh username@server_ip
    ```

## 安装docker

自动安装docker命令

```bash
curl -fsSL https://test.docker.com -o test-docker.sh
sudo sh test-docker.sh
```

## 部署Postgres数据库

```bash
docker network create pg
docker run --name lobechat-postgres --network pg -e POSTGRES_PASSWORD=密码 -p 服务器端口:docker端口 -d pgvector/pgvector:pg16
```

## 部署lobechat

创建lobe-chat.env文件

```.env
# 网站域名
APP_URL=https://lanshi.space

# DB 必须的环境变量
# 用于加密敏感信息的密钥，可以使用 openssl rand -base64 32 生成
KEY_VAULTS_SECRET='xxxA='
# Postgres 数据库连接字符串
# 格式：postgres://username:password@host:port/dbname，如果你的 pg 实例为 Docker 容器，请使用容器名
DATABASE_URL=postgres://postgres:Yxxx7@lobechat-postgres:5432/postgres

# NEXT_AUTH 相关，可以使用 auth0、Azure AD、GitHub、Authentik、zitadel 等，如有其他接入诉求欢迎提 PR
NEXT_AUTH_SECRET=owB59cmgjyrNcgxxx
NEXT_AUTH_SSO_PROVIDERS=github
NEXTAUTH_URL=https://laxxx
AUTH_GITHUB_ID=Ivxx
AUTH_GITHUB_SECRET=5xx

# S3 相关
S3_ACCESS_KEY_ID=xxx
S3_SECRET_ACCESS_KEY=xxx
# 用于 S3 API 访问的域名
S3_ENDPOINT=https://xxxx
S3_BUCKET=lobechat
# 用于外网访问 S3 的公共域名，需配置 CORS
S3_PUBLIC_DOMAIN=https://xxxx
# S3_REGION=ap-chengdu # 如果需要指定地域

# 其他环境变量，视需求而定
OPENAI_API_KEY=sk-xxx
OPENAI_PROXY_URL=xxx
# ...
```

### 部署命令

```bash
docker run -it -d -p 3210:3210 --network pg --env-file lobe-chat.env --name lobe-chat-database lobehub/lobe-chat-database
```

如果测试失败可以使用以下指令重装:

```bash
docker stop lobechat-postgres lobe-chat-database 
docker rm lobechat-postgres lobe-chat-database 
docker run --name lobechat-postgres \
           --network pg \
           -e POSTGRES_PASSWORD=密码 \
           -v $(pwd)/initdb:/docker-entrypoint-initdb.d \
           -p 5432:5432 \
           -d pgvector/pgvector:pg17

docker run -it -d -p 3210:3210 --network pg --env-file lobe-chat.env --name lobe-chat-database lobehub/lobe-chat-database

```

## nginx反向代理

```nginx
server {
    listen 80;
    server_name lanshi.space www.lanshi.space;

    location / {
        proxy_pass http://localhost:3210;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_cache_bypass $http_upgrade;

        # 确保 Cookies 能够正常传递
        proxy_cookie_domain localhost lanshi.space;
        proxy_cookie_path / "/; HttpOnly; Secure";

        # 允许所有请求头
        proxy_set_header Accept-Encoding "";
        proxy_set_header Accept-Language $http_accept_language;
        proxy_set_header User-Agent $http_user_agent;
        proxy_set_header Referer $http_referer;
    }
}
```

### 编写自动更新脚本

````bash
vim auto-update-lobe-chat.sh
````

```bash
#!/bin/bash
# 更新LobeChat数据库容器 [Author: AI] [Version: 1.0] [Date: 2025-6-15]

set -euo pipefail  # 启用严格错误检查

# 设置环境变量
readonly ENV_FILE="lobe-chat.env"  # 使用大写字母命名变量
readonly LOG_FILE="./log/lobechat_update.log"

# 检查日志目录是否存在，若不存在则创建
mkdir -p "$(dirname "$LOG_FILE")"

# 拉取最新镜像并记录日志
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"
}

log "Pulling latest image for lobehub/lobe-chat..."
if ! output=$(docker pull lobehub/lobe-chat 2>&1); then
  log "Failed to pull image: $output"
  exit 1
fi
log "Image pulled successfully."

# 停止并删除旧容器
log "Stopping and removing old container 'lobe-chat-database'..."
docker stop lobe-chat-database &> /dev/null || true
docker rm lobe-chat-database &> /dev/null || true

# 启动新容器
log "Starting new container 'lobe-chat-database'..."
docker run -d --restart=unless-stopped \
  -p 3210:3210 \
  --network pg \
  --env-file "$ENV_FILE" \
  --name lobe-chat-database \
  lobehub/lobe-chat-database

log "Update completed successfully."

```

```bash
chmod 755 auto-update-lobe-chat.sh
./auto-update-lobe-chat.sh 
```

### 检查日志

```bash
docker logs -f lobe-chat-database
```

### 可能遇到的问题

权限被拒,执行shell并重启

```bash
sudo chmod 666 /var/run/docker.sock
sudo systemctl stop docker.socket
sudo systemctl disable docker.socket
sudo systemctl stop docker
sudo systemctl disable docker
sudo systemctl status docker
sudo systemctl status docker.socket
```

注意:解决问题后请启用服务:

```bash
sudo systemctl enable docker
sudo systemctl start docker
sudo systemctl enable docker.socket
sudo systemctl start docker.socket
```
