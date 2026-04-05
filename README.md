# Hugo Blog

基于 Hugo + PaperMod 的技术博客站点，包含沉浸式阅读优化、站内搜索、评论、统计面板，以及面向文章内容的 AI 助手入口。

## 功能概览

- 沉浸式阅读界面：优化文章页排版、目录、进度条与专注模式
- 文章级 AI 助手：支持基于当前文章内容的提问与回答
- 站内搜索：基于索引文件的前端搜索
- Giscus 评论集成
- Hugo 静态构建与部署脚本

## 环境要求

- Hugo
- Node.js
- npm

## 本地开发

安装依赖：

```bash
npm install
```

启动 Hugo 本地预览：

```bash
hugo server
```

生产构建：

```bash
npm run build
```

说明：

- `npm run build` 会先执行 `./gen-pdfinfo.sh`
- 然后运行 `hugo --minify`

## AI 文章助手

站点已集成文章页 AI 浮窗助手。默认推荐使用“代理模式”，避免在前端暴露密钥。

当前前端配置位于：

- [config.yaml](/home/dave_paine/hugo_blog/config.yaml#L196)

默认配置示例：

```yaml
params:
  aiAssistant:
    enabled: true
    endpoint: "/api/blog-assistant"
    requestFormat: "openai-chat"
    model: "gpt-4.1-mini"
    models:
      - "gpt-4.1-mini"
      - "gpt-4.1"
```

### 推荐接法：代理模式

服务端代理脚本：

- [server/ai-assistant-proxy.mjs](/home/dave_paine/hugo_blog/server/ai-assistant-proxy.mjs)

启动示例：

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4.1-mini"
npm run ai-proxy
```

代理默认监听：

```text
http://127.0.0.1:8787/api/blog-assistant
```

如果你使用 Nginx，可将站点同域的 `/api/blog-assistant` 反代到本机代理：

- [docs/ai-assistant-nginx.example.conf](/home/dave_paine/hugo_blog/docs/ai-assistant-nginx.example.conf)

完整说明见：

- [docs/ai-assistant-openai-compatible.md](/home/dave_paine/hugo_blog/docs/ai-assistant-openai-compatible.md)

### 不推荐接法：前端直连

项目保留了前端直连远程 OpenAI 兼容接口的能力，但只有在显式开启时才会注入密钥。

这不适合公开站点，因为：

- API Key 会暴露到浏览器
- 访客可以在源码或网络请求中看到密钥

如果你后续要推送到 GitHub 或公开部署，不要启用 `directRemote.exposeApiKeyInBrowser: true`。

## 配置说明

主要站点配置文件：

- [config.yaml](/home/dave_paine/hugo_blog/config.yaml)

与这次界面和 AI 功能相关的主要文件：

- [layouts/partials/extend_head.html](/home/dave_paine/hugo_blog/layouts/partials/extend_head.html)
- [layouts/_default/list.html](/home/dave_paine/hugo_blog/layouts/_default/list.html)
- [layouts/partials/home_info.html](/home/dave_paine/hugo_blog/layouts/partials/home_info.html)
- [assets/js/immersive-reading.js](/home/dave_paine/hugo_blog/assets/js/immersive-reading.js)
- [assets/js/ai-assistant.js](/home/dave_paine/hugo_blog/assets/js/ai-assistant.js)
- [assets/css/extended/zz-reading-refresh.css](/home/dave_paine/hugo_blog/assets/css/extended/zz-reading-refresh.css)
- [assets/css/extended/zzz-ai-assistant.css](/home/dave_paine/hugo_blog/assets/css/extended/zzz-ai-assistant.css)

## 部署

项目包含部署脚本：

- [deploy.sh](/home/dave_paine/hugo_blog/deploy.sh)

如果你已经在服务器上配置了 Hugo 构建目录和 Nginx，可按脚本参数执行部署。

## 安全提醒

- 不要把真实 API Key 写进仓库文件
- 推荐把密钥放在服务端环境变量，例如 `OPENAI_API_KEY`
- 公开部署时优先使用代理模式，而不是前端直连
