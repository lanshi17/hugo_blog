# AI 文章助手接入 OpenAI 兼容接口

这个博客助手现在支持两种接法：

1. 前端直连远程 OpenAI 兼容接口
2. 通过服务端代理转发到 OpenAI 兼容接口

如果你坚持把 `apiKey` 统一配在 `config.yaml`，那就是“前端直连”模式。

这能用，但有一个明确代价：

- `apiKey` 会被注入到浏览器页面里
- 所有访客都能在页面源码或网络请求里看到它
- 不适合生产环境

如果只是自己内网用、临时演示、或无敏感额度约束，可以这样配；正式上线仍然建议用代理。

## 推荐做法

如果你要推 GitHub 或上线公开站点，默认就用代理模式：

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

然后：

1. 在服务端环境变量里放 `OPENAI_API_KEY`
2. 运行 `npm run ai-proxy`
3. 用 Nginx 把 `/api/blog-assistant` 反代到本机 `127.0.0.1:8787`

Nginx 示例见：

- `[ai-assistant-nginx.example.conf](/home/dave_paine/hugo_blog/docs/ai-assistant-nginx.example.conf)`

如果你使用项目自带的 `[deploy.sh](/home/dave_paine/hugo_blog/deploy.sh)`，部署流程会一并安装/更新 AI 代理的 systemd 服务并自动重启。

## 开发环境 endpoint 覆盖

如果你本地跑的是：

- Hugo 预览：`http://127.0.0.1:1313`
- AI 代理：`http://127.0.0.1:8787`

那前端默认写死成相对路径 `/api/blog-assistant` 并不一定能打到代理，除非你本地也做了反代。

现在项目支持用环境变量覆盖前端注入的接口地址：

```bash
export AI_ASSISTANT_ENDPOINT="http://127.0.0.1:8787/api/blog-assistant"
hugo server
```

如果你要从手机、平板或局域网其他设备访问开发站点，可以改成宿主机 IP：

```bash
export AI_ASSISTANT_ENDPOINT="http://192.168.1.20:8787/api/blog-assistant"
hugo server --bind 0.0.0.0 --baseURL http://192.168.1.20:1313
```

如果不覆盖 `AI_ASSISTANT_ENDPOINT`，开发页在 `127.0.0.1` 或 `localhost` 下会先尝试把相对路径 `/api/blog-assistant` 改写到本地代理：

```text
http://127.0.0.1:8787/api/blog-assistant
```

这个本地代理根地址可以用 `AI_ASSISTANT_LOCAL_PROXY_BASE_URL` 覆盖。

覆盖优先级：

1. `AI_ASSISTANT_ENDPOINT`
2. `config.yaml` 中的 `params.aiAssistant.endpoint`

对应注入逻辑见：

- `[layouts/partials/extend_head.html](/home/dave_paine/hugo_blog/layouts/partials/extend_head.html#L45)`

## 1. 用 `.zshrc` 存明文密钥，`config.yaml` 只配变量名

先在 `~/.zshrc` 里导出：

```bash
export OPENAI_API_KEY="sk-..."
```

如果你改了变量名，比如：

```bash
export MY_BLOG_LLM_KEY="sk-..."
```

那后面在 `config.yaml` 里把 `apiKeyEnv` 改成对应名字即可。

注意：这个项目已经在 Hugo 安全策略里放行了 `OPENAI_` 和 `AI_ASSISTANT_` 前缀的环境变量。

也就是说，推荐你把变量命名成：

- `OPENAI_API_KEY`
- `AI_ASSISTANT_API_KEY`

如果你改成其他完全不相关的名字，Hugo 构建时会因为 `getenv` 安全策略失败。

## 2. 在 config 里配置远程接口

在 `[config.yaml](/home/dave_paine/hugo_blog/config.yaml#L196)` 里配置：

```yaml
params:
  aiAssistant:
    enabled: true
    requestFormat: "openai-chat"
    model: "gpt-4.1-mini"
    models:
      - "gpt-4.1-mini"
      - "gpt-4.1"
      - "o4-mini"
    directRemote:
      enabled: true
      baseURL: "https://api.openai.com/v1"
      apiPath: "/chat/completions"
      apiKeyEnv: "OPENAI_API_KEY"
      apiKeyHeader: "Authorization"
      apiKeyPrefix: "Bearer "
      exposeApiKeyInBrowser: true
```

说明：

- `baseURL`：OpenAI 兼容服务根地址
- `apiPath`：聊天接口路径，默认 `/chat/completions`
- `apiKeyEnv`：从环境变量读取密钥，例如 `OPENAI_API_KEY`
- `apiKey`：也支持直接写死，但不推荐
- `apiKeyHeader`：认证头名称，默认 `Authorization`
- `apiKeyPrefix`：认证头前缀，默认 `Bearer `
- `model`：默认模型
- `models`：前端下拉可选模型
- `exposeApiKeyInBrowser`：必须显式设为 `true` 才允许前端直连

如果你接的是其他 OpenAI 兼容服务，只需要改：

```yaml
baseURL: "https://your-provider.example.com/v1"
```

如果该服务不是标准 `/chat/completions`，再改：

```yaml
apiPath: "/v1/chat/completions"
```

构建时 Hugo 会在模板里通过环境变量读取密钥，再注入前端配置。

相关注入逻辑在：

- `[layouts/partials/extend_head.html](/home/dave_paine/hugo_blog/layouts/partials/extend_head.html#L45)`

## 3. 前端现在会怎么请求

前端直连时，会直接向：

`baseURL + apiPath`

发送请求，格式是 OpenAI 兼容 Chat Completions：

```json
{
  "model": "gpt-4.1-mini",
  "temperature": 0.2,
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." }
  ]
}
```

相关逻辑在：

- `[layouts/partials/extend_head.html](/home/dave_paine/hugo_blog/layouts/partials/extend_head.html#L45)`
- `[assets/js/ai-assistant.js](/home/dave_paine/hugo_blog/assets/js/ai-assistant.js#L114)`
- `[assets/js/ai-assistant.js](/home/dave_paine/hugo_blog/assets/js/ai-assistant.js#L348)`

## 4. 如果你还是想用代理

代理模式仍然保留，适合生产环境。

### 启动代理

仓库已提供一个最小代理：

`[server/ai-assistant-proxy.mjs](/home/dave_paine/hugo_blog/server/ai-assistant-proxy.mjs)`

启动示例：

```bash
OPENAI_BASE_URL="https://api.openai.com/v1" \
OPENAI_API_KEY="sk-..." \
OPENAI_MODEL="gpt-4.1-mini" \
ALLOWED_MODELS="gpt-4.1-mini,gpt-4.1,o4-mini" \
PORT="8787" \
npm run ai-proxy
```

如果你接的是其他 OpenAI 兼容服务，只需要替换 `OPENAI_BASE_URL`，例如：

```bash
OPENAI_BASE_URL="https://your-provider.example.com/v1"
```

默认代理转发到：

`$OPENAI_BASE_URL/chat/completions`

如果你的兼容服务路径不同，可以再设置：

```bash
OPENAI_CHAT_PATH="/v1/chat/completions"
```

### Hugo 前端配置

在 `[config.yaml](/home/dave_paine/hugo_blog/config.yaml#L196)` 里配置：

```yaml
params:
  aiAssistant:
    enabled: true
    endpoint: "http://127.0.0.1:8787/api/blog-assistant"
    requestFormat: "openai-chat"
    model: "gpt-4.1-mini"
    models:
      - "gpt-4.1-mini"
      - "gpt-4.1"
      - "o4-mini"
```

说明：

- `endpoint`：你的代理地址，不是 OpenAI 官方地址
- `model`：默认模型
- `models`：前端可选模型列表
- `requestFormat`：当前使用 `openai-chat`，即 OpenAI 兼容的 Chat Completions 格式

## 5. 生产环境建议

- 不要在生产环境把 `apiKey` 直接写进静态站点配置
- 用反向代理或独立服务部署 `server/ai-assistant-proxy.mjs`
- 用 `ALLOWED_MODELS` 限制前端可切换模型，避免任意透传
- 用 `ALLOWED_ORIGIN` 限制允许访问代理的站点域名

例如：

```bash
ALLOWED_ORIGIN="https://blog.lanshi.site"
```

## 6. 如果你要接 Azure / One API / New API / OpenRouter

原则一样，只要该服务兼容 OpenAI Chat Completions：

- 直连模式：改 `directRemote.baseURL`、`model`、`models`
- 代理模式：改 `OPENAI_BASE_URL`、`OPENAI_MODEL`
- 如有必要改 `apiPath` 或 `OPENAI_CHAT_PATH`

如果某个服务不是完全兼容 `/chat/completions`，那就需要改代理脚本的转发逻辑，而不是直接改前端。
