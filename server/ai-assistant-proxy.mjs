import http from 'node:http';

const PORT = Number(process.env.PORT || 8787);
const OPENAI_BASE_URL = (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/+$/, '');
const OPENAI_CHAT_PATH = process.env.OPENAI_CHAT_PATH || '/chat/completions';
const OPENAI_EMBEDDING_PATH = process.env.OPENAI_EMBEDDING_PATH || '/embeddings';
const OPENAI_RERANK_PATH = process.env.OPENAI_RERANK_PATH || '/rerank';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4.1-mini';
const OPENAI_EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || 'baai/bge-m3(free)';
const OPENAI_RERANK_MODEL = process.env.OPENAI_RERANK_MODEL || 'BAAI/bge-reranker-v2-m3(free)';
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || '*';
const MAX_BODY_BYTES = Number(process.env.MAX_BODY_BYTES || 256 * 1024);
const ALLOWED_MODELS = (process.env.ALLOWED_MODELS || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
const ALLOWED_EMBEDDING_MODELS = (process.env.ALLOWED_EMBEDDING_MODELS || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
const ALLOWED_RERANK_MODELS = (process.env.ALLOWED_RERANK_MODELS || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);

if (!OPENAI_API_KEY) {
    console.error('[ai-proxy] Missing OPENAI_API_KEY');
    process.exit(1);
}

function writeJson(response, statusCode, payload) {
    response.writeHead(statusCode, {
        'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'no-store'
    });
    response.end(JSON.stringify(payload));
}

function createMessages(body) {
    if (Array.isArray(body.messages) && body.messages.length) {
        return body.messages;
    }

    const systemPrompt = body.systemPrompt || '你是技术博客的文章问答助手。';
    const article = body.article || {};
    const context = Array.isArray(body.context) ? body.context : [];
    const question = body.question || '';

    const contextText = context
        .map((chunk, index) => {
            const heading = chunk && chunk.heading ? `章节：${chunk.heading}\n` : '';
            const text = chunk && chunk.text ? chunk.text : '';
            return `[片段 ${index + 1}]\n${heading}${text}`;
        })
        .join('\n\n');

    const userPrompt = [
        article.title ? `文章标题：${article.title}` : '',
        article.description ? `文章简介：${article.description}` : '',
        article.url ? `页面地址：${article.url}` : '',
        contextText ? `\n文章相关片段：\n${contextText}` : '',
        question ? `\n问题：${question}` : ''
    ]
        .filter(Boolean)
        .join('\n');

    return [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
    ];
}

function pickModel(body, defaultModel, allowedModels = []) {
    const requestedModel = typeof body.model === 'string' && body.model.trim()
        ? body.model.trim()
        : defaultModel;

    if (allowedModels.length && !allowedModels.includes(requestedModel)) {
        return null;
    }

    return requestedModel;
}

async function readRequestBody(request) {
    const chunks = [];
    let size = 0;

    for await (const chunk of request) {
        size += chunk.length;
        if (size > MAX_BODY_BYTES) {
            throw new Error('Request body too large');
        }
        chunks.push(chunk);
    }

    const raw = Buffer.concat(chunks).toString('utf8').trim();
    if (!raw) {
        return {};
    }

    return JSON.parse(raw);
}

async function forwardUpstream(upstreamPath, payload) {
    const upstream = await fetch(`${OPENAI_BASE_URL}${upstreamPath}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${OPENAI_API_KEY}`
        },
        body: JSON.stringify(payload)
    });

    const text = await upstream.text();
    let data = {};

    try {
        data = text ? JSON.parse(text) : {};
    } catch (error) {
        data = {
            error: 'Invalid upstream response',
            raw: text
        };
    }

    return {
        status: upstream.status,
        data
    };
}

const server = http.createServer(async (request, response) => {
    if (request.method === 'OPTIONS') {
        response.writeHead(204, {
            'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '86400'
        });
        response.end();
        return;
    }

    if (request.method !== 'POST' || !['/api/blog-assistant', '/api/blog-search-embedding', '/api/blog-search-rerank'].includes(request.url)) {
        writeJson(response, 404, {
            error: 'Not found',
            message: 'Use POST /api/blog-assistant or POST /api/blog-search-embedding or POST /api/blog-search-rerank'
        });
        return;
    }

    try {
        const body = await readRequestBody(request);
        if (request.url === '/api/blog-search-embedding') {
            const model = pickModel(body, OPENAI_EMBEDDING_MODEL, ALLOWED_EMBEDDING_MODELS);
            if (!model) {
                writeJson(response, 400, {
                    error: 'Invalid model',
                    message: 'Requested model is not in ALLOWED_EMBEDDING_MODELS'
                });
                return;
            }

            const input = body.input ?? body.query;
            if (!(typeof input === 'string' && input.trim()) && !(Array.isArray(input) && input.length)) {
                writeJson(response, 400, {
                    error: 'Invalid input',
                    message: 'Embedding request requires a non-empty input'
                });
                return;
            }

            const upstreamPayload = {
                model,
                input
            };

            if (typeof body.dimensions === 'number') {
                upstreamPayload.dimensions = body.dimensions;
            }

            if (typeof body.encoding_format === 'string' && body.encoding_format.trim()) {
                upstreamPayload.encoding_format = body.encoding_format.trim();
            }

            const upstream = await forwardUpstream(OPENAI_EMBEDDING_PATH, upstreamPayload);
            writeJson(response, upstream.status, upstream.data);
            return;
        }

        if (request.url === '/api/blog-search-rerank') {
            const model = pickModel(body, OPENAI_RERANK_MODEL, ALLOWED_RERANK_MODELS);
            if (!model) {
                writeJson(response, 400, {
                    error: 'Invalid model',
                    message: 'Requested model is not in ALLOWED_RERANK_MODELS'
                });
                return;
            }

            const query = typeof body.query === 'string' ? body.query.trim() : '';
            const documents = Array.isArray(body.documents) ? body.documents.filter((item) => typeof item === 'string' && item.trim()) : [];
            if (!query || !documents.length) {
                writeJson(response, 400, {
                    error: 'Invalid input',
                    message: 'Rerank request requires query and a non-empty documents array'
                });
                return;
            }

            const upstreamPayload = {
                model,
                query,
                documents
            };

            if (typeof body.top_n === 'number') {
                upstreamPayload.top_n = body.top_n;
            }

            if (typeof body.return_documents === 'boolean') {
                upstreamPayload.return_documents = body.return_documents;
            }

            const upstream = await forwardUpstream(OPENAI_RERANK_PATH, upstreamPayload);
            writeJson(response, upstream.status, upstream.data);
            return;
        }

        const model = pickModel(body, OPENAI_MODEL, ALLOWED_MODELS);
        if (!model) {
            writeJson(response, 400, {
                error: 'Invalid model',
                message: 'Requested model is not in ALLOWED_MODELS'
            });
            return;
        }

        const upstreamPayload = {
            model,
            temperature: typeof body.temperature === 'number' ? body.temperature : 0.2,
            messages: createMessages(body)
        };

        const upstream = await forwardUpstream(OPENAI_CHAT_PATH, upstreamPayload);
        writeJson(response, upstream.status, upstream.data);
    } catch (error) {
        writeJson(response, 500, {
            error: 'Proxy request failed',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});

server.listen(PORT, () => {
    console.log(`[ai-proxy] Listening on http://127.0.0.1:${PORT}/api/blog-assistant`);
    console.log(`[ai-proxy] Embedding endpoint on http://127.0.0.1:${PORT}/api/blog-search-embedding`);
    console.log(`[ai-proxy] Rerank endpoint on http://127.0.0.1:${PORT}/api/blog-search-rerank`);
    console.log(`[ai-proxy] Forwarding to ${OPENAI_BASE_URL}${OPENAI_CHAT_PATH}`);
    console.log(`[ai-proxy] Embedding forward target: ${OPENAI_BASE_URL}${OPENAI_EMBEDDING_PATH}`);
    console.log(`[ai-proxy] Rerank forward target: ${OPENAI_BASE_URL}${OPENAI_RERANK_PATH}`);
    console.log(`[ai-proxy] Default model: ${OPENAI_MODEL}`);
    console.log(`[ai-proxy] Default embedding model: ${OPENAI_EMBEDDING_MODEL}`);
    console.log(`[ai-proxy] Default rerank model: ${OPENAI_RERANK_MODEL}`);
    if (ALLOWED_MODELS.length) {
        console.log(`[ai-proxy] Allowed models: ${ALLOWED_MODELS.join(', ')}`);
    }
    if (ALLOWED_EMBEDDING_MODELS.length) {
        console.log(`[ai-proxy] Allowed embedding models: ${ALLOWED_EMBEDDING_MODELS.join(', ')}`);
    }
    if (ALLOWED_RERANK_MODELS.length) {
        console.log(`[ai-proxy] Allowed rerank models: ${ALLOWED_RERANK_MODELS.join(', ')}`);
    }
});
