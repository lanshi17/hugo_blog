import http from 'node:http';

const PORT = Number(process.env.PORT || 8787);
const OPENAI_BASE_URL = (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/+$/, '');
const OPENAI_CHAT_PATH = process.env.OPENAI_CHAT_PATH || '/chat/completions';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4.1-mini';
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || '*';
const MAX_BODY_BYTES = Number(process.env.MAX_BODY_BYTES || 256 * 1024);
const ALLOWED_MODELS = (process.env.ALLOWED_MODELS || '')
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

function pickModel(body) {
    const requestedModel = typeof body.model === 'string' && body.model.trim()
        ? body.model.trim()
        : OPENAI_MODEL;

    if (ALLOWED_MODELS.length && !ALLOWED_MODELS.includes(requestedModel)) {
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

    if (request.method !== 'POST' || request.url !== '/api/blog-assistant') {
        writeJson(response, 404, {
            error: 'Not found',
            message: 'Use POST /api/blog-assistant'
        });
        return;
    }

    try {
        const body = await readRequestBody(request);
        const model = pickModel(body);

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

        const upstream = await fetch(`${OPENAI_BASE_URL}${OPENAI_CHAT_PATH}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${OPENAI_API_KEY}`
            },
            body: JSON.stringify(upstreamPayload)
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

        writeJson(response, upstream.status, data);
    } catch (error) {
        writeJson(response, 500, {
            error: 'Proxy request failed',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});

server.listen(PORT, () => {
    console.log(`[ai-proxy] Listening on http://127.0.0.1:${PORT}/api/blog-assistant`);
    console.log(`[ai-proxy] Forwarding to ${OPENAI_BASE_URL}${OPENAI_CHAT_PATH}`);
    console.log(`[ai-proxy] Default model: ${OPENAI_MODEL}`);
    if (ALLOWED_MODELS.length) {
        console.log(`[ai-proxy] Allowed models: ${ALLOWED_MODELS.join(', ')}`);
    }
});
