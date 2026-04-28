import fs from 'node:fs/promises';
import path from 'node:path';

const OUTPUT_DIR = process.env.SEARCH_OUTPUT_DIR || process.env.HUGO_PUBLIC_DIR || path.resolve(process.cwd(), 'public');
const SOURCE_FILE = process.env.SEARCH_SOURCE_FILE || path.join(OUTPUT_DIR, 'index.json');
const OUTPUT_FILE = process.env.SEARCH_VECTOR_OUTPUT_FILE || path.join(OUTPUT_DIR, 'search-vectors.json');
const OPENAI_BASE_URL = (process.env.OPENAI_BASE_URL || process.env.AI_SEARCH_BASE_URL || 'https://api.openai.com/v1').replace(/\/+$/, '');
const OPENAI_EMBEDDING_PATH = process.env.OPENAI_EMBEDDING_PATH || process.env.AI_SEARCH_EMBEDDING_PATH || '/embeddings';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || process.env.AI_SEARCH_API_KEY || '';
const CUSTOM_OPENAI_EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || process.env.AI_SEARCH_EMBEDDING_MODEL || '';
const OPENAI_EMBEDDING_MODEL = CUSTOM_OPENAI_EMBEDDING_MODEL || 'baai/bge-m3(free)';
const SEARCH_EMBEDDING_BATCH_SIZE = Math.max(1, Number(process.env.SEARCH_EMBEDDING_BATCH_SIZE || process.env.AI_SEARCH_EMBEDDING_BATCH_SIZE || 16));
const SEARCH_EMBEDDING_MAX_CHARS = Math.max(500, Number(process.env.SEARCH_EMBEDDING_MAX_CHARS || process.env.AI_SEARCH_EMBEDDING_MAX_CHARS || 6000));
const SEARCH_PREVIEW_MAX_CHARS = Math.max(200, Number(process.env.SEARCH_PREVIEW_MAX_CHARS || process.env.AI_SEARCH_PREVIEW_MAX_CHARS || 1600));

function log(message) {
    console.log(`[search-embeddings] ${message}`);
}

function buildProviderHint(message) {
    if (CUSTOM_OPENAI_EMBEDDING_MODEL) {
        return '';
    }

    if (typeof message === 'string' && /model|channel/i.test(message)) {
        return ' 当前提供商可能不支持默认的 baai/bge-m3(free)，请设置 OPENAI_EMBEDDING_MODEL 或 AI_SEARCH_EMBEDDING_MODEL 为该服务商可用的 embedding 模型。';
    }

    return '';
}

async function removeOutputFile() {
    await fs.rm(OUTPUT_FILE, { force: true }).catch(() => {});
}

function normalizeText(text) {
    if (typeof text !== 'string') {
        return '';
    }
    return text.replace(/\s+/g, ' ').trim();
}

function clipText(text, maxChars) {
    return normalizeText(text).slice(0, maxChars);
}

function buildEmbeddingInput(item) {
    return [
        item.title ? `标题：${item.title}` : '',
        item.summary ? `摘要：${item.summary}` : '',
        item.content ? `正文：${item.content}` : ''
    ]
        .filter(Boolean)
        .join('\n\n')
        .slice(0, SEARCH_EMBEDDING_MAX_CHARS);
}

function normalizeVector(values) {
    if (!Array.isArray(values) || !values.length) {
        return null;
    }

    let magnitude = 0;
    for (const rawValue of values) {
        const value = Number(rawValue);
        if (!Number.isFinite(value)) {
            return null;
        }
        magnitude += value * value;
    }

    if (!magnitude) {
        return null;
    }

    const divisor = Math.sqrt(magnitude);
    return values.map((value) => Number((Number(value) / divisor).toFixed(6)));
}

function pickEmbeddingValues(item) {
    if (Array.isArray(item)) {
        return item;
    }

    if (!item || typeof item !== 'object') {
        return null;
    }

    return item.embedding || item.vector || item.values || null;
}

function sortEmbeddingData(data) {
    return data
        .slice()
        .sort((left, right) => {
            const leftIndex = Number(left && left.index);
            const rightIndex = Number(right && right.index);

            if (!Number.isFinite(leftIndex) || !Number.isFinite(rightIndex)) {
                return 0;
            }

            return leftIndex - rightIndex;
        });
}

async function loadSourceDocuments() {
    const raw = await fs.readFile(SOURCE_FILE, 'utf8');
    const payload = JSON.parse(raw);

    if (!Array.isArray(payload)) {
        throw new Error('搜索源索引不是数组');
    }

    return payload
        .map((item) => ({
            title: clipText(item && item.title, 200),
            permalink: normalizeText(item && item.permalink),
            summary: clipText(item && item.summary, 400),
            content: clipText(item && item.content, SEARCH_PREVIEW_MAX_CHARS),
            embeddingInput: buildEmbeddingInput({
                title: clipText(item && item.title, 200),
                summary: clipText(item && item.summary, 800),
                content: clipText(item && item.content, SEARCH_EMBEDDING_MAX_CHARS)
            })
        }))
        .filter((item) => item.permalink && item.embeddingInput);
}

async function requestEmbeddings(inputs) {
    const response = await fetch(`${OPENAI_BASE_URL}${OPENAI_EMBEDDING_PATH}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${OPENAI_API_KEY}`
        },
        body: JSON.stringify({
            model: OPENAI_EMBEDDING_MODEL,
            input: inputs
        })
    });

    const text = await response.text();
    let payload = {};

    try {
        payload = text ? JSON.parse(text) : {};
    } catch (error) {
        throw new Error(`上游返回了无法解析的响应: ${text.slice(0, 200)}`);
    }

    if (!response.ok) {
        const message = payload && (payload.message || (payload.error && payload.error.message) || payload.error);
        throw new Error(message || `Embedding 请求失败: HTTP ${response.status}`);
    }

    const embeddings = Array.isArray(payload.data)
        ? sortEmbeddingData(payload.data).map(pickEmbeddingValues)
        : Array.isArray(payload.embeddings)
            ? payload.embeddings.map(pickEmbeddingValues)
            : [];

    if (embeddings.length !== inputs.length) {
        throw new Error('Embedding 返回数量与输入数量不一致');
    }

    return embeddings;
}

async function writeVectorIndex(documents, dimensions) {
    const payload = {
        version: 1,
        model: OPENAI_EMBEDDING_MODEL,
        dimensions,
        generatedAt: new Date().toISOString(),
        documents
    };

    await fs.mkdir(path.dirname(OUTPUT_FILE), { recursive: true });
    await fs.writeFile(OUTPUT_FILE, JSON.stringify(payload), 'utf8');
}

async function main() {
    if (!OPENAI_API_KEY) {
        log('未设置 OPENAI_API_KEY，跳过向量索引生成，搜索页将自动降级到关键词搜索。');
        await removeOutputFile();
        return;
    }

    let documents;

    try {
        documents = await loadSourceDocuments();
    } catch (error) {
        log(`未找到可用的搜索源索引: ${error instanceof Error ? error.message : '未知错误'}`);
        await removeOutputFile();
        return;
    }

    if (!documents.length) {
        log('搜索源索引为空，跳过向量索引生成。');
        await removeOutputFile();
        return;
    }

    const outputDocuments = [];
    let dimensions = 0;

    try {
        for (let index = 0; index < documents.length; index += SEARCH_EMBEDDING_BATCH_SIZE) {
            const batch = documents.slice(index, index + SEARCH_EMBEDDING_BATCH_SIZE);
            const batchNumber = Math.floor(index / SEARCH_EMBEDDING_BATCH_SIZE) + 1;
            const totalBatches = Math.ceil(documents.length / SEARCH_EMBEDDING_BATCH_SIZE);

            log(`生成第 ${batchNumber}/${totalBatches} 批文档向量...`);
            const embeddings = await requestEmbeddings(batch.map((item) => item.embeddingInput));

            batch.forEach((item, batchIndex) => {
                const normalizedEmbedding = normalizeVector(embeddings[batchIndex]);
                if (!normalizedEmbedding) {
                    return;
                }

                if (!dimensions) {
                    dimensions = normalizedEmbedding.length;
                }

                if (normalizedEmbedding.length !== dimensions) {
                    return;
                }

                outputDocuments.push({
                    title: item.title,
                    permalink: item.permalink,
                    summary: item.summary,
                    content: item.content,
                    embedding: normalizedEmbedding
                });
            });
        }

        if (!outputDocuments.length || !dimensions) {
            throw new Error('没有生成任何有效文档向量');
        }

        await writeVectorIndex(outputDocuments, dimensions);
        log(`向量索引生成完成，共 ${outputDocuments.length} 条记录，输出到 ${OUTPUT_FILE}`);
    } catch (error) {
        const message = error instanceof Error ? error.message : '未知错误';
        log(`向量索引生成失败，将保留关键词降级路径: ${message}${buildProviderHint(message)}`);
        await removeOutputFile();
    }
}

await main();
