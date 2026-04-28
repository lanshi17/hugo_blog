(function () {
    'use strict';

    const config = window.aiArticleAssistantConfig || {};
    const article = document.querySelector('.post-single');
    const content = article ? article.querySelector('.post-content') : null;

    if (!config.enabled || !article || !content) {
        return;
    }

    const STORAGE_KEY = 'ai-article-assistant-open';
    const MODEL_STORAGE_KEY = 'ai-article-assistant-model';
    const POSITION_STORAGE_KEY = 'ai-article-assistant-position';
    const DRAG_THRESHOLD = 6;
    const HOVER_CLOSE_DELAY = 160;
    const VIEWPORT_GAP = 18;
    const INPUT_MIN_HEIGHT = 60;
    const INPUT_MAX_HEIGHT = 160;
    const STREAM_DONE_MARKER = '[DONE]';
    const DEFAULT_LOCAL_PROXY_BASE_URL = 'http://127.0.0.1:8787';
    const SUMMARY_PATTERN = /(总结|概括|概述|摘要|主要讲|讲了什么|说了什么|要点|重点|总览|梳理)/i;
    const QUESTION_PATTERN = /[?？]|(为什么|如何|怎么|哪些|哪里|区别|作用|含义|定义|流程|步骤|原理|代码)/i;

    function normalizeWhitespace(value) {
        return (value || '').replace(/\s+/g, ' ').trim();
    }

    function joinUrl(baseURL, path) {
        const safeBase = normalizeWhitespace(baseURL).replace(/\/+$/, '');
        const safePath = `/${normalizeWhitespace(path || '').replace(/^\/+/, '')}`;
        return `${safeBase}${safePath}`;
    }

    function isLocalPreview() {
        return ['127.0.0.1', 'localhost'].includes(window.location.hostname);
    }

    function buildLocalProxyUrl(path) {
        const base = normalizeWhitespace(config.localProxyBaseUrl || DEFAULT_LOCAL_PROXY_BASE_URL).replace(/\/+$/, '');
        if (!base || !path.startsWith('/')) {
            return '';
        }

        return `${base}${path}`;
    }

    function uniqueStrings(values) {
        return values.filter((value, index, array) => value && array.indexOf(value) === index);
    }

    function truncate(value, maxLength) {
        const text = normalizeWhitespace(value);
        if (text.length <= maxLength) {
            return text;
        }
        return `${text.slice(0, maxLength).trim()}…`;
    }

    function clampNumber(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function readPreference() {
        try {
            return window.localStorage.getItem(STORAGE_KEY);
        } catch (error) {
            return null;
        }
    }

    function writePreference(value) {
        try {
            window.localStorage.setItem(STORAGE_KEY, value);
        } catch (error) {
            /* noop */
        }
    }

    function readPosition() {
        try {
            const raw = window.localStorage.getItem(POSITION_STORAGE_KEY);
            if (!raw) {
                return null;
            }

            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed.x !== 'number' || typeof parsed.y !== 'number') {
                return null;
            }

            return {
                x: parsed.x,
                y: parsed.y
            };
        } catch (error) {
            return null;
        }
    }

    function writePosition(position) {
        try {
            window.localStorage.setItem(POSITION_STORAGE_KEY, JSON.stringify(position));
        } catch (error) {
            /* noop */
        }
    }

    function getAvailableModels() {
        const configuredModels = Array.isArray(config.models)
            ? config.models.map((item) => normalizeWhitespace(item)).filter(Boolean)
            : [];

        if (configuredModels.length) {
            return configuredModels;
        }

        const defaultModel = normalizeWhitespace(config.model);
        return defaultModel ? [defaultModel] : [];
    }

    function readSelectedModel() {
        const availableModels = getAvailableModels();
        const storedModel = normalizeWhitespace(
            (() => {
                try {
                    return window.localStorage.getItem(MODEL_STORAGE_KEY) || '';
                } catch (error) {
                    return '';
                }
            })()
        );

        if (storedModel && availableModels.includes(storedModel)) {
            return storedModel;
        }

        const configuredDefault = normalizeWhitespace(config.model);
        if (configuredDefault && availableModels.includes(configuredDefault)) {
            return configuredDefault;
        }

        return availableModels[0] || '';
    }

    function writeSelectedModel(value) {
        try {
            window.localStorage.setItem(MODEL_STORAGE_KEY, value);
        } catch (error) {
            /* noop */
        }
    }

    function getRemoteClientConfig() {
        const endpoint = normalizeWhitespace(config.endpoint);
        if (endpoint) {
            const urls = [endpoint];
            if (isLocalPreview() && endpoint.startsWith('/')) {
                urls.unshift(buildLocalProxyUrl(endpoint));
            }

            return {
                enabled: true,
                mode: 'endpoint',
                url: endpoint,
                urls: uniqueStrings(urls),
                headers: {
                    'Content-Type': 'application/json'
                },
                statusText: 'AI 代理模式',
                actionHint: '结合当前文章内容作答',
                warningText: ''
            };
        }

        const remote = config.remote || {};
        const baseURL = normalizeWhitespace(remote.baseURL);
        const apiPath = normalizeWhitespace(remote.apiPath || '/chat/completions') || '/chat/completions';
        const apiKey = normalizeWhitespace(remote.apiKey);
        const apiKeyHeader = normalizeWhitespace(remote.apiKeyHeader || 'Authorization') || 'Authorization';
        const apiKeyPrefix = typeof remote.apiKeyPrefix === 'string' ? remote.apiKeyPrefix : 'Bearer ';
        const allowBrowserKey = Boolean(remote.exposeApiKeyInBrowser);

        if (!remote.enabled || !baseURL || !apiKey || !allowBrowserKey) {
            return {
                enabled: false,
                mode: 'local',
                url: '',
                urls: [],
                headers: {},
                statusText: '文章检索模式',
                actionHint: '仅基于当前文章内容',
                warningText: remote.enabled && baseURL && apiKey && !allowBrowserKey
                    ? '已检测到远程接口配置，但未开启前端暴露密钥开关，当前仍使用文章检索模式。'
                    : ''
            };
        }

        const headers = {
            'Content-Type': 'application/json'
        };
        headers[apiKeyHeader] = `${apiKeyPrefix}${apiKey}`;

        return {
            enabled: true,
            mode: 'direct',
            url: joinUrl(baseURL, apiPath),
            urls: [joinUrl(baseURL, apiPath)],
            headers,
            statusText: '前端直连模式',
            actionHint: '结合文章内容调用远程模型',
            warningText: '当前配置为前端直连，API Key 会暴露给所有访客，不适合生产环境。'
        };
    }

    const remoteClientConfig = getRemoteClientConfig();

    function extractTerms(question) {
        const normalized = normalizeWhitespace(question).toLowerCase();
        const terms = new Set();

        (normalized.match(/[a-z0-9_-]{2,}/g) || []).forEach((term) => terms.add(term));
        (normalized.match(/[\u4e00-\u9fff]{2,}/g) || []).forEach((term) => {
            terms.add(term);
            if (term.length > 4) {
                for (let index = 0; index < term.length - 1; index += 1) {
                    terms.add(term.slice(index, index + 2));
                }
            }
        });

        return Array.from(terms);
    }

    function scoreChunk(question, chunk) {
        const haystack = `${chunk.heading} ${chunk.text}`.toLowerCase();
        const normalizedQuestion = normalizeWhitespace(question).toLowerCase();
        const terms = extractTerms(question);
        let score = 0;

        if (normalizedQuestion && haystack.includes(normalizedQuestion)) {
            score += 24;
        }

        terms.forEach((term) => {
            if (!term || term.length < 2 || !haystack.includes(term)) {
                return;
            }

            score += Math.min(14, 2 + term.length * 1.6);
        });

        const cjkChars = Array.from(new Set((normalizedQuestion.match(/[\u4e00-\u9fff]/g) || [])));
        cjkChars.forEach((char) => {
            if (haystack.includes(char)) {
                score += 0.75;
            }
        });

        if (chunk.heading && normalizedQuestion.includes(chunk.heading.toLowerCase())) {
            score += 18;
        }

        if (chunk.type === 'pre' && QUESTION_PATTERN.test(normalizedQuestion)) {
            score += 4;
        }

        return score;
    }

    function mergeChunks(chunks) {
        const merged = [];
        let current = null;

        chunks.forEach((chunk) => {
            if (!current) {
                current = { ...chunk };
                return;
            }

            const sameHeading = current.heading === chunk.heading;
            const withinLimit = current.text.length + chunk.text.length < 560;
            const canMerge = sameHeading && withinLimit && current.type !== 'pre' && chunk.type !== 'pre';

            if (canMerge) {
                current.text = `${current.text}\n${chunk.text}`;
                return;
            }

            merged.push(current);
            current = { ...chunk };
        });

        if (current) {
            merged.push(current);
        }

        return merged;
    }

    function collectArticleData() {
        const title = normalizeWhitespace(
            config.page && config.page.title
                ? config.page.title
                : article.querySelector('.post-title')?.textContent
        );
        const description = normalizeWhitespace(
            config.page && config.page.description
                ? config.page.description
                : article.querySelector('.post-description')?.textContent
        );
        const url = (config.page && config.page.permalink) || window.location.href;
        const headings = Array.from(content.querySelectorAll('h2, h3, h4'))
            .map((heading) => normalizeWhitespace(heading.textContent))
            .filter(Boolean);

        const rawChunks = [];
        let currentHeading = title || '正文';
        const nodes = content.querySelectorAll('h2, h3, h4, h5, h6, p, li, blockquote, pre, figcaption, td');

        nodes.forEach((node) => {
            const tag = node.tagName.toLowerCase();
            let text = '';

            if (tag === 'pre') {
                text = normalizeWhitespace(node.innerText || node.textContent);
            } else {
                text = normalizeWhitespace(node.textContent);
            }

            if (!text) {
                return;
            }

            if (/^h[2-6]$/.test(tag)) {
                currentHeading = text;
                return;
            }

            if (tag !== 'pre' && text.length < 18) {
                return;
            }

            rawChunks.push({
                heading: currentHeading || '正文',
                text: truncate(text, tag === 'pre' ? 720 : 420),
                type: tag
            });
        });

        const mergedChunks = mergeChunks(rawChunks);
        const plainText = truncate(content.innerText || content.textContent || '', 24000);

        return {
            title,
            description,
            url,
            headings,
            plainText,
            chunks: mergedChunks
        };
    }

    function findRelevantChunks(question, articleData) {
        const ranked = articleData.chunks
            .map((chunk) => ({
                ...chunk,
                score: scoreChunk(question, chunk)
            }))
            .sort((left, right) => right.score - left.score);

        const selected = [];
        let totalLength = 0;
        const maxChunks = Number(config.maxChunks) || 6;
        const maxContextChars = Number(config.maxContextChars) || 12000;

        ranked.forEach((chunk) => {
            if (chunk.score <= 0 || selected.length >= maxChunks) {
                return;
            }

            if (totalLength + chunk.text.length > maxContextChars) {
                return;
            }

            selected.push(chunk);
            totalLength += chunk.text.length;
        });

        if (!selected.length) {
            articleData.chunks.slice(0, 3).forEach((chunk) => selected.push({ ...chunk, score: 0 }));
        }

        return selected;
    }

    function buildSummaryAnswer(articleData, selectedChunks) {
        const bullets = selectedChunks.slice(0, 3).map((chunk, index) => {
            const prefix = chunk.heading && chunk.heading !== articleData.title ? `【${chunk.heading}】` : '';
            return `${index + 1}. ${prefix}${truncate(chunk.text, 92)}`;
        });

        return [
            `《${articleData.title}》主要围绕这几个部分展开：`,
            ...bullets,
            '如果你想继续深挖某一段，我可以再针对术语、代码或结论继续解释。'
        ].join('\n');
    }

    function buildLocalAnswer(question, articleData, selectedChunks, hint) {
        if (!selectedChunks.length) {
            return {
                text: `我暂时没有在《${articleData.title}》里找到足够相关的内容。你可以换个更具体的问法，比如某个术语、代码片段或结论。`,
                sources: []
            };
        }

        if (SUMMARY_PATTERN.test(question)) {
            return {
                text: `${hint ? `${hint}\n\n` : ''}${buildSummaryAnswer(articleData, selectedChunks)}`,
                sources: selectedChunks.slice(0, 3)
            };
        }

        const primary = selectedChunks[0];
        const primaryLabel = primary.heading && primary.heading !== articleData.title
            ? `文中在“${primary.heading}”这一部分`
            : '根据当前文章内容';
        const supplemental = selectedChunks[1]
            ? `\n\n补充相关段落：${truncate(selectedChunks[1].text, 120)}`
            : '';

        return {
            text: `${hint ? `${hint}\n\n` : ''}${primaryLabel}，比较相关的信息是：\n${truncate(primary.text, 220)}${supplemental}\n\n如果你愿意，我还可以继续针对这个问题往下追问。`,
            sources: selectedChunks.slice(0, 3)
        };
    }

    function collectTextParts(value) {
        if (!value) {
            return [];
        }

        if (typeof value === 'string') {
            return [value];
        }

        if (Array.isArray(value)) {
            return value.flatMap((item) => collectTextParts(item));
        }

        if (typeof value.text === 'string') {
            return [value.text];
        }

        if (typeof value.content === 'string') {
            return [value.content];
        }

        if (Array.isArray(value.content)) {
            return collectTextParts(value.content);
        }

        if (typeof value.delta === 'string') {
            return [value.delta];
        }

        return [];
    }

    function joinTextParts(value, joiner, trim) {
        const text = collectTextParts(value).join(joiner);
        return trim === false ? text : text.trim();
    }

    function extractPayloadError(payload) {
        if (!payload) {
            return '';
        }

        if (typeof payload.error === 'string') {
            return payload.error.trim();
        }

        if (payload.error && typeof payload.error.message === 'string') {
            return payload.error.message.trim();
        }

        if (payload.type === 'error' && typeof payload.message === 'string') {
            return payload.message.trim();
        }

        return '';
    }

    function parseResponseText(payload) {
        if (!payload) {
            return '';
        }

        if (typeof payload === 'string') {
            return payload.trim();
        }

        if (typeof payload.answer === 'string') {
            return payload.answer.trim();
        }

        if (typeof payload.output_text === 'string') {
            return payload.output_text.trim();
        }

        if (Array.isArray(payload.content)) {
            const contentText = joinTextParts(payload.content, '\n\n');

            if (contentText) {
                return contentText;
            }
        }

        if (Array.isArray(payload.output)) {
            const outputText = payload.output
                .map((item) => joinTextParts(item?.content || item, '\n\n'))
                .filter(Boolean)
                .join('\n\n')
                .trim();

            if (outputText) {
                return outputText;
            }
        }

        if (Array.isArray(payload.choices) && payload.choices[0]?.message?.content) {
            return joinTextParts(payload.choices[0].message.content, '\n\n');
        }

        return '';
    }

    function parseStreamChunk(payload) {
        const chunk = {
            answerDelta: '',
            reasoningDelta: ''
        };

        if (!payload) {
            return chunk;
        }

        if (typeof payload.output_text_delta === 'string') {
            chunk.answerDelta += payload.output_text_delta;
        }

        if (typeof payload.delta === 'string') {
            if (payload.type === 'response.output_text.delta') {
                chunk.answerDelta += payload.delta;
            } else if (typeof payload.type === 'string' && payload.type.includes('reasoning')) {
                chunk.reasoningDelta += payload.delta;
            }
        }

        if (Array.isArray(payload.choices)) {
            payload.choices.forEach((choice) => {
                if (!choice) {
                    return;
                }

                if (typeof choice.delta?.content === 'string') {
                    chunk.answerDelta += choice.delta.content;
                }

                if (Array.isArray(choice.delta?.content)) {
                    chunk.answerDelta += joinTextParts(choice.delta.content, '', false);
                }

                if (typeof choice.delta?.reasoning_content === 'string') {
                    chunk.reasoningDelta += choice.delta.reasoning_content;
                }

                if (typeof choice.delta?.reasoning === 'string') {
                    chunk.reasoningDelta += choice.delta.reasoning;
                }

                if (typeof choice.text === 'string') {
                    chunk.answerDelta += choice.text;
                }
            });
        }

        if (typeof payload.delta === 'string' && !payload.type) {
            chunk.answerDelta += payload.delta;
        }

        if (typeof payload.reasoning_content === 'string') {
            chunk.reasoningDelta += payload.reasoning_content;
        }

        return chunk;
    }

    function getNextSseBlock(buffer) {
        const match = buffer.match(/\r?\n\r?\n/);
        if (!match || typeof match.index !== 'number') {
            return null;
        }

        return {
            block: buffer.slice(0, match.index),
            rest: buffer.slice(match.index + match[0].length)
        };
    }

    async function consumeEventStream(stream, handlers) {
        const reader = stream.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function handleEventBlock(block) {
            const dataText = block
                .split(/\r?\n/)
                .filter((line) => line.startsWith('data:'))
                .map((line) => line.slice(5).trimStart())
                .join('\n');

            if (!dataText) {
                return false;
            }

            if (dataText === STREAM_DONE_MARKER) {
                return true;
            }

            let payload;
            try {
                payload = JSON.parse(dataText);
            } catch (error) {
                return false;
            }

            const payloadError = extractPayloadError(payload);
            if (payloadError) {
                throw new Error(payloadError);
            }

            const streamChunk = parseStreamChunk(payload);
            if (streamChunk.reasoningDelta) {
                handlers.onReasoning?.(streamChunk.reasoningDelta, payload);
            }

            if (streamChunk.answerDelta) {
                handlers.onDelta?.(streamChunk.answerDelta, payload);
            }

            return false;
        }

        try {
            while (true) {
                const { value, done } = await reader.read();
                buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

                let eventBlock = getNextSseBlock(buffer);
                while (eventBlock) {
                    buffer = eventBlock.rest;
                    if (handleEventBlock(eventBlock.block)) {
                        return;
                    }
                    eventBlock = getNextSseBlock(buffer);
                }

                if (done) {
                    break;
                }
            }

            const tail = decoder.decode();
            if (tail) {
                buffer += tail;
            }

            if (buffer.trim()) {
                handleEventBlock(buffer);
            }
        } finally {
            reader.releaseLock();
        }
    }

    async function readResponseError(response) {
        const contentType = (response.headers.get('content-type') || '').toLowerCase();

        try {
            if (contentType.includes('application/json')) {
                const payload = await response.json();
                return extractPayloadError(payload) || `AI request failed: ${response.status}`;
            }

            const text = (await response.text()).trim();
            return text || `AI request failed: ${response.status}`;
        } catch (error) {
            return `AI request failed: ${response.status}`;
        }
    }

    function isRetriableStatus(status) {
        return [404, 405, 408, 429, 500, 502, 503, 504].includes(status);
    }

    async function fetchRemotePayload(payload, streamEnabled) {
        const requestUrls = Array.isArray(remoteClientConfig.urls) && remoteClientConfig.urls.length
            ? remoteClientConfig.urls
            : [remoteClientConfig.url].filter(Boolean);
        let lastError = null;

        for (let index = 0; index < requestUrls.length; index += 1) {
            const url = requestUrls[index];

            try {
                const response = await fetch(url, {
                    method: 'POST',
                    headers: {
                        ...remoteClientConfig.headers,
                        Accept: streamEnabled ? 'text/event-stream, application/json' : 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                if (response.ok) {
                    return response;
                }

                lastError = new Error(await readResponseError(response));
                if (index < requestUrls.length - 1 && isRetriableStatus(response.status)) {
                    continue;
                }

                throw lastError;
            } catch (error) {
                lastError = error instanceof Error ? error : new Error('AI 请求失败');
                if (index < requestUrls.length - 1) {
                    continue;
                }

                throw lastError;
            }
        }

        throw lastError || new Error('AI 请求失败');
    }

    function buildRemotePayload(question, articleData, selectedChunks, selectedModel, streamEnabled) {
        const contextText = selectedChunks
            .map((chunk, index) => {
                const heading = chunk.heading ? `章节：${chunk.heading}\n` : '';
                return `[片段 ${index + 1}]\n${heading}${chunk.text}`;
            })
            .join('\n\n');

        const systemPrompt = [
            '你是技术博客的文章问答助手。',
            '只能依据用户给出的文章上下文回答，不要编造文章里没有提到的信息。',
            '优先使用中文，回答简洁、准确，并尽量直接回应问题。',
            '如果上下文不足，请明确说明当前文章没有给出这个信息。'
        ].join(' ');

        const userPrompt = [
            `文章标题：${articleData.title}`,
            articleData.description ? `文章简介：${articleData.description}` : '',
            `页面地址：${articleData.url}`,
            '',
            '以下是从当前文章中检索到的相关片段，请仅基于这些内容回答：',
            contextText,
            '',
            `问题：${question}`
        ]
            .filter(Boolean)
            .join('\n');

        if ((config.requestFormat || 'openai-chat') === 'openai-chat') {
            return {
                model: selectedModel || config.model || 'gpt-4.1-mini',
                temperature: Number(config.temperature) || 0.2,
                stream: Boolean(streamEnabled),
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt }
                ]
            };
        }

        return {
            model: selectedModel || config.model || 'gpt-4.1-mini',
            temperature: Number(config.temperature) || 0.2,
            stream: Boolean(streamEnabled),
            systemPrompt,
            question,
            article: {
                title: articleData.title,
                description: articleData.description,
                url: articleData.url
            },
            context: selectedChunks
        };
    }

    async function requestRemoteAnswer(question, articleData, selectedChunks, selectedModel, handlers) {
        if (!remoteClientConfig.enabled) {
            return null;
        }

        const streamPayload = buildRemotePayload(
            question,
            articleData,
            selectedChunks,
            selectedModel,
            true
        );
        let answerText = '';
        let streamError = null;

        try {
            const response = await fetchRemotePayload(streamPayload, true);
            const contentType = (response.headers.get('content-type') || '').toLowerCase();

            if (contentType.includes('text/event-stream') && response.body) {
                await consumeEventStream(response.body, {
                    onReasoning(deltaText, payload) {
                        handlers?.onReasoning?.(deltaText, payload);
                    },
                    onDelta(deltaText, payload) {
                        answerText += deltaText;
                        handlers?.onDelta?.(answerText, deltaText, payload);
                    }
                });

                return answerText.trim() || null;
            }

            const data = await response.json();
            const answer = parseResponseText(data);
            return answer || null;
        } catch (error) {
            if (answerText.trim()) {
                throw error;
            }

            streamError = error;
        }

        const fallbackPayload = buildRemotePayload(
            question,
            articleData,
            selectedChunks,
            selectedModel,
            false
        );
        const response = await fetchRemotePayload(fallbackPayload, false)
            .catch((error) => {
                throw error || streamError;
            });
        const data = await response.json();
        const answer = parseResponseText(data);
        return answer || null;
    }

    function createIcon(type) {
        if (type === 'close') {
            return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';
        }

        if (type === 'send') {
            return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m21 3-9.5 9.5M21 3l-6 18-3.5-8.5L3 9l18-6Z"/></svg>';
        }

        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3c4.97 0 9 3.58 9 8 0 1.86-.72 3.58-1.94 4.97L20 21l-5.28-1.76c-.86.18-1.77.27-2.72.27-4.97 0-9-3.58-9-8s4.03-8 9-8Zm-3.75 7.5h7.5M8.25 14h4.5"/></svg>';
    }

    function AssistantWidget(articleData) {
        this.articleData = articleData;
        this.root = null;
        this.panel = null;
        this.messages = null;
        this.emptyState = null;
        this.form = null;
        this.input = null;
        this.toggle = null;
        this.status = null;
        this.sendButton = null;
        this.modelSelect = null;
        this.suggestionContainer = null;
        this.isOpen = readPreference() === 'true';
        this.isHoverPreview = false;
        this.isBusy = false;
        this.availableModels = getAvailableModels();
        this.selectedModel = readSelectedModel();
        this.position = readPosition();
        this.hoverCloseTimer = null;
        this.dragState = {
            active: false,
            moved: false,
            pointerId: null,
            startPointerX: 0,
            startPointerY: 0,
            startX: 0,
            startY: 0,
            suppressClick: false
        };
    }

    AssistantWidget.prototype.mount = function mount() {
        const root = document.createElement('aside');
        root.className = 'ai-article-assistant';
        root.setAttribute('aria-live', 'polite');
        document.documentElement.classList.add('has-ai-article-assistant');
        document.body.classList.add('has-ai-article-assistant');

        const modelSelector = remoteClientConfig.enabled && this.availableModels.length > 1
            ? `
                <label class="ai-article-assistant__model-field">
                    <span>模型</span>
                    <select class="ai-article-assistant__model-select" aria-label="选择模型">
                        ${this.availableModels
                            .map((model) => `<option value="${escapeHtml(model)}"${model === this.selectedModel ? ' selected' : ''}>${escapeHtml(model)}</option>`)
                            .join('')}
                    </select>
                </label>
            `
            : '';

        root.innerHTML = `
            <button class="ai-article-assistant__toggle" type="button" aria-expanded="${this.isOpen ? 'true' : 'false'}" aria-pressed="${this.isOpen ? 'true' : 'false'}" aria-controls="ai-article-assistant-panel" aria-label="${this.isOpen ? '关闭 AI 文章助手' : '打开 AI 文章助手'}">
                <span class="ai-article-assistant__toggle-hint">问这篇文章</span>
                <span class="ai-article-assistant__toggle-icon">${createIcon('spark')}</span>
                <span class="ai-article-assistant__toggle-label">AI 文章助手</span>
                <span class="ai-article-assistant__toggle-dot" aria-hidden="true"></span>
            </button>
            <section class="ai-article-assistant__panel${this.isOpen ? ' is-open' : ''}" id="ai-article-assistant-panel" role="dialog" aria-modal="false" aria-label="${escapeHtml(config.title || 'AI 文章助手')}" ${this.isOpen ? '' : 'hidden'}>
                <header class="ai-article-assistant__header">
                    <div class="ai-article-assistant__header-main">
                        <div class="ai-article-assistant__header-copy">
                            <p class="ai-article-assistant__eyebrow">当前文章</p>
                            <h2 class="ai-article-assistant__title">${escapeHtml(config.title || 'AI 文章助手')}</h2>
                            <p class="ai-article-assistant__status"></p>
                        </div>
                        <div class="ai-article-assistant__header-actions">
                            ${modelSelector}
                            <button class="ai-article-assistant__close" type="button" aria-label="关闭助手">
                                ${createIcon('close')}
                            </button>
                        </div>
                    </div>
                    <div class="ai-article-assistant__meta">
                        <span class="ai-article-assistant__meta-pill">${escapeHtml(remoteClientConfig.actionHint)}</span>
                    </div>
                </header>
                <div class="ai-article-assistant__messages"></div>
                <div class="ai-article-assistant__suggestions"></div>
                <form class="ai-article-assistant__composer">
                    <label class="ai-article-assistant__sr-only" for="ai-article-assistant-input">提问内容</label>
                    <textarea id="ai-article-assistant-input" class="ai-article-assistant__input" rows="2" placeholder="${escapeHtml(config.placeholder || '请输入问题')}"></textarea>
                    <div class="ai-article-assistant__actions">
                        <span class="ai-article-assistant__hint">${escapeHtml(remoteClientConfig.enabled ? '基于文章上下文作答' : '基于文章内容检索作答')}</span>
                        <button class="ai-article-assistant__send" type="submit">
                            ${createIcon('send')}
                            <span>发送</span>
                        </button>
                    </div>
                </form>
            </section>
        `;

        document.body.appendChild(root);

        this.root = root;
        this.panel = root.querySelector('.ai-article-assistant__panel');
        this.messages = root.querySelector('.ai-article-assistant__messages');
        this.form = root.querySelector('.ai-article-assistant__composer');
        this.input = root.querySelector('.ai-article-assistant__input');
        this.toggle = root.querySelector('.ai-article-assistant__toggle');
        this.status = root.querySelector('.ai-article-assistant__status');
        this.sendButton = root.querySelector('.ai-article-assistant__send');
        this.modelSelect = root.querySelector('.ai-article-assistant__model-select');
        this.suggestionContainer = root.querySelector('.ai-article-assistant__suggestions');
        this.root.classList.add('is-open-up');
        this.syncViewportMode();
        this.syncPanelState();
        this.refreshStatus();
        this.syncInputHeight();

        this.bindEvents();
        this.renderIntro();
        this.renderSuggestions();

        if (this.isOpen) {
            window.requestAnimationFrame(() => {
                this.updateFloatingClasses();
                this.input.focus({ preventScroll: true });
            });
        }
    };

    AssistantWidget.prototype.bindEvents = function bindEvents() {
        const closeButton = this.root.querySelector('.ai-article-assistant__close');

        this.toggle.addEventListener('click', (event) => {
            if (this.dragState.suppressClick) {
                event.preventDefault();
                event.stopPropagation();
                this.dragState.suppressClick = false;
                return;
            }

            this.setOpen(!this.isOpen);
        });

        this.root.addEventListener('mouseenter', () => {
            if (this.isCompactViewport() || this.dragState.active) {
                return;
            }

            this.clearHoverCloseTimer();
            this.setHoverPreview(true);
        });

        this.root.addEventListener('mouseleave', () => {
            if (this.isCompactViewport() || this.dragState.active) {
                return;
            }

            this.scheduleHoverClose();
        });

        this.root.addEventListener('focusin', () => {
            this.clearHoverCloseTimer();
            if (!this.isCompactViewport()) {
                this.setHoverPreview(true);
            }
        });

        this.root.addEventListener('focusout', () => {
            window.requestAnimationFrame(() => {
                if (!this.root.contains(document.activeElement)) {
                    this.scheduleHoverClose();
                }
            });
        });

        this.toggle.addEventListener('pointerdown', (event) => {
            this.startDrag(event);
        });

        this.toggle.addEventListener('pointermove', (event) => {
            this.handleDragMove(event);
        });

        this.toggle.addEventListener('pointerup', (event) => {
            this.handleDragEnd(event);
        });

        this.toggle.addEventListener('pointercancel', (event) => {
            this.handleDragEnd(event);
        });

        this.toggle.addEventListener('dragstart', (event) => {
            event.preventDefault();
        });

        closeButton.addEventListener('click', () => {
            this.setOpen(false);
        });

        this.form.addEventListener('submit', (event) => {
            event.preventDefault();
            this.submitQuestion();
        });

        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', () => {
                this.selectedModel = normalizeWhitespace(this.modelSelect.value);
                writeSelectedModel(this.selectedModel);
                this.refreshStatus();
            });
        }

        this.input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.submitQuestion();
            }
        });

        this.input.addEventListener('input', () => {
            this.syncInputHeight();
        });

        document.addEventListener('pointerdown', (event) => {
            if (!this.isPanelVisible() || this.dragState.active || this.root.contains(event.target)) {
                return;
            }

            this.setOpen(false);
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && this.isPanelVisible()) {
                this.setOpen(false);
                this.toggle.focus({ preventScroll: true });
            }
        });

        window.addEventListener('resize', () => {
            this.handleResize();
        }, { passive: true });
    };

    AssistantWidget.prototype.isPanelVisible = function isPanelVisible() {
        return this.isOpen || this.isHoverPreview;
    };

    AssistantWidget.prototype.syncPanelState = function syncPanelState(options) {
        const settings = options || {};
        const panelVisible = this.isPanelVisible();

        this.toggle.setAttribute('aria-expanded', String(panelVisible));
        this.toggle.setAttribute('aria-pressed', String(this.isOpen));
        this.toggle.setAttribute('aria-label', panelVisible ? '关闭 AI 文章助手' : '打开 AI 文章助手');
        this.panel.classList.toggle('is-open', panelVisible);
        this.root.classList.toggle('is-open', panelVisible);
        this.root.classList.toggle('is-preview', !this.isOpen && this.isHoverPreview);
        this.root.classList.toggle('is-pinned', this.isOpen);

        if (panelVisible) {
            this.panel.hidden = false;
            window.requestAnimationFrame(() => {
                this.updateFloatingClasses();
                if (settings.focusInput) {
                    this.input.focus({ preventScroll: true });
                }
            });
            return;
        }

        this.panel.hidden = true;
        this.root.classList.add('is-open-up');
        this.root.classList.remove('is-open-down');
    };

    AssistantWidget.prototype.clearHoverCloseTimer = function clearHoverCloseTimer() {
        if (!this.hoverCloseTimer) {
            return;
        }

        window.clearTimeout(this.hoverCloseTimer);
        this.hoverCloseTimer = null;
    };

    AssistantWidget.prototype.setHoverPreview = function setHoverPreview(nextHover, options) {
        if (this.isCompactViewport() && nextHover) {
            return;
        }

        if (this.isHoverPreview === nextHover) {
            return;
        }

        this.isHoverPreview = nextHover;
        this.syncPanelState(options);
    };

    AssistantWidget.prototype.scheduleHoverClose = function scheduleHoverClose() {
        if (this.isCompactViewport() || this.isOpen) {
            return;
        }

        this.clearHoverCloseTimer();
        this.hoverCloseTimer = window.setTimeout(() => {
            this.hoverCloseTimer = null;

            if (
                this.isOpen ||
                this.dragState.active ||
                this.root.matches(':hover') ||
                this.root.contains(document.activeElement)
            ) {
                return;
            }

            this.setHoverPreview(false);
        }, HOVER_CLOSE_DELAY);
    };

    AssistantWidget.prototype.setOpen = function setOpen(nextOpen) {
        this.isOpen = nextOpen;
        this.clearHoverCloseTimer();

        if (!nextOpen) {
            this.isHoverPreview = false;
        }

        writePreference(String(nextOpen));
        this.syncPanelState({ focusInput: nextOpen });
    };

    AssistantWidget.prototype.refreshStatus = function refreshStatus(overrideText) {
        if (!this.status) {
            return;
        }

        if (overrideText) {
            this.status.textContent = overrideText;
            return;
        }

        this.status.textContent = remoteClientConfig.enabled
            ? `${remoteClientConfig.statusText}${this.selectedModel ? ` · ${this.selectedModel}` : ''}`
            : '文章检索模式';
    };

    AssistantWidget.prototype.getToggleSize = function getToggleSize() {
        return {
            width: this.toggle ? this.toggle.offsetWidth || 68 : 68,
            height: this.toggle ? this.toggle.offsetHeight || 68 : 68
        };
    };

    AssistantWidget.prototype.isCompactViewport = function isCompactViewport() {
        return window.innerWidth <= 768;
    };

    AssistantWidget.prototype.getDefaultPosition = function getDefaultPosition() {
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const toggleSize = this.getToggleSize();

        return {
            x: viewportWidth - toggleSize.width - VIEWPORT_GAP,
            y: viewportHeight - toggleSize.height - VIEWPORT_GAP
        };
    };

    AssistantWidget.prototype.clampPosition = function clampPosition(position) {
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const toggleSize = this.getToggleSize();
        const maxX = Math.max(VIEWPORT_GAP, viewportWidth - toggleSize.width - VIEWPORT_GAP);
        const maxY = Math.max(VIEWPORT_GAP, viewportHeight - toggleSize.height - VIEWPORT_GAP);

        return {
            x: clampNumber(position.x, VIEWPORT_GAP, maxX),
            y: clampNumber(position.y, VIEWPORT_GAP, maxY)
        };
    };

    AssistantWidget.prototype.snapPosition = function snapPosition(position) {
        const clamped = this.clampPosition(position);
        const viewportWidth = window.innerWidth;
        const toggleSize = this.getToggleSize();
        const dockLeft = VIEWPORT_GAP;
        const dockRight = Math.max(VIEWPORT_GAP, viewportWidth - toggleSize.width - VIEWPORT_GAP);
        const shouldDockRight = clamped.x + toggleSize.width / 2 >= viewportWidth / 2;

        return {
            x: shouldDockRight ? dockRight : dockLeft,
            y: clamped.y
        };
    };

    AssistantWidget.prototype.applyPosition = function applyPosition(position, options) {
        const settings = options || {};
        const nextPosition = this.clampPosition(position);

        this.position = nextPosition;
        if (this.isCompactViewport()) {
            this.root.style.left = '';
            this.root.style.top = '';
            this.updateFloatingClasses();
            if (settings.persist !== false) {
                writePosition(nextPosition);
            }
            return;
        }

        this.root.style.left = `${nextPosition.x}px`;
        this.root.style.top = `${nextPosition.y}px`;
        this.updateFloatingClasses();

        if (settings.persist !== false) {
            writePosition(nextPosition);
        }
    };

    AssistantWidget.prototype.updateFloatingClasses = function updateFloatingClasses() {
        if (!this.root || !this.position) {
            return;
        }

        const toggleSize = this.getToggleSize();
        const isRight = this.position.x + toggleSize.width / 2 >= window.innerWidth / 2;

        this.root.classList.toggle('is-align-right', isRight);
        this.root.classList.toggle('is-align-left', !isRight);
        this.updatePanelPlacement();
    };

    AssistantWidget.prototype.syncViewportMode = function syncViewportMode() {
        const compact = this.isCompactViewport();

        this.root.classList.toggle('is-compact', compact);

        if (compact) {
            this.clearHoverCloseTimer();
            this.isHoverPreview = false;
            this.root.style.left = '';
            this.root.style.top = '';
            this.root.classList.remove('is-align-left');
            this.root.classList.add('is-align-right');
            this.updatePanelPlacement();
            this.syncPanelState();
            return;
        }

        this.applyPosition(this.position || this.getDefaultPosition(), { persist: false });
        this.syncPanelState();
    };

    AssistantWidget.prototype.updatePanelPlacement = function updatePanelPlacement() {
        if (!this.root) {
            return;
        }

        if (!this.isPanelVisible() || !this.panel || this.panel.hidden) {
            this.root.classList.add('is-open-up');
            this.root.classList.remove('is-open-down');
            return;
        }

        const toggleRect = this.toggle.getBoundingClientRect();
        const panelRect = this.panel.getBoundingClientRect();
        const panelHeight = panelRect.height || 560;
        const spaceAbove = toggleRect.top - VIEWPORT_GAP;
        const spaceBelow = window.innerHeight - toggleRect.bottom - VIEWPORT_GAP;
        const shouldOpenDown = spaceAbove < Math.min(panelHeight, 320) && spaceBelow > spaceAbove;

        this.root.classList.toggle('is-open-down', shouldOpenDown);
        this.root.classList.toggle('is-open-up', !shouldOpenDown);
    };

    AssistantWidget.prototype.startDrag = function startDrag(event) {
        if (this.isCompactViewport() || (typeof event.button === 'number' && event.button !== 0)) {
            return;
        }

        this.clearHoverCloseTimer();

        if (!this.isOpen && this.isHoverPreview) {
            this.isHoverPreview = false;
            this.syncPanelState();
        }

        this.dragState.suppressClick = false;
        this.dragState.active = true;
        this.dragState.moved = false;
        this.dragState.pointerId = event.pointerId;
        this.dragState.startPointerX = event.clientX;
        this.dragState.startPointerY = event.clientY;
        this.dragState.startX = this.position ? this.position.x : this.getDefaultPosition().x;
        this.dragState.startY = this.position ? this.position.y : this.getDefaultPosition().y;

        if (this.toggle.setPointerCapture) {
            try {
                this.toggle.setPointerCapture(event.pointerId);
            } catch (error) {
                /* noop */
            }
        }
    };

    AssistantWidget.prototype.handleDragMove = function handleDragMove(event) {
        if (!this.dragState.active || this.dragState.pointerId !== event.pointerId) {
            return;
        }

        const deltaX = event.clientX - this.dragState.startPointerX;
        const deltaY = event.clientY - this.dragState.startPointerY;

        if (!this.dragState.moved && Math.hypot(deltaX, deltaY) >= DRAG_THRESHOLD) {
            this.dragState.moved = true;
            this.root.classList.add('is-dragging');
        }

        if (!this.dragState.moved) {
            return;
        }

        event.preventDefault();
        this.applyPosition(
            {
                x: this.dragState.startX + deltaX,
                y: this.dragState.startY + deltaY
            },
            { persist: false }
        );
    };

    AssistantWidget.prototype.handleDragEnd = function handleDragEnd(event) {
        if (!this.dragState.active || this.dragState.pointerId !== event.pointerId) {
            return;
        }

        if (this.toggle.releasePointerCapture) {
            try {
                this.toggle.releasePointerCapture(event.pointerId);
            } catch (error) {
                /* noop */
            }
        }

        if (this.dragState.moved) {
            this.dragState.suppressClick = true;
            this.applyPosition(this.snapPosition(this.position), { persist: true });
        }

        this.root.classList.remove('is-dragging');
        this.dragState.active = false;
        this.dragState.moved = false;
        this.dragState.pointerId = null;
    };

    AssistantWidget.prototype.handleResize = function handleResize() {
        if (this.isCompactViewport()) {
            this.syncViewportMode();
            return;
        }

        const fallbackPosition = this.position || this.getDefaultPosition();
        this.root.classList.remove('is-compact');
        this.applyPosition(this.snapPosition(fallbackPosition), { persist: false });
    };

    AssistantWidget.prototype.syncInputHeight = function syncInputHeight() {
        if (!this.input) {
            return;
        }

        this.input.style.height = 'auto';
        this.input.style.height = `${clampNumber(this.input.scrollHeight, INPUT_MIN_HEIGHT, INPUT_MAX_HEIGHT)}px`;
    };

    AssistantWidget.prototype.renderIntro = function renderIntro() {
        if (remoteClientConfig.warningText) {
            console.warn(`[ai-assistant] ${remoteClientConfig.warningText}`);
        }

        const emptyState = document.createElement('section');
        emptyState.className = 'ai-article-assistant__empty';

        const label = document.createElement('p');
        label.className = 'ai-article-assistant__empty-kicker';
        label.textContent = '直接提问';

        const copy = document.createElement('p');
        copy.className = 'ai-article-assistant__empty-copy';
        copy.textContent = config.greeting || '可以直接问文章要点、术语或代码细节。';

        const meta = document.createElement('div');
        meta.className = 'ai-article-assistant__empty-meta';

        [
            remoteClientConfig.enabled ? '流式回复' : '文章检索回复',
            this.articleData.headings.length
                ? `${this.articleData.headings.length} 个小节可检索`
                : '基于正文片段回答'
        ].forEach((text) => {
            const pill = document.createElement('span');
            pill.className = 'ai-article-assistant__empty-pill';
            pill.textContent = text;
            meta.appendChild(pill);
        });

        emptyState.appendChild(label);
        emptyState.appendChild(copy);
        emptyState.appendChild(meta);

        this.emptyState = emptyState;
        this.messages.appendChild(emptyState);
    };

    AssistantWidget.prototype.renderSuggestions = function renderSuggestions() {
        const suggestionContainer = this.suggestionContainer;
        const questions = Array.isArray(config.suggestedQuestions) ? config.suggestedQuestions : [];

        if (!suggestionContainer) {
            return;
        }

        if (!questions.length) {
            suggestionContainer.hidden = true;
            return;
        }

        suggestionContainer.hidden = false;
        suggestionContainer.innerHTML = '';

        questions.forEach((question) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'ai-article-assistant__chip';
            button.textContent = question;
            button.title = question;
            button.setAttribute('aria-label', question);
            button.addEventListener('click', () => {
                this.input.value = question;
                this.syncInputHeight();
                this.submitQuestion();
            });
            suggestionContainer.appendChild(button);
        });
    };

    AssistantWidget.prototype.clearIntro = function clearIntro() {
        if (!this.emptyState) {
            return;
        }

        this.emptyState.remove();
        this.emptyState = null;
    };

    AssistantWidget.prototype.scrollMessagesToEnd = function scrollMessagesToEnd() {
        if (!this.messages) {
            return;
        }

        this.messages.scrollTop = this.messages.scrollHeight;
    };

    AssistantWidget.prototype.createMessage = function createMessage(role, options) {
        const settings = options || {};
        this.clearIntro();

        const message = document.createElement('article');
        message.className = `ai-article-assistant__message ai-article-assistant__message--${role}`;
        if (settings.subtle) {
            message.classList.add('is-subtle');
        }
        if (settings.streaming) {
            message.classList.add('is-streaming');
        }

        const bubble = document.createElement('div');
        bubble.className = 'ai-article-assistant__bubble';
        bubble.textContent = settings.text || '';
        message.appendChild(bubble);

        this.messages.appendChild(message);
        this.scrollMessagesToEnd();

        const messageRef = {
            element: message,
            bubble,
            sources: null
        };

        if (Array.isArray(settings.sources) && settings.sources.length) {
            this.setMessageSources(messageRef, settings.sources);
        }

        return messageRef;
    };

    AssistantWidget.prototype.setMessageText = function setMessageText(messageRef, text) {
        if (!messageRef || !messageRef.bubble) {
            return;
        }

        messageRef.bubble.textContent = text;
        this.scrollMessagesToEnd();
    };

    AssistantWidget.prototype.getMessageText = function getMessageText(messageRef) {
        if (!messageRef || !messageRef.bubble) {
            return '';
        }

        return messageRef.bubble.textContent || '';
    };

    AssistantWidget.prototype.setMessageStreaming = function setMessageStreaming(messageRef, nextStreaming) {
        if (!messageRef || !messageRef.element) {
            return;
        }

        messageRef.element.classList.toggle('is-streaming', Boolean(nextStreaming));
        this.scrollMessagesToEnd();
    };

    AssistantWidget.prototype.setMessageSources = function setMessageSources(messageRef, sources) {
        if (!messageRef || !messageRef.element) {
            return;
        }

        if (messageRef.sources) {
            messageRef.sources.remove();
            messageRef.sources = null;
        }

        if (!Array.isArray(sources) || !sources.length) {
            return;
        }

        const sourceList = document.createElement('div');
        sourceList.className = 'ai-article-assistant__sources';

        sources.slice(0, 3).forEach((source) => {
            const item = document.createElement('div');
            item.className = 'ai-article-assistant__source';

            const label = document.createElement('strong');
            label.textContent = source.heading || '正文';

            const excerpt = document.createElement('span');
            excerpt.textContent = truncate(source.text, 86);

            item.appendChild(label);
            item.appendChild(excerpt);
            sourceList.appendChild(item);
        });

        messageRef.sources = sourceList;
        messageRef.element.appendChild(sourceList);
        this.scrollMessagesToEnd();
    };

    AssistantWidget.prototype.appendMessage = function appendMessage(role, text, options) {
        const messageRef = this.createMessage(role, {
            subtle: options && options.subtle,
            streaming: options && options.streaming,
            text
        });

        if (options && Array.isArray(options.sources) && options.sources.length) {
            this.setMessageSources(messageRef, options.sources);
        }

        return messageRef;
    };

    AssistantWidget.prototype.setBusy = function setBusy(nextBusy) {
        this.isBusy = nextBusy;
        this.form.classList.toggle('is-busy', nextBusy);
        this.sendButton.disabled = nextBusy;
        this.input.disabled = nextBusy;
        if (this.modelSelect) {
            this.modelSelect.disabled = nextBusy;
        }
        this.refreshStatus(nextBusy ? (remoteClientConfig.enabled ? '正在流式作答...' : '正在整理答案...') : '');
    };

    AssistantWidget.prototype.submitQuestion = async function submitQuestion() {
        const question = normalizeWhitespace(this.input.value);
        if (!question || this.isBusy) {
            return;
        }

        const selectedChunks = findRelevantChunks(question, this.articleData);
        this.appendMessage('user', question, { sources: [] });
        this.input.value = '';
        this.syncInputHeight();
        this.setBusy(true);
        this.setOpen(true);
        const assistantMessage = this.createMessage('assistant', {
            streaming: remoteClientConfig.enabled
        });

        try {
            let answerText = null;
            let hasRemoteAnswerDelta = false;

            if (remoteClientConfig.enabled) {
                try {
                    answerText = await requestRemoteAnswer(
                        question,
                        this.articleData,
                        selectedChunks,
                        this.selectedModel || config.model || '',
                        {
                            onReasoning: () => {
                                if (!hasRemoteAnswerDelta && !this.getMessageText(assistantMessage)) {
                                    this.setMessageStreaming(assistantMessage, true);
                                    this.setMessageText(assistantMessage, '正在推理...');
                                }
                            },
                            onDelta: (nextText) => {
                                hasRemoteAnswerDelta = true;
                                this.setMessageStreaming(assistantMessage, true);
                                this.setMessageText(assistantMessage, nextText);
                            }
                        }
                    );
                } catch (error) {
                    if (hasRemoteAnswerDelta) {
                        this.setMessageStreaming(assistantMessage, false);
                        this.setMessageSources(assistantMessage, selectedChunks.slice(0, 3));
                        return;
                    }

                    const localResult = buildLocalAnswer(
                        question,
                        this.articleData,
                        selectedChunks,
                        config.errorHint || 'AI 接口暂时不可用，已自动切换到文章检索模式。'
                    );
                    this.setMessageText(assistantMessage, localResult.text);
                    this.setMessageSources(assistantMessage, localResult.sources);
                    return;
                }
            }

            if (answerText) {
                this.setMessageText(assistantMessage, answerText);
                this.setMessageSources(assistantMessage, selectedChunks.slice(0, 3));
                return;
            }

            const localResult = buildLocalAnswer(question, this.articleData, selectedChunks, '');
            this.setMessageText(assistantMessage, localResult.text);
            this.setMessageSources(assistantMessage, localResult.sources);
        } finally {
            this.setMessageStreaming(assistantMessage, false);
            this.setBusy(false);
            this.input.focus({ preventScroll: true });
        }
    };

    const widget = new AssistantWidget(collectArticleData());
    widget.mount();
})();
