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

    function truncate(value, maxLength) {
        const text = normalizeWhitespace(value);
        if (text.length <= maxLength) {
            return text;
        }
        return `${text.slice(0, maxLength).trim()}…`;
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
            return {
                enabled: true,
                mode: 'endpoint',
                url: endpoint,
                headers: {
                    'Content-Type': 'application/json'
                },
                statusText: '已连接 AI 代理',
                actionHint: '基于当前文章 + 远程 AI 回答',
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
                headers: {},
                statusText: '文章检索模式',
                actionHint: '当前先用文章内容检索回答',
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
            headers,
            statusText: '前端直连 AI 接口',
            actionHint: '基于当前文章 + 远程 LLM 回答',
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

    async function parseResponseText(payload) {
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
            const contentText = payload.content
                .map((item) => {
                    if (typeof item === 'string') {
                        return item;
                    }

                    if (item && typeof item.text === 'string') {
                        return item.text;
                    }

                    return '';
                })
                .filter(Boolean)
                .join('\n\n')
                .trim();

            if (contentText) {
                return contentText;
            }
        }

        if (Array.isArray(payload.output)) {
            const outputText = payload.output
                .flatMap((item) => {
                    if (!item || !Array.isArray(item.content)) {
                        return [];
                    }

                    return item.content.map((contentItem) => contentItem?.text || '').filter(Boolean);
                })
                .join('\n\n')
                .trim();

            if (outputText) {
                return outputText;
            }
        }

        if (Array.isArray(payload.choices) && payload.choices[0]?.message?.content) {
            const contentValue = payload.choices[0].message.content;
            if (typeof contentValue === 'string') {
                return contentValue.trim();
            }

            if (Array.isArray(contentValue)) {
                return contentValue
                    .map((item) => item?.text || '')
                    .filter(Boolean)
                    .join('\n\n')
                    .trim();
            }
        }

        return '';
    }

    async function requestRemoteAnswer(question, articleData, selectedChunks, selectedModel) {
        if (!remoteClientConfig.enabled) {
            return null;
        }

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

        let payload;
        if ((config.requestFormat || 'openai-chat') === 'openai-chat') {
            payload = {
                model: selectedModel || config.model || 'gpt-4.1-mini',
                temperature: Number(config.temperature) || 0.2,
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userPrompt }
                ]
            };
        } else {
            payload = {
                model: selectedModel || config.model || 'gpt-4.1-mini',
                temperature: Number(config.temperature) || 0.2,
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

        const response = await fetch(remoteClientConfig.url, {
            method: 'POST',
            headers: remoteClientConfig.headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`AI request failed: ${response.status}`);
        }

        const data = await response.json();
        const answer = await parseResponseText(data);
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
        this.form = null;
        this.input = null;
        this.toggle = null;
        this.status = null;
        this.sendButton = null;
        this.modelSelect = null;
        this.isOpen = readPreference() === 'true';
        this.isBusy = false;
        this.availableModels = getAvailableModels();
        this.selectedModel = readSelectedModel();
    }

    AssistantWidget.prototype.mount = function mount() {
        const root = document.createElement('aside');
        root.className = 'ai-article-assistant';
        root.setAttribute('aria-live', 'polite');

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
            <button class="ai-article-assistant__toggle" type="button" aria-expanded="${this.isOpen ? 'true' : 'false'}" aria-controls="ai-article-assistant-panel">
                <span class="ai-article-assistant__toggle-icon">${createIcon('spark')}</span>
                <span class="ai-article-assistant__toggle-label">问文章</span>
            </button>
            <section class="ai-article-assistant__panel${this.isOpen ? ' is-open' : ''}" id="ai-article-assistant-panel" ${this.isOpen ? '' : 'hidden'}>
                <header class="ai-article-assistant__header">
                    <div>
                        <h2 class="ai-article-assistant__title">${escapeHtml(config.title || 'AI 文章助手')}</h2>
                        <p class="ai-article-assistant__status">${escapeHtml(remoteClientConfig.statusText)}</p>
                    </div>
                    <button class="ai-article-assistant__close" type="button" aria-label="关闭助手">
                        ${createIcon('close')}
                    </button>
                </header>
                <div class="ai-article-assistant__messages"></div>
                <div class="ai-article-assistant__suggestions"></div>
                <form class="ai-article-assistant__composer">
                    ${modelSelector}
                    <label class="ai-article-assistant__sr-only" for="ai-article-assistant-input">提问内容</label>
                    <textarea id="ai-article-assistant-input" class="ai-article-assistant__input" rows="3" placeholder="${escapeHtml(config.placeholder || '请输入问题')}"></textarea>
                    <div class="ai-article-assistant__actions">
                        <span class="ai-article-assistant__hint">${escapeHtml(remoteClientConfig.actionHint)}</span>
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

        this.bindEvents();
        this.renderIntro();
        this.renderSuggestions();

        if (this.isOpen) {
            this.input.focus({ preventScroll: true });
        }
    };

    AssistantWidget.prototype.bindEvents = function bindEvents() {
        const closeButton = this.root.querySelector('.ai-article-assistant__close');

        this.toggle.addEventListener('click', () => {
            this.setOpen(!this.isOpen);
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
                this.status.textContent = `${remoteClientConfig.statusText} · ${this.selectedModel}`;
            });
        }

        this.input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.submitQuestion();
            }
        });
    };

    AssistantWidget.prototype.setOpen = function setOpen(nextOpen) {
        this.isOpen = nextOpen;
        this.toggle.setAttribute('aria-expanded', String(nextOpen));
        this.panel.classList.toggle('is-open', nextOpen);

        if (nextOpen) {
            this.panel.hidden = false;
            this.input.focus({ preventScroll: true });
        } else {
            this.panel.hidden = true;
        }

        writePreference(String(nextOpen));
    };

    AssistantWidget.prototype.renderIntro = function renderIntro() {
        this.appendMessage('assistant', config.greeting || '我可以结合当前文章内容回答问题。', {
            sources: [],
            subtle: true
        });

        if (!remoteClientConfig.enabled) {
            this.appendMessage('assistant', config.fallbackHint || '当前未配置 AI 接口，先使用文章检索模式回答。', {
                sources: [],
                subtle: true
            });
        }

        if (remoteClientConfig.warningText) {
            this.appendMessage('assistant', remoteClientConfig.warningText, {
                sources: [],
                subtle: true
            });
        }
    };

    AssistantWidget.prototype.renderSuggestions = function renderSuggestions() {
        const suggestionContainer = this.root.querySelector('.ai-article-assistant__suggestions');
        const questions = Array.isArray(config.suggestedQuestions) ? config.suggestedQuestions : [];

        if (!questions.length) {
            suggestionContainer.hidden = true;
            return;
        }

        suggestionContainer.innerHTML = '';

        questions.forEach((question) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'ai-article-assistant__chip';
            button.textContent = question;
            button.addEventListener('click', () => {
                this.input.value = question;
                this.submitQuestion();
            });
            suggestionContainer.appendChild(button);
        });
    };

    AssistantWidget.prototype.appendMessage = function appendMessage(role, text, options) {
        const message = document.createElement('article');
        message.className = `ai-article-assistant__message ai-article-assistant__message--${role}`;
        if (options && options.subtle) {
            message.classList.add('is-subtle');
        }

        const bubble = document.createElement('div');
        bubble.className = 'ai-article-assistant__bubble';
        bubble.textContent = text;
        message.appendChild(bubble);

        if (options && Array.isArray(options.sources) && options.sources.length) {
            const sourceList = document.createElement('div');
            sourceList.className = 'ai-article-assistant__sources';

            options.sources.slice(0, 3).forEach((source) => {
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

            message.appendChild(sourceList);
        }

        this.messages.appendChild(message);
        this.messages.scrollTop = this.messages.scrollHeight;
    };

    AssistantWidget.prototype.setBusy = function setBusy(nextBusy) {
        this.isBusy = nextBusy;
        this.form.classList.toggle('is-busy', nextBusy);
        this.sendButton.disabled = nextBusy;
        this.input.disabled = nextBusy;
        if (this.modelSelect) {
            this.modelSelect.disabled = nextBusy;
        }
        this.status.textContent = nextBusy
            ? '正在整理答案...'
            : (remoteClientConfig.enabled
                ? `${remoteClientConfig.statusText}${this.selectedModel ? ` · ${this.selectedModel}` : ''}`
                : '文章检索模式');
    };

    AssistantWidget.prototype.submitQuestion = async function submitQuestion() {
        const question = normalizeWhitespace(this.input.value);
        if (!question || this.isBusy) {
            return;
        }

        const selectedChunks = findRelevantChunks(question, this.articleData);
        this.appendMessage('user', question, { sources: [] });
        this.input.value = '';
        this.setBusy(true);
        this.setOpen(true);

        try {
            let answerText = null;

            if (remoteClientConfig.enabled) {
                try {
                    answerText = await requestRemoteAnswer(
                        question,
                        this.articleData,
                        selectedChunks,
                        this.selectedModel || config.model || ''
                    );
                } catch (error) {
                    const localResult = buildLocalAnswer(
                        question,
                        this.articleData,
                        selectedChunks,
                        config.errorHint || 'AI 接口暂时不可用，已自动切换到文章检索模式。'
                    );
                    this.appendMessage('assistant', localResult.text, { sources: localResult.sources });
                    return;
                }
            }

            if (answerText) {
                this.appendMessage('assistant', answerText, { sources: selectedChunks.slice(0, 3) });
                return;
            }

            const localResult = buildLocalAnswer(question, this.articleData, selectedChunks, '');
            this.appendMessage('assistant', localResult.text, { sources: localResult.sources });
        } finally {
            this.setBusy(false);
            this.input.focus({ preventScroll: true });
        }
    };

    const widget = new AssistantWidget(collectArticleData());
    widget.mount();
})();
