(() => {
    const DEFAULT_CONFIG = {
        embeddingEndpoint: '/api/blog-search-embedding',
        rerankEndpoint: '/api/blog-search-rerank',
        keywordIndexFile: '/index.json',
        vectorIndexFile: '/search-vectors.json',
        embeddingModel: 'baai/bge-m3(free)',
        rerankEnabled: true,
        rerankModel: 'BAAI/bge-reranker-v2-m3(free)',
        minQueryLength: 2,
        maxResults: 8,
        scoreThreshold: 0.18,
        recallK: 24,
        rerankTopN: 8,
        rerankDocumentMaxChars: 1800
    };

    const QUICK_TAGS = ['Python', '数据库', '机器学习', '算法', 'Linux'];

    const debounce = (fn, delay = 300) => {
        let timerId = 0;
        return (...args) => {
            window.clearTimeout(timerId);
            timerId = window.setTimeout(() => fn(...args), delay);
        };
    };

    const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

    const escapeHtml = (text) => {
        const node = document.createElement('div');
        node.textContent = text;
        return node.innerHTML;
    };

    const clampScore = (value) => Math.max(0, Math.min(1, Number(value) || 0));

    const normalizeText = (text) => {
        if (typeof text !== 'string') {
            return '';
        }
        return text.replace(/\s+/g, ' ').trim();
    };

    const summarizeText = (text, maxLength = 150) => {
        const normalized = normalizeText(text);
        if (normalized.length <= maxLength) {
            return normalized;
        }
        return `${normalized.slice(0, maxLength)}...`;
    };

    const normalizeVector = (values) => {
        if (!Array.isArray(values) || !values.length) {
            return null;
        }

        let magnitude = 0;
        for (let index = 0; index < values.length; index += 1) {
            const value = Number(values[index]);
            if (!Number.isFinite(value)) {
                return null;
            }
            magnitude += value * value;
        }

        if (!magnitude) {
            return null;
        }

        const normalized = new Float32Array(values.length);
        const divisor = Math.sqrt(magnitude);

        for (let index = 0; index < values.length; index += 1) {
            normalized[index] = Number(values[index]) / divisor;
        }

        return normalized;
    };

    const dotProduct = (left, right) => {
        if (!left || !right || left.length !== right.length) {
            return 0;
        }

        let score = 0;
        for (let index = 0; index < left.length; index += 1) {
            score += left[index] * right[index];
        }
        return score;
    };

    const buildInitialStateMarkup = () => `
        <div class="search-initial-state">
            <svg class="initial-icon" xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <path d="m21 21-4.35-4.35"></path>
            </svg>
            <h3>开始搜索</h3>
            <p>输入问题、主题或关键词，先做向量召回，再用 reranker 重排结果</p>
            <div class="quick-search-tags">
                ${QUICK_TAGS.map((tag) => `<span class="quick-tag">${escapeHtml(tag)}</span>`).join('')}
            </div>
        </div>
    `;

    const buildNoResultsMarkup = (query, modeLabel) => `
        <div class="no-results">
            <p>未找到与 "${escapeHtml(query)}" 高相关的内容</p>
            <p>当前模式：${escapeHtml(modeLabel)}</p>
            <ul>
                <li>尝试描述具体问题，而不是只输入单个词</li>
                <li>换一种更通用的表达方式</li>
                <li>检查拼写是否正确</li>
            </ul>
        </div>
    `;

    const buildErrorMarkup = (message) => `
        <div class="error-message">
            <p>搜索功能初始化失败</p>
            <p>错误信息：${escapeHtml(message)}</p>
            <p>请刷新页面重试</p>
        </div>
    `;

    const toSearchDocument = (item) => {
        const title = normalizeText(item && item.title);
        const permalink = normalizeText(item && item.permalink);
        const summary = normalizeText(item && item.summary);
        const content = normalizeText(item && item.content);
        const embedding = item && item.embedding instanceof Float32Array
            ? item.embedding
            : normalizeVector(item && item.embedding);

        return {
            title,
            permalink,
            summary,
            content,
            embedding
        };
    };

    const createFuseIndex = (documents) => {
        if (!window.Fuse || !Array.isArray(documents) || !documents.length) {
            return null;
        }

        return new window.Fuse(documents, {
            keys: ['title', 'summary', 'content'],
            threshold: 0.32,
            distance: 100,
            minMatchCharLength: 2,
            includeScore: true,
            ignoreLocation: true
        });
    };

    document.addEventListener('DOMContentLoaded', async () => {
        const config = {
            ...DEFAULT_CONFIG,
            ...(window.siteSearchConfig || {})
        };

        const maxResults = Math.max(1, Number(config.maxResults) || DEFAULT_CONFIG.maxResults);
        const recallK = Math.max(maxResults, Number(config.recallK) || DEFAULT_CONFIG.recallK);
        const rerankTopN = Math.max(1, Math.min(recallK, Number(config.rerankTopN) || maxResults));
        const rerankEnabled = Boolean(config.rerankEnabled && config.rerankEndpoint && config.rerankModel);
        const rerankDocumentMaxChars = Math.max(200, Number(config.rerankDocumentMaxChars) || DEFAULT_CONFIG.rerankDocumentMaxChars);
        const minQueryLength = Math.max(1, Number(config.minQueryLength) || DEFAULT_CONFIG.minQueryLength);
        const scoreThreshold = Number(config.scoreThreshold) || DEFAULT_CONFIG.scoreThreshold;

        const searchInput = document.getElementById('searchQuery');
        const searchResults = document.getElementById('searchResults');
        const resultCount = document.getElementById('resultCount');
        const searchTime = document.getElementById('searchTime');
        const clearButton = document.getElementById('clearSearch');
        const searchMode = document.getElementById('searchMode');

        if (!searchInput || !searchResults || !resultCount || !searchTime) {
            return;
        }

        const loadingIndicator = document.createElement('div');
        loadingIndicator.textContent = '加载中...';
        loadingIndicator.className = 'search-loading-indicator';
        loadingIndicator.hidden = true;
        const container = searchResults.parentElement || document.body;
        container.insertBefore(loadingIndicator, searchResults);

        let vectorIndex = null;
        let fuse = null;
        let activeSearchToken = 0;
        let activeEmbeddingController = null;
        let activeRerankController = null;

        const defaultModeLabel = () => {
            if (vectorIndex) {
                return rerankEnabled ? '向量召回 + 重排' : '向量召回';
            }
            return '关键词';
        };

        const abortActiveRequests = () => {
            if (activeEmbeddingController) {
                activeEmbeddingController.abort();
                activeEmbeddingController = null;
            }

            if (activeRerankController) {
                activeRerankController.abort();
                activeRerankController = null;
            }
        };

        const setModeLabel = (label) => {
            if (searchMode) {
                searchMode.textContent = label;
            }
        };

        const resetStats = () => {
            resultCount.textContent = '0';
            searchTime.textContent = '0';
            setModeLabel(defaultModeLabel());
        };

        const attachResultNavigation = () => {
            document.querySelectorAll('.search-result-item').forEach((item) => {
                item.addEventListener('click', (event) => {
                    if (event.target.tagName === 'A') {
                        return;
                    }

                    const url = item.getAttribute('data-url');
                    if (url) {
                        window.location.href = url;
                    }
                });
            });
        };

        const bindQuickTags = () => {
            document.querySelectorAll('.quick-tag').forEach((tag) => {
                if (tag.dataset.bound === 'true') {
                    return;
                }

                tag.dataset.bound = 'true';
                tag.addEventListener('click', () => {
                    searchInput.value = tag.textContent || '';
                    if (clearButton) {
                        clearButton.hidden = false;
                    }
                    void performSearch(searchInput.value);
                });
            });
        };

        const renderInitialState = () => {
            searchResults.innerHTML = buildInitialStateMarkup();
            searchResults.style.opacity = '1';
            bindQuickTags();
            resetStats();
        };

        const buildScoreMarkup = (result) => {
            const parts = [
                `<span class="result-score">${escapeHtml(result.scoreLabel || '得分')}: ${(clampScore(result.score) * 100).toFixed(1)}%</span>`
            ];

            if (
                result.scoreLabel === '重排得分' &&
                Number.isFinite(result.vectorScore)
            ) {
                parts.push(`<span class="result-score">召回分: ${(clampScore(result.vectorScore) * 100).toFixed(1)}%</span>`);
            }

            return parts.join('');
        };

        const renderResults = (query, results, modeLabel, duration) => {
            resultCount.textContent = String(results.length);
            searchTime.textContent = String(duration);
            setModeLabel(modeLabel);

            if (!results.length) {
                searchResults.innerHTML = buildNoResultsMarkup(query, modeLabel);
                searchResults.style.opacity = '1';
                return;
            }

            const markup = results.map((result) => {
                const item = result.item;
                const summary = summarizeText(item.summary || item.content, 160);
                const safeTitle = escapeHtml(item.title || '未命名内容');
                const safeSummary = escapeHtml(summary || '暂无摘要');
                const safePermalink = escapeHtml(item.permalink || '#');

                return `
                    <article class="search-result-item" data-url="${safePermalink}">
                        <h3 class="result-title">
                            <a href="${safePermalink}">${safeTitle}</a>
                        </h3>
                        <p class="result-summary">${safeSummary}</p>
                        <div class="result-meta">
                            ${buildScoreMarkup(result)}
                        </div>
                    </article>
                `;
            }).join('');

            searchResults.innerHTML = markup;
            searchResults.style.opacity = '1';
            attachResultNavigation();
        };

        const keywordSearch = (query) => {
            if (!fuse) {
                return [];
            }

            return fuse
                .search(query.trim(), { limit: maxResults })
                .map((entry) => ({
                    item: entry.item,
                    score: 1 - (Number(entry.score) || 1),
                    scoreLabel: '关键词匹配度'
                }));
        };

        const buildRerankDocument = (item) => [
            item.title ? `标题：${item.title}` : '',
            item.summary ? `摘要：${item.summary}` : '',
            item.content ? `正文：${item.content}` : ''
        ]
            .filter(Boolean)
            .join('\n\n')
            .slice(0, rerankDocumentMaxChars);

        const extractEmbedding = (payload) => payload
            && payload.data
            && payload.data[0]
            && payload.data[0].embedding;

        const extractRerankResults = (payload) => {
            if (Array.isArray(payload && payload.results)) {
                return payload.results;
            }

            if (Array.isArray(payload && payload.data)) {
                return payload.data;
            }

            return [];
        };

        const fetchQueryEmbedding = async (query) => {
            if (!vectorIndex || !vectorIndex.documents.length) {
                throw new Error('向量索引不可用');
            }

            if (!config.embeddingEndpoint) {
                throw new Error('Embedding 接口未配置');
            }

            activeEmbeddingController = new AbortController();

            const response = await fetch(config.embeddingEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    input: query.trim(),
                    model: vectorIndex.model || config.embeddingModel
                }),
                signal: activeEmbeddingController.signal
            });

            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                const message = payload && (payload.message || (payload.error && payload.error.message) || payload.error);
                throw new Error(message || `Embedding 请求失败: HTTP ${response.status}`);
            }

            const queryVector = normalizeVector(extractEmbedding(payload));
            if (!queryVector) {
                throw new Error('未获取到有效的查询向量');
            }

            return queryVector;
        };

        const recallCandidates = (queryVector) => vectorIndex.documents
            .map((item) => {
                const vectorScore = dotProduct(queryVector, item.embedding);
                return {
                    item,
                    score: vectorScore,
                    vectorScore,
                    scoreLabel: '语义相似度'
                };
            })
            .filter((entry) => Number.isFinite(entry.score))
            .sort((left, right) => right.score - left.score)
            .slice(0, recallK);

        const selectVectorResults = (candidates) => candidates
            .filter((entry, index) => entry.score >= scoreThreshold || index < 3)
            .slice(0, maxResults)
            .map((entry) => ({
                ...entry,
                score: clampScore(entry.score),
                scoreLabel: '语义相似度'
            }));

        const rerankCandidates = async (query, candidates) => {
            if (!rerankEnabled || candidates.length < 2) {
                return null;
            }

            activeRerankController = new AbortController();

            const response = await fetch(config.rerankEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: config.rerankModel,
                    query: query.trim(),
                    documents: candidates.map((candidate) => buildRerankDocument(candidate.item)),
                    top_n: Math.min(rerankTopN, candidates.length)
                }),
                signal: activeRerankController.signal
            });

            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                const message = payload && (payload.message || (payload.error && payload.error.message) || payload.error);
                throw new Error(message || `Rerank 请求失败: HTTP ${response.status}`);
            }

            const results = extractRerankResults(payload);
            if (!results.length) {
                throw new Error('Rerank 返回为空');
            }

            return results
                .map((entry) => {
                    const index = Number(entry && entry.index);
                    if (!Number.isInteger(index) || index < 0 || index >= candidates.length) {
                        return null;
                    }

                    const candidate = candidates[index];
                    const rerankScore = clampScore(entry && entry.relevance_score);

                    return {
                        item: candidate.item,
                        score: rerankScore,
                        rerankScore,
                        vectorScore: clampScore(candidate.vectorScore),
                        scoreLabel: '重排得分'
                    };
                })
                .filter(Boolean)
                .sort((left, right) => right.score - left.score)
                .slice(0, maxResults);
        };

        const semanticSearch = async (query) => {
            const queryVector = await fetchQueryEmbedding(query);
            const candidates = recallCandidates(queryVector);

            if (!candidates.length) {
                return {
                    results: [],
                    modeLabel: '向量召回'
                };
            }

            try {
                const rerankedResults = await rerankCandidates(query, candidates);
                if (rerankedResults && rerankedResults.length) {
                    return {
                        results: rerankedResults,
                        modeLabel: '向量召回 + 重排'
                    };
                }
            } catch (error) {
                if (error && error.name === 'AbortError') {
                    throw error;
                }

                console.warn('Rerank failed, falling back to vector-only ranking.', error);
            } finally {
                activeRerankController = null;
            }

            return {
                results: selectVectorResults(candidates),
                modeLabel: '向量召回'
            };
        };

        const performSearch = async (rawQuery) => {
            const query = rawQuery.trim();
            activeSearchToken += 1;
            const currentToken = activeSearchToken;

            abortActiveRequests();

            if (!query || query.length < minQueryLength) {
                loadingIndicator.hidden = true;
                renderInitialState();
                return;
            }

            loadingIndicator.textContent = vectorIndex
                ? (rerankEnabled ? '召回与重排中...' : '语义检索中...')
                : '关键词检索中...';
            loadingIndicator.hidden = false;
            searchResults.style.opacity = '0';

            await wait(150);
            if (currentToken !== activeSearchToken) {
                return;
            }

            const searchStart = performance.now();
            let results = [];
            let modeLabel = defaultModeLabel();

            try {
                if (vectorIndex) {
                    const semanticResult = await semanticSearch(query);
                    results = semanticResult.results;
                    modeLabel = semanticResult.modeLabel;

                    if (!results.length && fuse) {
                        results = keywordSearch(query);
                        modeLabel = '关键词补充';
                    }
                } else {
                    results = keywordSearch(query);
                    modeLabel = '关键词';
                }
            } catch (error) {
                if (error && error.name === 'AbortError') {
                    return;
                }

                console.warn('Semantic search failed, falling back to keyword search.', error);

                if (fuse) {
                    results = keywordSearch(query);
                    modeLabel = '关键词降级';
                } else {
                    searchResults.innerHTML = buildErrorMarkup(error instanceof Error ? error.message : '未知错误');
                    searchResults.style.opacity = '1';
                    loadingIndicator.hidden = true;
                    setModeLabel('不可用');
                    return;
                }
            } finally {
                activeEmbeddingController = null;
                activeRerankController = null;
            }

            if (currentToken !== activeSearchToken) {
                return;
            }

            const duration = Math.round(performance.now() - searchStart);
            renderResults(query, results, modeLabel, duration);
            loadingIndicator.hidden = true;
        };

        if (clearButton) {
            clearButton.addEventListener('click', () => {
                activeSearchToken += 1;
                abortActiveRequests();

                searchInput.value = '';
                clearButton.hidden = true;
                loadingIndicator.hidden = true;
                renderInitialState();
                searchInput.focus();
            });
        }

        searchInput.addEventListener('input', (event) => {
            if (clearButton) {
                clearButton.hidden = !event.target.value.trim();
            }
        });

        try {
            loadingIndicator.hidden = false;
            loadingIndicator.textContent = '正在加载搜索索引...';

            const response = await fetch(config.vectorIndexFile, { headers: { Accept: 'application/json' } });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const payload = await response.json();
            const documents = Array.isArray(payload && payload.documents) ? payload.documents.map(toSearchDocument) : [];
            const validDocuments = documents.filter((item) => item.permalink);
            const semanticDocuments = validDocuments.filter((item) => item.embedding);

            if (!semanticDocuments.length) {
                throw new Error('向量索引中没有有效 embedding');
            }

            vectorIndex = {
                model: payload.model || config.embeddingModel || '',
                dimensions: payload.dimensions || 0,
                documents: semanticDocuments
            };
            fuse = createFuseIndex(validDocuments);
            setModeLabel(defaultModeLabel());

            console.log(`搜索向量索引加载完成，共 ${vectorIndex.documents.length} 条记录`);
        } catch (vectorError) {
            console.warn('Vector search index unavailable, falling back to keyword search.', vectorError);

            try {
                const response = await fetch(config.keywordIndexFile, { headers: { Accept: 'application/json' } });
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const keywordDocuments = (await response.json())
                    .map(toSearchDocument)
                    .filter((item) => item.permalink);

                if (!keywordDocuments.length) {
                    throw new Error('关键词索引为空');
                }

                fuse = createFuseIndex(keywordDocuments);
                setModeLabel('关键词');
            } catch (keywordError) {
                searchResults.innerHTML = buildErrorMarkup(keywordError instanceof Error ? keywordError.message : '未知错误');
                searchResults.style.opacity = '1';
                loadingIndicator.hidden = true;
                setModeLabel('不可用');
                return;
            }
        } finally {
            loadingIndicator.hidden = true;
        }

        const debouncedSearch = debounce((value) => {
            void performSearch(value);
        }, 350);

        searchInput.addEventListener('input', (event) => {
            debouncedSearch(event.target.value);
        });

        if (clearButton) {
            clearButton.hidden = !searchInput.value.trim();
        }

        renderInitialState();

        const queryParam = new URLSearchParams(window.location.search).get('q');
        if (queryParam) {
            searchInput.value = queryParam;
            if (clearButton) {
                clearButton.hidden = false;
            }
            void performSearch(queryParam);
        }
    });
})();
