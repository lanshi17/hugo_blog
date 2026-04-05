/* ================= 沉浸式阅读 - JavaScript 增强 ================= */

(function () {
    'use strict';

    const STORAGE_KEY = 'immersive-reading-focus-mode';
    const DEFAULT_CONFIG = {
        readingProgressBar: true,
        progressPosition: 'top',
        focusMode: true,
        labels: {
            toolbar: 'Reading tools',
            enterFocus: 'Focus mode',
            exitFocus: 'Exit focus mode',
            closeImage: 'Close preview',
            openImage: 'Open image preview',
            imagePreview: 'Image preview'
        }
    };

    const FOCUSABLE_SELECTOR = [
        'a[href]',
        'area[href]',
        'button:not([disabled])',
        'input:not([disabled])',
        'select:not([disabled])',
        'textarea:not([disabled])',
        'iframe',
        '[tabindex]:not([tabindex="-1"])',
        '[contenteditable="true"]'
    ].join(',');

    function normalizeStringValue(value) {
        if (typeof value !== 'string') {
            return value;
        }

        const trimmed = value.trim();
        if (trimmed.length >= 2) {
            const first = trimmed[0];
            const last = trimmed[trimmed.length - 1];
            if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
                return trimmed.slice(1, -1);
            }
        }

        return trimmed;
    }

    function getConfig() {
        const provided = window.immersiveReadingConfig || {};
        const providedLabels = Object.fromEntries(
            Object.entries(provided.labels || {}).map(([key, value]) => [key, normalizeStringValue(value)])
        );

        return {
            ...DEFAULT_CONFIG,
            ...provided,
            progressPosition: normalizeStringValue(provided.progressPosition) || DEFAULT_CONFIG.progressPosition,
            labels: {
                ...DEFAULT_CONFIG.labels,
                ...providedLabels
            }
        };
    }

    function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }

    function slugify(value, index) {
        const slug = value
            .trim()
            .toLowerCase()
            .replace(/[^\w\u4e00-\u9fa5\s-]/g, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-');

        return slug || `section-${index}`;
    }

    function readPreference(key) {
        try {
            return window.localStorage.getItem(key);
        } catch (error) {
            console.warn('[immersive-reading] Failed to read preference.', error);
            return null;
        }
    }

    function writePreference(key, value) {
        try {
            window.localStorage.setItem(key, value);
        } catch (error) {
            console.warn('[immersive-reading] Failed to save preference.', error);
        }
    }

    function getFocusableElements(root) {
        return Array.from(root.querySelectorAll(FOCUSABLE_SELECTOR)).filter((node) => {
            if (!(node instanceof HTMLElement)) {
                return false;
            }

            if (node.hidden || node.getAttribute('aria-hidden') === 'true') {
                return false;
            }

            return !node.closest('[inert]');
        });
    }

    class ReadingProgress {
        constructor(content, config) {
            this.content = content;
            this.config = config;
            this.frame = 0;
            this.element = null;
            this.bar = null;
            this.onScroll = this.onScroll.bind(this);
            this.update = this.update.bind(this);
            this.init();
        }

        init() {
            this.element = document.createElement('div');
            this.element.className = 'reading-progress';
            this.element.dataset.position = this.config.progressPosition === 'bottom' ? 'bottom' : 'top';

            this.bar = document.createElement('span');
            this.bar.className = 'reading-progress__bar';

            this.element.appendChild(this.bar);
            document.body.appendChild(this.element);

            window.addEventListener('scroll', this.onScroll, { passive: true });
            window.addEventListener('resize', this.onScroll, { passive: true });

            this.update();
            requestAnimationFrame(() => {
                if (this.element) {
                    this.element.classList.add('is-visible');
                }
            });
        }

        onScroll() {
            if (this.frame) {
                return;
            }

            this.frame = window.requestAnimationFrame(this.update);
        }

        update() {
            this.frame = 0;

            if (!this.bar) {
                return;
            }

            const rect = this.content.getBoundingClientRect();
            const total = Math.max(this.content.scrollHeight - window.innerHeight * 0.55, 1);
            const consumed = clamp(window.innerHeight * 0.22 - rect.top, 0, total);
            const progress = clamp(consumed / total, 0, 1);

            this.bar.style.setProperty('--reading-progress', progress.toFixed(4));
        }
    }

    class TableOfContents {
        constructor(content, toc) {
            this.content = content;
            this.toc = toc;
            this.links = [];
            this.items = [];
            this.frame = 0;
            this.activeId = '';
            this.onScroll = this.onScroll.bind(this);
            this.update = this.update.bind(this);
            this.init();
        }

        init() {
            this.links = Array.from(this.toc.querySelectorAll('a[href^="#"]'));
            if (!this.links.length) {
                return;
            }

            this.items = this.links
                .map((link, index) => {
                    const id = decodeURIComponent(link.getAttribute('href').slice(1));
                    let heading = document.getElementById(id);

                    if (!heading) {
                        return null;
                    }

                    if (!heading.id) {
                        heading.id = slugify(heading.textContent || '', index);
                    }

                    return { link, heading };
                })
                .filter(Boolean);

            if (!this.items.length) {
                return;
            }

            window.addEventListener('scroll', this.onScroll, { passive: true });
            window.addEventListener('resize', this.onScroll, { passive: true });
            this.update();
        }

        onScroll() {
            if (this.frame) {
                return;
            }

            this.frame = window.requestAnimationFrame(this.update);
        }

        update() {
            this.frame = 0;

            const threshold = window.innerHeight * 0.24;
            let current = this.items[0];

            this.items.forEach((item) => {
                if (item.heading.getBoundingClientRect().top <= threshold) {
                    current = item;
                }
            });

            if (!current || current.heading.id === this.activeId) {
                return;
            }

            this.activeId = current.heading.id;
            this.items.forEach((item) => {
                const isActive = item.heading.id === this.activeId;
                item.link.classList.toggle('is-active', isActive);

                if (isActive) {
                    item.link.setAttribute('aria-current', 'true');
                } else {
                    item.link.removeAttribute('aria-current');
                }
            });
        }
    }

    class FocusMode {
        constructor(config, article) {
            this.config = config;
            this.article = article;
            this.button = null;
            this.enabled = false;
            this.hiddenRegions = [];
            this.hiddenState = new Map();
            this.regionObserver = null;
            this.init();
        }

        init() {
            const toolbar = document.createElement('div');
            toolbar.className = 'reading-toolbar';
            toolbar.setAttribute('role', 'toolbar');
            toolbar.setAttribute('aria-label', this.config.labels.toolbar);

            this.button = document.createElement('button');
            this.button.type = 'button';
            this.button.className = 'reading-toolbar__button';
            this.button.addEventListener('click', () => this.toggle());

            toolbar.appendChild(this.button);
            document.body.appendChild(toolbar);

            this.hiddenRegions = this.collectHiddenRegions();
            this.startRegionObserver();

            this.enabled = readPreference(STORAGE_KEY) === 'true';
            this.apply(this.enabled, false);
        }

        collectHiddenRegions() {
            const regions = new Set([
                this.article.querySelector(':scope > .toc'),
                this.article.querySelector(':scope > .post-footer'),
                this.article.querySelector('.giscus'),
                this.article.querySelector('iframe.giscus-frame'),
                document.querySelector('.footer'),
                document.querySelector('.top-link'),
                document.querySelector('.header')
            ]);

            return Array.from(regions).filter(Boolean);
        }

        startRegionObserver() {
            if (!('MutationObserver' in window)) {
                return;
            }

            this.regionObserver = new MutationObserver(() => {
                if (!this.enabled) {
                    return;
                }

                const updatedRegions = this.collectHiddenRegions();
                updatedRegions.forEach((region) => {
                    if (!this.hiddenState.has(region)) {
                        this.setRegionState(region, true);
                    }
                });
                this.hiddenRegions = updatedRegions;
            });

            this.regionObserver.observe(this.article, {
                childList: true,
                subtree: true
            });
        }

        setRegionState(region, hidden) {
            if (hidden) {
                if (!this.hiddenState.has(region)) {
                    const focusableState = getFocusableElements(region).map((node) => ({
                        node,
                        previousTabIndex: node.getAttribute('tabindex')
                    }));

                    this.hiddenState.set(region, {
                        previousAriaHidden: region.getAttribute('aria-hidden'),
                        previousInert: 'inert' in region ? region.inert : null,
                        focusableState
                    });
                }

                region.setAttribute('aria-hidden', 'true');

                if ('inert' in region) {
                    region.inert = true;
                }

                const state = this.hiddenState.get(region);
                state.focusableState.forEach(({ node }) => {
                    if (node.isConnected) {
                        node.setAttribute('tabindex', '-1');
                    }
                });

                return;
            }

            const state = this.hiddenState.get(region);
            if (!state) {
                region.removeAttribute('aria-hidden');
                if ('inert' in region) {
                    region.inert = false;
                }
                return;
            }

            if (state.previousAriaHidden === null) {
                region.removeAttribute('aria-hidden');
            } else {
                region.setAttribute('aria-hidden', state.previousAriaHidden);
            }

            if ('inert' in region && state.previousInert !== null) {
                region.inert = state.previousInert;
            }

            state.focusableState.forEach(({ node, previousTabIndex }) => {
                if (!node.isConnected) {
                    return;
                }

                if (previousTabIndex === null) {
                    node.removeAttribute('tabindex');
                } else {
                    node.setAttribute('tabindex', previousTabIndex);
                }
            });

            this.hiddenState.delete(region);
        }

        toggle() {
            this.apply(!this.enabled, true);
        }

        apply(enabled, persist) {
            this.enabled = enabled;
            document.documentElement.classList.toggle('focus-mode', enabled);
            document.body.classList.toggle('focus-mode', enabled);

            if (enabled) {
                this.hiddenRegions = this.collectHiddenRegions();
                this.hiddenRegions.forEach((region) => this.setRegionState(region, true));
            } else {
                Array.from(this.hiddenState.keys()).forEach((region) => this.setRegionState(region, false));
                this.hiddenRegions = [];
            }

            if (this.button) {
                this.button.textContent = enabled
                    ? this.config.labels.exitFocus
                    : this.config.labels.enterFocus;
                this.button.setAttribute('aria-pressed', String(enabled));
            }

            if (persist) {
                writePreference(STORAGE_KEY, String(enabled));
            }
        }
    }

    class ImageZoom {
        constructor(content, config) {
            this.content = content;
            this.config = config;
            this.viewer = null;
            this.lastTrigger = null;
            this.overlayState = [];
            this.onKeyDown = this.onKeyDown.bind(this);
            this.init();
        }

        init() {
            const images = Array.from(this.content.querySelectorAll('img')).filter((img) => {
                const width = img.naturalWidth || img.clientWidth;
                const height = img.naturalHeight || img.clientHeight;
                return !img.closest('a') && Math.max(width, height) >= 160;
            });

            images.forEach((img) => {
                img.classList.add('is-zoomable');
                img.tabIndex = img.tabIndex >= 0 ? img.tabIndex : 0;
                img.setAttribute('role', 'button');
                img.setAttribute('aria-haspopup', 'dialog');
                const label = img.alt
                    ? `${this.config.labels.openImage}: ${img.alt}`
                    : this.config.labels.openImage;
                img.setAttribute('aria-label', label);

                img.addEventListener('click', () => this.open(img));
                img.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        this.open(img);
                    }
                });
            });
        }

        open(sourceImage) {
            if (this.viewer) {
                this.close();
            }

            this.lastTrigger = sourceImage;

            const viewer = document.createElement('div');
            viewer.className = 'reading-image-viewer';
            viewer.setAttribute('role', 'dialog');
            viewer.setAttribute('aria-modal', 'true');
            viewer.setAttribute('aria-label', sourceImage.alt || this.config.labels.imagePreview);

            const panel = document.createElement('div');
            panel.className = 'reading-image-viewer__panel';

            const closeButton = document.createElement('button');
            closeButton.type = 'button';
            closeButton.className = 'reading-image-viewer__close';
            closeButton.textContent = this.config.labels.closeImage;
            closeButton.addEventListener('click', () => this.close());

            const image = document.createElement('img');
            image.className = 'reading-image-viewer__image';
            image.src = sourceImage.currentSrc || sourceImage.src;
            image.alt = sourceImage.alt || '';

            panel.appendChild(closeButton);
            panel.appendChild(image);

            if (sourceImage.alt) {
                const caption = document.createElement('p');
                caption.className = 'reading-image-viewer__caption';
                caption.textContent = sourceImage.alt;
                panel.appendChild(caption);
            }

            viewer.appendChild(panel);
            viewer.addEventListener('click', (event) => {
                if (event.target === viewer) {
                    this.close();
                }
            });

            this.viewer = viewer;
            document.body.appendChild(viewer);
            document.body.classList.add('reading-image-viewer-open');
            this.setOverlayState(true);
            document.addEventListener('keydown', this.onKeyDown);

            closeButton.focus({ preventScroll: true });
        }

        setOverlayState(active) {
            if (active) {
                this.overlayState = Array.from(document.body.children)
                    .filter((node) => node instanceof HTMLElement && node !== this.viewer)
                    .map((node) => ({
                        node,
                        previousAriaHidden: node.getAttribute('aria-hidden'),
                        previousInert: 'inert' in node ? node.inert : null,
                        focusableState: getFocusableElements(node).map((focusable) => ({
                            node: focusable,
                            previousTabIndex: focusable.getAttribute('tabindex')
                        }))
                    }));

                this.overlayState.forEach(({ node, focusableState }) => {
                    node.setAttribute('aria-hidden', 'true');
                    if ('inert' in node) {
                        node.inert = true;
                    }

                    focusableState.forEach(({ node: focusable }) => {
                        if (focusable.isConnected) {
                            focusable.setAttribute('tabindex', '-1');
                        }
                    });
                });
                return;
            }

            this.overlayState.forEach(({ node, previousAriaHidden, previousInert, focusableState }) => {
                if (!node.isConnected) {
                    return;
                }

                if (previousAriaHidden === null) {
                    node.removeAttribute('aria-hidden');
                } else {
                    node.setAttribute('aria-hidden', previousAriaHidden);
                }

                if ('inert' in node && previousInert !== null) {
                    node.inert = previousInert;
                }

                focusableState.forEach(({ node: focusable, previousTabIndex }) => {
                    if (!focusable.isConnected) {
                        return;
                    }

                    if (previousTabIndex === null) {
                        focusable.removeAttribute('tabindex');
                    } else {
                        focusable.setAttribute('tabindex', previousTabIndex);
                    }
                });
            });

            this.overlayState = [];
        }

        trapFocus(event) {
            if (!this.viewer || event.key !== 'Tab') {
                return;
            }

            const focusables = getFocusableElements(this.viewer);
            if (!focusables.length) {
                event.preventDefault();
                return;
            }

            const first = focusables[0];
            const last = focusables[focusables.length - 1];
            const active = document.activeElement;

            if (!this.viewer.contains(active)) {
                event.preventDefault();
                first.focus();
                return;
            }

            if (event.shiftKey && active === first) {
                event.preventDefault();
                last.focus();
                return;
            }

            if (!event.shiftKey && active === last) {
                event.preventDefault();
                first.focus();
            }
        }

        onKeyDown(event) {
            if (event.key === 'Escape') {
                this.close();
                return;
            }

            this.trapFocus(event);
        }

        close() {
            if (!this.viewer) {
                return;
            }

            this.viewer.remove();
            this.viewer = null;
            document.body.classList.remove('reading-image-viewer-open');
            this.setOverlayState(false);
            document.removeEventListener('keydown', this.onKeyDown);

            if (this.lastTrigger) {
                this.lastTrigger.focus({ preventScroll: true });
            }
        }
    }

    class ImmersiveReading {
        constructor() {
            this.config = getConfig();
        }

        init() {
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', () => this.setup(), { once: true });
                return;
            }

            this.setup();
        }

        setup() {
            const article = document.querySelector('.post-single');
            const content = article ? article.querySelector('.post-content') : null;
            if (!article || !content) {
                return;
            }

            document.documentElement.classList.add('immersive-reading');
            document.body.classList.add('immersive-reading');

            if (this.config.readingProgressBar) {
                new ReadingProgress(content, this.config);
            }

            const toc = article.querySelector('.toc');
            if (toc) {
                new TableOfContents(content, toc);
            }

            if (this.config.focusMode) {
                new FocusMode(this.config, article);
            }

            new ImageZoom(content, this.config);
        }
    }

    new ImmersiveReading().init();
})();
