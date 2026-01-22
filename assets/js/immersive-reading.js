/* ================= 沉浸式阅读 - JavaScript 增强 ================= */

(function() {
    'use strict';

    // 阅读进度条
    class ReadingProgress {
        constructor() {
            this.init();
        }

        init() {
            const progressBar = document.createElement('div');
            progressBar.className = 'reading-progress';
            progressBar.id = 'reading-progress';
            document.body.appendChild(progressBar);

            window.addEventListener('scroll', () => this.updateProgress());
        }

        updateProgress() {
            const progressBar = document.getElementById('reading-progress');
            if (!progressBar) return;

            const scrollTop = window.scrollY;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;

            progressBar.style.width = scrollPercent + '%';
        }
    }

    // 平滑滚动到标题
    class SmoothScroll {
        constructor() {
            this.init();
        }

        init() {
            document.addEventListener('click', (e) => {
                const target = e.target.closest('a[href^="#"]');
                if (!target) return;

                const id = target.getAttribute('href').slice(1);
                const element = document.getElementById(id);
                if (!element) return;

                e.preventDefault();
                element.scrollIntoView({ behavior: 'smooth' });
            });
        }
    }

    // 代码块复制功能增强
    class CodeCopyButton {
        constructor() {
            this.init();
        }

        init() {
            document.querySelectorAll('pre').forEach((pre) => {
                if (pre.querySelector('.copy-code')) return;

                const button = document.createElement('button');
                button.className = 'copy-code';
                button.textContent = '复制代码';
                button.type = 'button';

                button.addEventListener('click', () => this.copyCode(pre, button));
                pre.appendChild(button);
            });
        }

        copyCode(pre, button) {
            const code = pre.querySelector('code');
            const text = code.textContent;

            navigator.clipboard.writeText(text).then(() => {
                const originalText = button.textContent;
                button.textContent = '已复制!';
                setTimeout(() => {
                    button.textContent = originalText;
                }, 2000);
            }).catch(() => {
                button.textContent = '复制失败';
            });
        }
    }

    // 图片放大功能
    class ImageZoom {
        constructor() {
            this.init();
        }

        init() {
            document.querySelectorAll('.post-content img').forEach((img) => {
                img.style.cursor = 'zoom-in';
                img.addEventListener('click', () => this.showModal(img));
            });
        }

        showModal(img) {
            const modal = document.createElement('div');
            modal.className = 'image-modal';
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.9);
                z-index: 9999;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: fadeIn 0.3s ease;
            `;

            const image = document.createElement('img');
            image.src = img.src;
            image.style.cssText = `
                max-width: 90vw;
                max-height: 90vh;
                object-fit: contain;
                animation: zoomIn 0.3s ease;
            `;

            modal.appendChild(image);

            modal.addEventListener('click', (e) => {
                if (e.target === modal) modal.remove();
            });

            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') modal.remove();
            });

            document.body.appendChild(modal);
        }
    }

    // 目录自动生成和高亮
    class TableOfContents {
        constructor() {
            this.init();
        }

        init() {
            const headings = document.querySelectorAll('.post-content h2, .post-content h3');
            if (headings.length === 0) return;

            const toc = document.querySelector('.toc');
            if (!toc) return;

            // 为标题添加ID
            headings.forEach((heading, index) => {
                if (!heading.id) {
                    heading.id = `heading-${index}`;
                }
            });

            // 监听滚动，高亮当前章节
            window.addEventListener('scroll', () => this.highlightTOC(headings));
        }

        highlightTOC(headings) {
            let current = null;

            headings.forEach((heading) => {
                const rect = heading.getBoundingClientRect();
                if (rect.top <= 100) {
                    current = heading;
                }
            });

            document.querySelectorAll('.toc a').forEach((link) => {
                link.classList.remove('active');
            });

            if (current) {
                const link = document.querySelector(`.toc a[href="#${current.id}"]`);
                if (link) {
                    link.classList.add('active');
                    link.style.color = 'var(--primary)';
                }
            }
        }
    }

    // 焦点阅读模式切换
    class FocusMode {
        constructor() {
            this.init();
        }

        init() {
            const button = this.createFocusButton();
            if (!button) return;

            button.addEventListener('click', () => this.toggleFocusMode());
        }

        createFocusButton() {
            const header = document.querySelector('header');
            if (!header) return null;

            const button = document.createElement('button');
            button.textContent = '焦点模式';
            button.className = 'focus-mode-toggle';
            button.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 999;
                padding: 10px 16px;
                border-radius: 50px;
                border: none;
                background: var(--secondary);
                color: white;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
            `;

            button.addEventListener('mouseenter', () => {
                button.style.transform = 'scale(1.1)';
            });

            button.addEventListener('mouseleave', () => {
                button.style.transform = 'scale(1)';
            });

            document.body.appendChild(button);
            return button;
        }

        toggleFocusMode() {
            document.body.classList.toggle('focus-mode');
            const button = document.querySelector('.focus-mode-toggle');
            if (button) {
                button.textContent = document.body.classList.contains('focus-mode')
                    ? '退出焦点'
                    : '焦点模式';
            }

            // 保存到本地存储
            const isFocusMode = document.body.classList.contains('focus-mode');
            localStorage.setItem('focusMode', isFocusMode);
        }
    }

    // 初始化所有功能
    class ImmersiveReading {
        init() {
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', () => this.setup());
            } else {
                this.setup();
            }
        }

        setup() {
            // 恢复焦点模式设置
            if (localStorage.getItem('focusMode') === 'true') {
                document.body.classList.add('focus-mode');
            }

            // 初始化各个功能
            new ReadingProgress();
            new SmoothScroll();
            new CodeCopyButton();
            new ImageZoom();
            new TableOfContents();
            new FocusMode();
        }
    }

    // 启动
    new ImmersiveReading().init();
})();
