// 防抖控制（300ms）
let searchTimer;
const debounceSearch = (callback) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(callback, 300);
};

// Fuse.js配置
const fuseOptions = {
  keys: [
    { name: 'title', weight: 0.6 },
    { name: 'content', weight: 0.3 },
    { name: 'tags', weight: 0.1 }
  ],
  includeMatches: true,
  minMatchCharLength: 2,
  threshold: 0.2,
  ignoreLocation: true,
  useExtendedSearch: true
};

// 动态加载索引
window.addEventListener('load', async () => {
  const searchQuery = document.getElementById('searchQuery');
  const resultsContainer = document.getElementById('searchResults');
  const metaContainer = document.getElementById('searchMeta');
  
  try {
    const response = await fetch('/index.json');
    const pages = await response.json();
    const fuse = new Fuse(pages, fuseOptions);

    // 输入监听
    searchQuery.addEventListener('input', (e) => {
      debounceSearch(() => {
        const startTime = performance.now();
        const query = e.target.value.trim();
        
        if (query.length < 2) {
          resultsContainer.innerHTML = '';
          return;
        }

        const results = fuse.search(query);
        renderResults(results, performance.now() - startTime);
      });
    });

  } catch (error) {
    console.error('搜索初始化失败:', error);
    resultsContainer.innerHTML = `<div class="error">搜索功能暂时不可用</div>`;
  }
});

// 结果渲染
function renderResults(results, duration) {
  const resultsHTML = results.map(({ item, matches }) => `
    <article class="result-card">
      <h3>${highlightMatches(item.title, matches)}</h3>
      <div class="excerpt">${getBestExcerpt(item.content, matches)}</div>
      <footer>
        <time>${new Date(item.date).toLocaleDateString()}</time>
        <a href="${item.permalink}" class="read-more">阅读全文 →</a>
      </footer>
    </article>
  `).join('');

  document.getElementById('resultCount').textContent = results.length;
  document.getElementById('searchTime').textContent = duration.toFixed(1);
  document.getElementById('searchResults').innerHTML = resultsHTML || '<p>未找到匹配内容</p>';
}

// 智能高亮
function highlightMatches(text, matches) {
  let highlighted = text;
  matches.forEach(({ indices }) => {
    indices.reverse().forEach(([start, end]) => {
      const segment = text.substring(start, end + 1);
      highlighted = highlighted.replace(segment, `<mark class="search-highlight">${segment}</mark>`);
    });
  });
  return highlighted;
}

// 最佳摘要提取
function getBestExcerpt(content, matches) {
  const contentMatches = matches.filter(m => m.key === 'content');
  if (!contentMatches.length) return content.substr(0, 150) + '...';

  const { indices } = contentMatches[0];
  const [start] = indices[0];
  return content.substr(Math.max(0, start - 50), 300) + '...';
}
// 修改事件监听逻辑
const safeAddListener = (element, event, handler) => {
    if (!element || !handler) return;
    
    const wrappedHandler = async (e) => {
      try {
        await handler(e);
      } catch (err) {
        console.error('异步处理失败:', err);
      }
    };
  
    element.removeEventListener(event, handler);
    element.addEventListener(event, wrappedHandler);
  };
  
  // 使用安全监听器
  safeAddListener(document.getElementById('searchQuery'), 'input', debouncedSearch);