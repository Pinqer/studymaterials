import re
import json

# Read markdown file
with open('study_guide.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Properly escape for JSON
content_json = json.dumps(content)

# HTML template with in-page search bar
html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Study Guide</title>
    <!-- Highlight.js for syntax highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/tomorrow-night-bright.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
    <!-- Marked for markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
    <!-- MathJax for math rendering -->
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            },
            svg: {
                fontCache: 'global'
            }
        };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 60px 40px 20px 40px;
            background: #fff;
            color: #333;
        }
        /* Floating search bar */
        .search-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .search-bar input {
            flex: 1;
            max-width: 500px;
            padding: 8px 15px;
            border: none;
            border-radius: 20px;
            font-size: 14px;
            outline: none;
        }
        .search-bar button {
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        .search-bar button:hover {
            background: rgba(255,255,255,0.3);
        }
        .search-info {
            color: white;
            font-size: 14px;
            min-width: 100px;
        }
        /* Highlight for search matches */
        .search-highlight {
            background-color: #ffeb3b !important;
            color: #000 !important;
            padding: 2px 0;
            border-radius: 2px;
        }
        .search-highlight.current {
            background-color: #ff9800 !important;
        }
        h1 { 
            border-bottom: 3px solid #007acc; 
            padding-bottom: 10px; 
            color: #2c3e50;
            margin-top: 30px;
        }
        h2 { 
            border-bottom: 1px solid #ddd; 
            padding-bottom: 8px; 
            margin-top: 30px;
            color: #34495e;
        }
        h3 { 
            margin-top: 24px;
            color: #34495e;
        }
        code { 
            background: #f4f4f4; 
            padding: 2px 6px; 
            border-radius: 3px; 
            font-size: 0.9em;
            font-family: 'Consolas', 'Monaco', monospace;
            color: #c7254e;
        }
        pre { 
            background: #000; 
            padding: 16px; 
            border-radius: 6px; 
            overflow-x: auto;
            margin: 16px 0;
        }
        pre code { 
            background: none; 
            padding: 0; 
            font-size: 14px;
            line-height: 1.5;
        }
        .hljs {
            background: #000 !important;
            color: #eaeaea !important;
        }
        hr { 
            border: none; 
            border-top: 2px solid #eee; 
            margin: 32px 0; 
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 16px 0; 
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 10px 12px;
            text-align: left;
        }
        th { 
            background: #f5f5f5;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background: #fafafa;
        }
        blockquote {
            border-left: 4px solid #007acc;
            margin-left: 0;
            padding-left: 16px;
            color: #666;
            font-style: italic;
        }
        ul, ol {
            padding-left: 30px;
        }
        li {
            margin: 8px 0;
        }
        strong {
            color: #2c3e50;
        }
        .mjx-chtml {
            font-size: 110% !important;
        }
    </style>
</head>
<body>
    <!-- Floating Search Bar -->
    <div class="search-bar">
        <input type="text" id="searchInput" placeholder="Search in page (Ctrl+F)..." onkeydown="if(event.key==='Enter'){findNext()}">
        <button onclick="findPrev()">◀ Prev</button>
        <button onclick="findNext()">Next ▶</button>
        <button onclick="clearSearch()">Clear</button>
        <span class="search-info" id="searchInfo"></span>
    </div>
    
    <div id="content"></div>
    
    <script>
        const markdownContent = """ + content_json + """;
        
        // Configure marked
        marked.setOptions({
            breaks: true,
            gfm: true
        });
        
        // Render markdown
        document.getElementById('content').innerHTML = marked.parse(markdownContent);
        
        // Apply syntax highlighting
        hljs.highlightAll();
        
        // Trigger MathJax
        if (window.MathJax) {
            MathJax.typesetPromise().then(() => {
                console.log('MathJax rendering complete');
            }).catch((err) => console.log('MathJax error:', err));
        }
        
        // In-page search functionality
        let searchMatches = [];
        let currentMatchIndex = -1;
        let originalContent = '';
        
        function saveOriginalContent() {
            if (!originalContent) {
                originalContent = document.getElementById('content').innerHTML;
            }
        }
        
        function clearSearch() {
            if (originalContent) {
                document.getElementById('content').innerHTML = originalContent;
                hljs.highlightAll();
                if (window.MathJax) {
                    MathJax.typesetPromise();
                }
            }
            searchMatches = [];
            currentMatchIndex = -1;
            document.getElementById('searchInfo').textContent = '';
            document.getElementById('searchInput').value = '';
        }
        
        function findMatches(query) {
            saveOriginalContent();
            
            if (!query || query.length < 2) {
                clearSearch();
                return;
            }
            
            // Restore original first
            document.getElementById('content').innerHTML = originalContent;
            
            const contentEl = document.getElementById('content');
            const walker = document.createTreeWalker(contentEl, NodeFilter.SHOW_TEXT, null, false);
            const textNodes = [];
            let node;
            
            while (node = walker.nextNode()) {
                // Skip nodes inside code blocks
                if (!node.parentElement.closest('pre') && !node.parentElement.closest('code')) {
                    textNodes.push(node);
                }
            }
            
            searchMatches = [];
            const queryLower = query.toLowerCase();
            
            textNodes.forEach(textNode => {
                const text = textNode.textContent;
                const textLower = text.toLowerCase();
                let index = 0;
                
                while ((index = textLower.indexOf(queryLower, index)) !== -1) {
                    searchMatches.push({
                        node: textNode,
                        start: index,
                        length: query.length
                    });
                    index += query.length;
                }
            });
            
            // Highlight matches (reverse order to preserve indices)
            for (let i = searchMatches.length - 1; i >= 0; i--) {
                const match = searchMatches[i];
                const text = match.node.textContent;
                const before = text.substring(0, match.start);
                const matchText = text.substring(match.start, match.start + match.length);
                const after = text.substring(match.start + match.length);
                
                const span = document.createElement('span');
                span.className = 'search-highlight';
                span.textContent = matchText;
                span.setAttribute('data-match-index', i);
                
                const frag = document.createDocumentFragment();
                if (before) frag.appendChild(document.createTextNode(before));
                frag.appendChild(span);
                if (after) frag.appendChild(document.createTextNode(after));
                
                match.node.parentNode.replaceChild(frag, match.node);
            }
            
            currentMatchIndex = -1;
            updateSearchInfo();
            
            if (searchMatches.length > 0) {
                findNext();
            }
        }
        
        function findNext() {
            const highlights = document.querySelectorAll('.search-highlight');
            if (highlights.length === 0) {
                const query = document.getElementById('searchInput').value;
                if (query) findMatches(query);
                return;
            }
            
            // Remove current highlight
            highlights.forEach(h => h.classList.remove('current'));
            
            currentMatchIndex = (currentMatchIndex + 1) % highlights.length;
            const current = highlights[currentMatchIndex];
            current.classList.add('current');
            current.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            updateSearchInfo();
        }
        
        function findPrev() {
            const highlights = document.querySelectorAll('.search-highlight');
            if (highlights.length === 0) return;
            
            highlights.forEach(h => h.classList.remove('current'));
            
            currentMatchIndex = currentMatchIndex <= 0 ? highlights.length - 1 : currentMatchIndex - 1;
            const current = highlights[currentMatchIndex];
            current.classList.add('current');
            current.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            updateSearchInfo();
        }
        
        function updateSearchInfo() {
            const highlights = document.querySelectorAll('.search-highlight');
            if (highlights.length === 0) {
                document.getElementById('searchInfo').textContent = 'No matches';
            } else {
                document.getElementById('searchInfo').textContent = 
                    `${currentMatchIndex + 1} of ${highlights.length}`;
            }
        }
        
        // Debounce search
        let searchTimeout;
        document.getElementById('searchInput').addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                findMatches(e.target.value);
            }, 300);
        });
        
        // Ctrl+F shortcut
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
                e.preventDefault();
                document.getElementById('searchInput').focus();
            }
        });
    </script>
</body>
</html>
"""

# Write HTML file
with open('study_guide.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("Conversion complete! study_guide.html created with in-page search.")
