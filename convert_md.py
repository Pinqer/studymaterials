import re
import json

# Read markdown file
with open('study_guide.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Properly escape for JSON
content_json = json.dumps(content)

# HTML template with fixed syntax highlighting
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
            padding: 20px 40px;
            background: #fff;
            color: #333;
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
        /* Override highlight.js defaults for better visibility */
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
    <div id="content"></div>
    <script>
        const markdownContent = """ + content_json + """;
        
        // Configure marked WITHOUT custom highlight function
        // We'll use hljs.highlightAll() instead
        marked.setOptions({
            breaks: true,
            gfm: true
        });
        
        // Render markdown
        document.getElementById('content').innerHTML = marked.parse(markdownContent);
        
        // Apply syntax highlighting to all code blocks AFTER rendering
        hljs.highlightAll();
        
        // Trigger MathJax after rendering
        if (window.MathJax) {
            MathJax.typesetPromise().then(() => {
                console.log('MathJax rendering complete');
            }).catch((err) => console.log('MathJax error:', err));
        }
    </script>
</body>
</html>
"""

# Write HTML file
with open('study_guide.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("Conversion complete! study_guide.html created with proper syntax highlighting.")
