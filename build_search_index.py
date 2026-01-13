"""
Build Search Index - COMPLETE VERSION
Indexes EVERYTHING: Full study guide + All exam HTML content
Run: python build_search_index.py
"""

import json
import re
import os
from html.parser import HTMLParser

class TextExtractor(HTMLParser):
    """Extract ALL text from HTML"""
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False
        
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'head'):
            self.skip = True
            
    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'head'):
            self.skip = False
            
    def handle_data(self, data):
        if not self.skip:
            text = data.strip()
            if text:
                self.text.append(text)
                
    def get_text(self):
        return ' '.join(self.text)

def extract_html_text(html):
    """Extract text from HTML - with fallback"""
    try:
        parser = TextExtractor()
        parser.feed(html)
        return parser.get_text()
    except:
        # Fallback: regex
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.I)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.I)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

def build_index():
    items = []
    
    # ========== 1. FULL STUDY GUIDE ==========
    print("📚 Indexing study guide...")
    if os.path.exists('study_guide.md'):
        with open('study_guide.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into smaller chunks with MORE overlap
        chunk_size = 8000
        overlap = 2000
        i = 0
        chunk_num = 0
        while i < len(content):
            chunk = content[i:i + chunk_size]
            items.append({
                'id': f'study-guide-{chunk_num}',
                'title': f'Study Guide (Part {chunk_num + 1})',
                'content': chunk,
                'type': 'study-guide',
                'source': 'study_guide.html'
            })
            i += (chunk_size - overlap)
            chunk_num += 1
        
        print(f"   ✓ {len(content):,} chars → {chunk_num} chunks")
    
    # ========== 2. ALL EXAM HTML FILES ==========
    print("📝 Indexing exam files...")
    exam_dir = 'past_exams'
    if os.path.exists(exam_dir):
        for fname in os.listdir(exam_dir):
            if fname.endswith('.html'):
                fpath = os.path.join(exam_dir, fname)
                print(f"   Processing {fname}...", end=' ', flush=True)
                
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    html = f.read()
                
                # Extract ALL text
                text = extract_html_text(html)
                
                # Split into chunks
                chunk_size = 6000
                overlap = 1500
                i = 0
                chunk_num = 0
                exam_title = fname.replace('.html', '').replace('_', ' ')
                
                while i < len(text):
                    chunk = text[i:i + chunk_size]
                    items.append({
                        'id': f'exam-{fname}-{chunk_num}',
                        'title': f'{exam_title}' + (f' (Part {chunk_num + 1})' if chunk_num > 0 else ''),
                        'content': chunk,
                        'type': 'exam',
                        'source': f'past_exams/{fname}'
                    })
                    i += (chunk_size - overlap)
                    chunk_num += 1
                
                print(f"{len(text):,} chars → {chunk_num} chunks")
    
    # ========== 3. JSON DATA ==========
    print("📊 Indexing JSON data...")
    
    for jfile, key, typ in [
        ('data/slides.json', 'slides', 'slide'),
        ('data/notes.json', 'notes', 'note'),
        ('data/exercises.json', 'exercises', 'exercise')
    ]:
        if os.path.exists(jfile):
            with open(jfile, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data.get(key, []):
                content = ' '.join(filter(None, [
                    item.get('title', ''),
                    item.get('description', ''),
                    item.get('content', ''),
                    item.get('question', ''),
                    item.get('answer', ''),
                    ' '.join(item.get('keywords', []))
                ]))
                items.append({
                    'id': item.get('id', ''),
                    'title': item.get('title', ''),
                    'content': content,
                    'type': typ,
                    'source': item.get('slidePath', typ + 's')
                })
            print(f"   ✓ {len(data.get(key, []))} {key}")
    
    # ========== SAVE ==========
    os.makedirs('data', exist_ok=True)
    with open('data/search_index.json', 'w', encoding='utf-8') as f:
        json.dump({'items': items}, f, ensure_ascii=False, indent=None)
    
    total_chars = sum(len(i.get('content', '')) for i in items)
    print(f"\n✅ DONE: {len(items)} items, {total_chars:,} total characters")
    print(f"   Saved to data/search_index.json")

if __name__ == '__main__':
    build_index()
