"""
Build Search Index - Optimized Version
Generates comprehensive search index for all website content.
"""

import json
import re
import os

def extract_text_from_html_simple(html_content):
    """Simple fast text extraction from HTML"""
    # Remove script and style blocks
    text = re.sub(r'<script[^>]*>[\s\S]*?</script>', ' ', html_content, flags=re.IGNORECASE)
    text = re.sub(r'<style[^>]*>[\s\S]*?</style>', ' ', text, flags=re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_index():
    search_items = []
    
    # 1. STUDY GUIDE - FULL CONTENT
    print("Indexing study guide...")
    if os.path.exists('study_guide.md'):
        with open('study_guide.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add full content as chunks (10KB each)
        chunk_size = 10000
        for i in range(0, len(content), chunk_size - 1000):  # 1000 overlap
            chunk = content[i:i + chunk_size]
            search_items.append({
                'id': f'study-guide-chunk-{i//chunk_size}',
                'title': f'Study Guide (Part {i//chunk_size + 1})',
                'content': chunk,
                'type': 'study-guide',
                'source': 'study_guide.html'
            })
        print(f"  -> {len(content):,} chars in {len(content)//chunk_size + 1} chunks")
    
    # 2. PAST EXAMS
    print("Indexing past exams...")
    exam_dir = 'past_exams'
    if os.path.exists(exam_dir):
        for filename in os.listdir(exam_dir):
            if filename.endswith('.html'):
                filepath = os.path.join(exam_dir, filename)
                print(f"  Processing {filename}...", end=' ')
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    html = f.read()
                
                text = extract_text_from_html_simple(html)
                
                # Add as chunks
                chunk_size = 8000
                chunks_added = 0
                for i in range(0, len(text), chunk_size - 500):
                    chunk = text[i:i + chunk_size]
                    exam_title = filename.replace('.html', '').replace('_', ' ')
                    search_items.append({
                        'id': f'exam-{filename}-{i//chunk_size}',
                        'title': f'{exam_title}' + (f' (Part {i//chunk_size + 1})' if len(text) > chunk_size else ''),
                        'content': chunk,
                        'type': 'exam',
                        'source': f'past_exams/{filename}'
                    })
                    chunks_added += 1
                print(f"{len(text):,} chars, {chunks_added} chunks")
    
    # 3. JSON DATA FILES
    print("Indexing JSON data...")
    
    for json_file, key, item_type in [
        ('data/slides.json', 'slides', 'slide'),
        ('data/notes.json', 'notes', 'note'),
        ('data/exercises.json', 'exercises', 'exercise')
    ]:
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data.get(key, []):
                content_parts = [
                    item.get('title', ''),
                    item.get('description', ''),
                    item.get('content', ''),
                    item.get('question', ''),
                    item.get('answer', ''),
                    ' '.join(item.get('keywords', []))
                ]
                search_items.append({
                    'id': item.get('id', ''),
                    'title': item.get('title', ''),
                    'content': ' '.join(filter(None, content_parts)),
                    'type': item_type,
                    'source': item.get('slidePath', item_type + 's')
                })
            print(f"  -> {len(data.get(key, []))} {key}")
    
    # 4. SAVE
    os.makedirs('data', exist_ok=True)
    with open('data/search_index.json', 'w', encoding='utf-8') as f:
        json.dump({'items': search_items}, f, ensure_ascii=False)
    
    total_chars = sum(len(item.get('content', '')) for item in search_items)
    print(f"\n✓ Built index: {len(search_items)} items, {total_chars:,} total characters")
    print("✓ Saved to data/search_index.json")

if __name__ == '__main__':
    build_index()
