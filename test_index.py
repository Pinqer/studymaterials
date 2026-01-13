import json

with open('data/search_index.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Check exam items
exam_items = [i for i in data['items'] if i['type'] == 'exam']
print(f'Exam items: {len(exam_items)}')
for item in exam_items[:3]:
    print(f"  - {item['title']}: {len(item['content'])} chars")

# Search all items for Aracaju
found = False
for item in data['items']:
    if 'Aracaju' in item.get('content', ''):
        print(f"Found 'Aracaju' in: {item['title']}")
        found = True
        break

if not found:
    print("'Aracaju' NOT found in any item")
    
# Search for "current city"
for item in data['items']:
    if 'current city' in item.get('content', ''):
        print(f"Found 'current city' in: {item['title']}")
        break
