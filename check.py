import json
with open('chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(chunks['aoi_coords'][-100:])
