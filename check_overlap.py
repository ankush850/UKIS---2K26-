import json
with open('chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print('aoi_coords length:', len(chunks['aoi_coords']))
print('model_panel length:', len(chunks['model_panel']))
print('Is model_panel inside aoi_coords?', chunks['model_panel'] in chunks['aoi_coords'])
