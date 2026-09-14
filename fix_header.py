import json

with open('chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Fix header_actions missing div
open_count = chunks['header_actions'].count('<div')
close_count = chunks['header_actions'].count('</div')
if open_count > close_count:
    chunks['header_actions'] += '\n</div>' * (open_count - close_count)

with open('chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2)

print("Fixed header_actions!")
