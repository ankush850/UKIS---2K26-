import json

with open('chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

for name, chunk in chunks.items():
    div_open = chunk.count('<div')
    div_close = chunk.count('</div')
    section_open = chunk.count('<section')
    section_close = chunk.count('</section')
    aside_open = chunk.count('<aside')
    aside_close = chunk.count('</aside')
    main_open = chunk.count('<main')
    main_close = chunk.count('</main')
    
    print(f"{name}: divs({div_open}-{div_close}), sections({section_open}-{section_close}), asides({aside_open}-{aside_close}), mains({main_open}-{main_close})")

