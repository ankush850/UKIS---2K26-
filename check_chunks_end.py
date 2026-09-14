import json

with open('chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Fix aoi_coords: extra </div>
chunks['aoi_coords'] = chunks['aoi_coords'].rsplit('</div>', 1)[0] + chunks['aoi_coords'].rsplit('</div>', 1)[1]
# Actually, the easiest way to fix extra/missing divs is just manual regex or replace.
# Let's see what aoi_coords ends with.
print('--- aoi_coords END ---')
print(chunks['aoi_coords'][-100:])

print('--- model_panel END ---')
print(chunks['model_panel'][-100:])

print('--- canvas END ---')
print(chunks['canvas'][-100:])

print('--- viewer_footer END ---')
print(chunks['viewer_footer'][-100:])

print('--- modals END ---')
print(chunks['modals'][-100:])

