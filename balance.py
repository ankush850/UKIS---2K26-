import json

with open('chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

def balance_divs(html):
    open_count = html.count('<div')
    close_count = html.count('</div')
    diff = open_count - close_count
    if diff > 0:
        html += '\n</div>' * diff
    elif diff < 0:
        # remove extra </div> from the end
        for _ in range(-diff):
            idx = html.rfind('</div>')
            if idx != -1:
                html = html[:idx] + html[idx+6:]
    return html

chunks['aoi_coords'] = balance_divs(chunks['aoi_coords'])
chunks['model_panel'] = balance_divs(chunks['model_panel'])
chunks['canvas'] = balance_divs(chunks['canvas'])
chunks['viewer_footer'] = chunks['viewer_footer'].replace('</section>', '')
chunks['modals'] = chunks['modals'].replace('</div>\n</body>', '</body>').replace('</div>\r\n</body>', '</body>')

with open('chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2)

print("Balanced chunks!")
