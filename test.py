import re

with open('frontend/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# We will just write a whole new frontend/index.html manually because it's too complex to regex perfectly.
# Wait, I should not overwrite everything if I can just provide the new CSS and HTML.
