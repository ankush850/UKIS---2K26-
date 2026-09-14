import re

with open('frontend/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Fix double text
html = html.replace('<p class="panel-desc">Select a preset Sentinel-2 target scene or draw on the interactive map below.</p>\n                        <p class="panel-desc">Select a preset Sentinel-2 target scene or draw on the interactive map below.</p>', '<p class="panel-desc">Select a preset Sentinel-2 target scene or draw on the interactive map below.</p>')

# Fix unclosed header-actions
# Wait, let's just completely replace workspace-header-inject and hide it because the new UI has the logo in the top bar.
# But we need the button for blockchain! The original button was id="btn-open-blockchain"
# Let's extract the actual button from chunks.json if it exists, or just recreate it.
import json
with open('chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# The correct header actions was originally:
correct_header_actions = '''<div class="header-actions" style="display:none;">
    <div class="system-status">
        <span class="status-indicator active"></span>
        <span class="status-label">Copernicus CDSE Linked</span>
    </div>
    <button id="btn-open-blockchain" type="button" class="btn-blockchain-badge" title="Polygon Amoy Blockchain Provenance & Version Chain">
        <span class="chain-dot-purple"></span>
        <span class="chain-label">Polygon Amoy</span>
        <span class="chain-version-tag" id="header-chain-version">v1 Verified</span>
    </button>
</div>'''

html = re.sub(r'<div class="workspace-header-inject".*?</div>\s*</div>\s*</div>', f'<div class="workspace-header-inject" style="display:none;">{correct_header_actions}</div>', html, flags=re.DOTALL)

with open('frontend/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("Fixed HTML issues.")
