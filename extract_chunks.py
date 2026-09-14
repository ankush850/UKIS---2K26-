import re

with open('frontend/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Extract Landing Page
landing_match = re.search(r'(<!-- Landing Page -->.*?<!-- Dashboard Container -->)', html, re.DOTALL)
landing_html = landing_match.group(1)

# Extract Dashboard Container Header (we'll rebuild it, so we don't need to extract exactly, but we need the button)
header_actions = re.search(r'(<div class="header-actions">.*?</div>)', html, re.DOTALL).group(1)

# Extract AOI Panel body
aoi_match = re.search(r'<h3 class="panel-title">.*?Area of Interest \(AOI\).*?</h3>(.*?)<!-- Unified Aerospace', html, re.DOTALL)
aoi_presets = aoi_match.group(1).strip()
aoi_coords = re.search(r'(<!-- Unified Aerospace Coordinates Nav Bar -->.*?)</aside>', html, re.DOTALL).group(1)

# Extract Model Panel
model_match = re.search(r'(<div class="panel-card">\s*<h3 class="panel-title">.*?Model &amp; Inference Engine.*?</div>\s*</div>)', html, re.DOTALL)
model_panel = model_match.group(1) if model_match else ""

# Extract Visualization Toolbar
toolbar_match = re.search(r'(<div class="viewer-toolbar">.*?</div>\s*<!-- High Cloud Cover)', html, re.DOTALL)
toolbar_html = toolbar_match.group(1)

# Extract Canvas Viewport
canvas_match = re.search(r'(<!-- Comparison Canvas Container.*?</div>\s*</div>\s*</div>)', html, re.DOTALL)
canvas_html = canvas_match.group(1)

# Extract Heatmap Legend
footer_match = re.search(r'(<div class="viewer-footer">.*?</div>\s*</section>)', html, re.DOTALL)
viewer_footer_html = footer_match.group(1)

# Extract USP Card
usp_match = re.search(r'(<div class="panel-card usp-card">.*?</div>\s*<!-- Paired Reference)', html, re.DOTALL)
usp_html = usp_match.group(1)

# Extract Validation Paired
val_paired_match = re.search(r'(<div class="panel-card" id="panel-validation-paired">.*?</div>\s*</div>)\s*<!-- No-Reference', html, re.DOTALL)
val_paired_html = val_paired_match.group(1)

# Extract Validation NR
val_nr_match = re.search(r'(<div class="panel-card panel-card-nr" id="panel-validation-nr">.*?</div>)\s*<!-- Detected', html, re.DOTALL)
val_nr_html = val_nr_match.group(1)

# Extract Infrastructure
infra_match = re.search(r'(<div class="panel-card panel-card-amber" id="panel-infrastructure">.*?</div>)\s*</aside>', html, re.DOTALL)
infra_html = infra_match.group(1)

# Extract Footer
pipeline_match = re.search(r'(<footer class="site-footer" id="site-footer".*?</footer>)', html, re.DOTALL)
pipeline_html = pipeline_match.group(1)

# Extract Modals & Scripts
modals_scripts_match = re.search(r'(<!-- Blockchain Provenance & Version Chain Modal -->.*)', html, re.DOTALL)
modals_scripts = modals_scripts_match.group(1)

# Write extracted chunks to text files so we can inspect or use them safely
import json
chunks = {
    "landing": landing_html,
    "header_actions": header_actions,
    "aoi_presets": aoi_presets,
    "aoi_coords": aoi_coords,
    "model_panel": model_panel,
    "toolbar": toolbar_html,
    "canvas": canvas_html,
    "viewer_footer": viewer_footer_html,
    "usp": usp_html,
    "val_paired": val_paired_html,
    "val_nr": val_nr_html,
    "infra": infra_html,
    "pipeline": pipeline_html,
    "modals": modals_scripts
}

with open('chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2)

print("Chunks extracted successfully!")
