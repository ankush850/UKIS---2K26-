# -*- coding: utf-8 -*-
import json

with open('chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

new_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIH26142 - Super Resolution Mapping (SRM) | Sentinel-2 (10m -> 2.5m)</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&family=Montserrat:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <!-- Leaflet & Leaflet.draw CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css">
    <!-- Existing Styles -->
    <link rel="stylesheet" href="/static/styles.css?v=2.0">
    <!-- New Layout Styles (Overrides) -->
    <link rel="stylesheet" href="/static/new_layout.css?v=1.0">
</head>
<body>
{chunks["landing"]}

<div id="dashboard-container" style="display: none; opacity: 0; transition: opacity 0.5s ease;" class="app-shell">
    <!-- Top Navbar -->
    <nav class="app-top-nav">
        <div class="nav-brand">
            <div class="logo-badge">
                <span class="pulse-orb"></span>
                <span class="brand-tag">NETRA</span>
            </div>
            <div class="breadcrumbs">
                <span class="bc-fade">GEO-SRM</span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
                <span class="bc-active">Sentinel-2</span>
            </div>
        </div>
        <div class="nav-center-links">
            <a href="#" class="nav-link active">Analyze</a>
            <a href="#" class="nav-link">Datasets</a>
            <a href="#" class="nav-link">Models</a>
            <a href="#" class="nav-link">Results</a>
            <a href="#" class="nav-link">Docs</a>
        </div>
        <div class="nav-actions">
            <div class="global-search">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                <input type="text" placeholder="Search locations, datasets...">
            </div>
            <button class="icon-btn theme-toggle"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg></button>
            <div class="user-profile">
                <div class="avatar">K</div>
                <span class="user-name">Karan</span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
            </div>
        </div>
    </nav>

    <div class="app-body">
        <!-- Left Sidebar -->
        <aside class="app-sidebar">
            <nav class="side-nav">
                <a href="#" class="side-item active">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
                    Analysis
                </a>
                <a href="#" class="side-item">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>
                    AOI Selection
                </a>
                <a href="#" class="side-item">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                    Model & Inference
                </a>
                <a href="#" class="side-item">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    Results
                </a>
                <a href="#" class="side-item">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                    Validation
                </a>
                <a href="#" class="side-item">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                    Export
                </a>
                <a href="#" class="side-item">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 12 12 17 22 12"></polyline><polyline points="2 17 12 22 22 17"></polyline></svg>
                    Pipeline
                </a>
                <a href="#" class="side-item">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                    Settings
                </a>
            </nav>
            <div class="sidebar-promo-card">
                <img src="/static/earth.jpg" alt="Earth" style="width:100%; border-radius:10px; margin-bottom:10px; opacity: 0.8;">
                <div class="promo-content">
                    <span class="promo-sub">From</span>
                    <span class="promo-main">Satellite Data<br>to Real Impact</span>
                </div>
            </div>
            <div class="sidebar-footer">
                <div class="brand-small"><span class="pulse-orb"></span> NETRA</div>
            </div>
        </aside>

        <!-- Main Workspace -->
        <main class="app-main">
            <!-- Legacy Header Text injection so functionality stays (can be hidden or styled to match the new top bar) -->
            <div class="workspace-header-inject" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <div class="brand-text">
                    <h1 style="font-size: 1.4rem; margin:0; font-family: 'Montserrat', sans-serif;">GEO-SRM <span class="cyan-text">Sentinel-2</span></h1>
                    <p class="subtitle" style="margin:0; font-size: 0.85rem; color: #94a3b8;">Deep Learning Super-Resolution (10m -> 2.5m) with Hallucination-Aware Uncertainty</p>
                </div>
                <div class="header-actions">
                    {chunks["header_actions"]}
                </div>
            </div>

            <!-- CSS Grid Layout -->
            <div class="main-workspace-grid">
                <!-- Col 1: AOI -->
                <div class="col-left">
                    <div class="panel-card-new panel-aoi">
                        <div class="panel-header-new">
                            <div class="step-badge">1</div>
                            <h3>Area of Interest (AOI)</h3>
                        </div>
                        <p class="panel-desc">Select a preset Sentinel-2 target scene or draw on the interactive map below.</p>
                        {chunks["aoi_presets"]}
                        {chunks["aoi_coords"]}
                    </div>
                </div>

                <!-- Col 2: Viewer -->
                <div class="col-center">
                    <div class="panel-card-new panel-visualization">
                        <div class="panel-header-new">
                            <div class="step-badge" style="background:#3b82f6;">2</div>
                            <h3>Visualization & Layers</h3>
                        </div>
                        {chunks["toolbar"]}
                        {chunks["canvas"]}
                        {chunks["viewer_footer"]}
                    </div>
                </div>

                <!-- Col 3: HUDs -->
                <div class="col-right">
                    {chunks["usp"].replace('<h3 class="panel-title">Hallucination &amp; Uncertainty</h3>', '<div class="panel-header-new"><div class="step-badge" style="background:#a855f7;">3</div><h3>Hallucination &amp; Uncertainty</h3></div>').replace('<span class="card-badge">THE USP</span>', '<span class="card-badge" style="background: rgba(168, 85, 247, 0.15); color: #d8b4fe; border-color: rgba(168, 85, 247, 0.3);">THE USP</span>')}
                    
                    {chunks["val_paired"].replace('<h3 class="panel-title">Validation vs HR Reference</h3>', '<div class="panel-header-new"><div class="step-badge" style="background:#38bdf8;">4</div><h3>Validation vs HR Reference</h3></div>')}
                    
                    {chunks["val_nr"].replace('<h3 class="panel-title panel-title-purple">No-Reference Quality Assessment</h3>', '<div class="panel-header-new"><div class="step-badge" style="background:#38bdf8;">4</div><h3>No-Reference Quality Assessment</h3></div>')}
                    
                    {chunks["infra"].replace('<h3 class="panel-title panel-title-amber">Detected Infrastructure</h3>', '<div class="panel-header-new"><div class="step-badge" style="background:#f59e0b;">*</div><h3>Detected Infrastructure</h3></div>')}
                </div>

                <!-- Bottom Col: Pipeline -->
                <div class="col-bottom">
                    {chunks["pipeline"].replace('<h3 class="footer-title">Pipeline Verification</h3>', '<div class="panel-header-new"><div class="step-badge" style="background:#06b6d4;">5</div><h3>Pipeline Verification</h3></div>').replace('<span class="footer-subtitle">Live inference telemetry, Monte-Carlo uncertainty, and paired satellite provenance</span>', '<span class="footer-subtitle" style="margin-left: 10px; color:#94a3b8; font-size: 0.85rem;">Live inference telemetry, Monte-Carlo uncertainty, and paired satellite provenance</span>')}
                </div>
            </div>
            
            <!-- Hidden Model Panel to keep logic intact without taking up space in the main grid -->
            <div style="display:none;">
                {chunks["model_panel"]}
            </div>
        </main>
    </div>
</div>

{chunks["modals"]}
</body>
</html>
'''

with open('frontend/index.html', 'w', encoding='utf-8') as f:
    f.write(new_html)

print("index.html rewritten successfully!")
