# Problem Statement & Solution Documentation: NETRA-D
**Project ID:** UKIS-2026 (Problem P-008)  
**Problem Owner:** Disaster Mitigation and Management Centre (DMMC), Department of Disaster Management, Government of Uttarakhand, Dehradun  
**System Name:** NETRA-D (Neural Enhancement for Terrain & Resolution Amplification + Tactical Drone Assessment)  
**Theme:** AI-Powered Multi-Tier Disaster Assessment & Infrastructure Monitoring  

---

## 1. Executive Summary

Disaster mitigation and infrastructure monitoring in rugged Himalayan theatres (Uttarakhand: Kedarnath, Chamoli, Alaknanda Valley, Badrinath Corridor) require continuous situational awareness. However, disaster response agencies face a severe **optical data paradox**:
- **Public satellite imagery (Copernicus Sentinel-2)** is free with high 5-day revisit rates, but its **10-meter Ground Sample Distance (GSD)** is too coarse to identify blocked evacuation corridors, washed-out culverts, or individual collapsed buildings.
- **Commercial high-resolution satellite imagery (WorldView, SPOT 6/7)** costs thousands of dollars per scene, requires lengthy tasking lead-times, and is frequently blinded by Himalayan monsoon cloud cover.
- **Tactical UAVs (Drones)** deliver sub-decimeter ground resolution (5cm–10cm) under cloud cover, but cannot survey thousands of square kilometers simultaneously and suffer severe domain-generalization failures when models encounter unseen terrain angles or off-domain features.

**NETRA-D** solves this dilemma through a synchronized **Dual-Tier AI Architecture**:
1. **Macro Tier (Netra Satellite):** Wide-area triage upscaling 10m Sentinel-2 tiles to **2.5m GSD ($16\times$ pixel density gain)** via a WorldStrat-trained Hybrid Attention Transformer (HAT) paired with an active **Hallucination-Aware Uncertainty Layer** and **Polygon Amoy Blockchain Provenance**.
2. **Micro Tier (Netra Aerial Tactical):** Localized tactical response using a **Tri-Model Hazard Segmentation Suite** (FloodNet + TransLandSeg + SegFormer ADE20K), **Microsoft SiamUnet** building damage assessment (xBD standard), and real-time **OpenStreetMap Overpass** road passability analysis.

---

## 2. The Core Problem Statement Analysis (UKIS-2026 Problem P-008)

### Challenge 1: The Resolution vs Cost Bottleneck in Mountainous Remote Sensing
Sentinel-2 Bottom-Of-Atmosphere (L2A) imagery is open-access and radiometrically calibrated, but its 10m pixel pitch ($100\text{ m}^2$ per pixel) cannot resolve:
- Rural mountain tracks and unpaved evacuation roads ($< 6\text{m}$ width).
- Bridges, culverts, and river crossings across the Alaknanda and Mandakini basins.
- Smallholder agricultural terraces and village settlement boundaries.

### Challenge 2: The "AI Hallucination" Hazard in Emergency Operations
Standard generative upscalers (such as commercial GANs or unconstrained diffusion models) invent sharp textures to satisfy human visual perception. In disaster response, an AI that hallucinates an intact bridge or fabricates a passable road can misdirect rescue convoys into danger.

### Challenge 3: Real-World Domain Generalization Failures in Drone Models
When tactical drones capture aerial imagery in complex disaster zones, off-the-shelf single-model deep learning pipelines fail catastrophically:
- **Missing Landslide Class:** Popular flood models (e.g. FloodNet) have no bare-soil/landslide class. In massive landslides (e.g. Wayanad, Chamoli), they detect near-zero hazard.
- **Oblique Horizon Sky Bias:** Nadir-trained drone models (trained looking straight down at 90°) have never seen the sky. When presented with oblique aerial photos, they misclassify blue sky as **66.66% floodwater**!

---

## 3. The Proposed Solution: NETRA-D

NETRA-D introduces a synchronized, scientifically grounded multi-tier platform:

```
+--------------------------------------------------------------------------------------------------+
|                                    NETRA-D DUAL-TIER ARCHITECTURE                                |
+--------------------------------------------------------------------------------------------------+
|  MACRO TIER (Netra Satellite)                         MICRO TIER (Netra Aerial Drone)            |
|  - Copernicus CDSE Sentinel-2 L2A Ingestion           - Tri-Model Hazard Segmentation:           |
|  - Deep Learning Super-Resolution (10m -> 2.5m)         * FloodNet DeepLabV3+ (Nadir Floods)     |
|    via Hybrid Attention Transformer (HAT)               * TransLandSeg SAM ViT-L (Landslides)    |
|  - Hallucination-Aware Uncertainty Layer (USP):         * SegFormer ADE20K (Water vs Sky)        |
|    Monte-Carlo Dropout (σ²) + Cycle Consistency       - Microsoft SiamUnet Pre/Post Damage (xBD) |
|  - ESA SCL Multi-Spectral Cloud Shield                - Real Overpass OSM Road Passability       |
|  - Multi-Spectral Indices: NDVI, NDWI, NBR, NDBI      - 60-Frame Live HUD Sortie Simulation      |
|  - Polygon Amoy Blockchain Immutable Provenance       - Automated DMMC Incident Briefing (PDF)   |
+--------------------------------------------------------------------------------------------------+
```

### Macro Tier Innovations:
1. **Certified HAT Super-Resolution:** 10m Sentinel-2 bands are enhanced to 2.5m GSD using Hybrid Attention Transformers trained on paired Sentinel-2/SPOT 6/7 scenes, recovering rural roads and building footprints.
2. **Hallucination-Aware Uncertainty Layer (The USP):** Runs $N=8$ stochastic Monte-Carlo Dropout passes to compute epistemic variance $\boldsymbol{\sigma}^2(x, y)$, coupled with ESA `opensr-test` cycle consistency and SAM spectral angle checks, generating an interactive **Confidence Heatmap (0% to 100%)**.
3. **Multi-Spectral Cloud Shield:** Automatically detects cirrus and dense cloud decks, clamping confidence strictly to **0.0%** to prevent false hazard alerts.
4. **NETRA Blockchain Provenance:** Deployed on Polygon Amoy (`0x2287c88b7764A9D386FeE490958e0aF1316b8F10`), anchoring rasters with SHA-256 hashes and sub-11cm integer-encoded coordinates for legal and forensic admissibility.

### Micro Tier Innovations:
1. **Tri-Model Hazard Segmentation Suite:**
   - **FloodNet DeepLabV3+:** High-precision nadir structural inundation and floodwater segmentation.
   - **TransLandSeg (SAM ViT-L · Bijie Dataset):** Dedicated 304M-parameter Vision Transformer trained on the Bijie Landslide Dataset, detecting active landslide scars and mudflow deposits with 93%+ confidence.
   - **SegFormer B0 (ADE20K 150-Class Transformer):** Disambiguates water, sea, river, and lake from sky horizons with per-pixel softmax confidence estimation, eliminating oblique aerial false floods.
2. **Intelligent Hazard Routing & Discrepancy Detection:** Flags model discrepancies (`models_disagree`) when oblique sky horizons trigger false positives in FloodNet, automatically suppressing the false flag and routing to the correct disaster mode.
3. **Microsoft SiamUnet Damage Inspector:** Evaluates pre- and post-disaster paired imagery into Destroyed, Major Damage, Minor Damage, and Intact categories (strictly normalized to $100.0\%$).
4. **Live OSM Overpass Road Passability:** Queries real highway vectors (NH-7, Badrinath, Kedarnath) and computes polygon intersections with detected hazards to flag impassable road corridors.
5. **60-Frame Simulated Live HUD Sortie:** Runs a Ken Burns UAV sortie over survey images with real-time per-frame PyTorch inference and flight telemetry HUD overlays.

---

## 4. Operational Impact & Beneficiaries

*   **Disaster Mitigation and Management Centre (DMMC), Uttarakhand:** Rapid, trustworthy triage of flash floods, glacial lake outburst floods (GLOF), and cloudbursts across the Garhwal and Kumaon Himalayas.
*   **State & National Disaster Response Forces (SDRF / NDRF):** Clear demarcation of impassable evacuation routes, blocked bridges, and high-priority rescue zones with automated incident briefing reports.
*   **District Magistrates (Rudraprayag, Chamoli, Uttarkashi):** Legally tamper-proof satellite intelligence anchored on the Polygon blockchain for disaster relief fund distribution and damage audits.
*   **DoNER / NESAC / ISRO:** Scalable, open-access remote sensing pipeline converting free Sentinel-2 data into commercial-grade intelligence at near-zero incremental cost.
