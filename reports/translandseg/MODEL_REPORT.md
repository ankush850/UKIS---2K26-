# Model Technical Dossier: TransLandSeg (SAM ViT-L · Bijie Dataset)
## Dedicated Mountain Landslide Scar & Debris Flow Segmentation Engine

**Project:** NETRA-D (UKIS-2026 Problem P-008)  
**Beneficiary:** Disaster Mitigation and Management Centre (DMMC), Department of Disaster Management, Government of Uttarakhand  
**Category:** Tactical Aerial Drone Intelligence (Micro Tier)  
**Model Name:** TransLandSeg  
**Base Architecture:** Meta Segment Anything Model (SAM) Vision Transformer Large (ViT-L)  
**Checkpoint Path:** [`checkpoints/Bijie.pth.tar`](../../checkpoints/Bijie.pth.tar)  
**Target Terrain:** Steep Himalayan Mountain Valleys, Debris Slopes, Road Corridors  

---

## 1. Executive Summary & Problem Context

In mountainous disaster theatres (such as the Uttarakhand Himalayas, Wayanad, and Chamoli), slope instability triggered by monsoon cloudbursts or seismic activity produces massive **bare-soil landslide scars, rockfalls, and mudflow debris tongues**.

Standard drone disaster segmentation models (such as FloodNet) were trained exclusively on flood-domain imagery. Consequently, they possess **zero class representation for bare-soil landslides or debris flows**. When presented with catastrophic landslides, standard models produce near-zero detection.

**TransLandSeg** bridges this root-cause void by deploying a dedicated 304M-parameter Vision Transformer Large (ViT-L) backbone fine-tuned on the **Bijie Landslide Dataset** (7,748 optical scenes of mountain slope failures). It isolates active landslide scars and mudflow deposits with **93%+ confidence**, seamlessly exporting GIS polygons for road blockage analysis and emergency rescue directives.

---

## 2. Technical Architecture & Mathematical Foundation

TransLandSeg slices input aerial imagery into $16 \times 16$ non-overlapping patches, applying learnable 2D positional embeddings before passing them through 24 transformer attention blocks:

$$\mathbf{z}_0 = [\mathbf{x}_p^1 \mathbf{E}; \dots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

Features are upsampled via a deconvolutional mask decoder to generate a continuous probability map $\hat{\mathbf{Y}} \in [0, 1]^{H \times W}$. Optimization employs a composite Dice + Binary Cross-Entropy loss to overcome severe background-to-scar class imbalance:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}}(\hat{\mathbf{Y}}, \mathbf{Y}) + 0.5 \cdot \mathcal{L}_{\text{Dice}}(\hat{\mathbf{Y}}, \mathbf{Y})$$

---

## 3. Quantitative Verification Benchmarks

| Test Scenario | Visual Characteristics | FloodNet Standalone | TransLandSeg Standalone | NETRA-D Consensus Verdict |
| :--- | :--- | :---: | :---: | :--- |
| **Meppadi Landslide (Wayanad, Kerala)** | Catastrophic mountain slope failure; massive red-brown mudflow scar cutting through tea estates. | ⚠️ 2.84% Water, 1.34% Debris *(Missed 500m scar)* | 🔴 **28.74% Active Landslide Scar (93.1% Conf)** | ✅ **Emergency Landslide Triggered:** Severity 88/100, Immediate Evacuation Directive. |
| **Mountain Slope Scree Control** | Steep rocky incline with dry scree and dry gravel; no active disaster. | 1.84% Water | **0.40% Scar** *(Below 4.0% threshold)* | ✅ **Stable Mountain Slope:** No false alarm. |
| **Sydney Beach Houses Control** | Coastal town with blue sky and ocean; zero disaster. | ⚠️ 66.66% Floodwater *(False Positive)* | **0.00% Scar** | ✅ **Stable Baseline:** Zero false landslide trigger. |

---

## 4. Operational Integration in NETRA-D

1. **Intelligent Hazard Consensus:** If TransLandSeg detects landslide scar $\ge 4.0\%$ and dominant over flood indicators, the pipeline automatically routes the sortie into **Landslide Disaster Mode**.
2. **Overpass OSM Road Blockage Intersection:** The segmented landslide debris mask is vector-buffered against real OpenStreetMap highway geometries (e.g. NH-7 Badrinath corridor) to pinpoint blocked road segments.
3. **Automated DMMC Incident Briefing:** Incorporates landslide footprint area (sq. meters), evacuation recommendations, and cryptographic verification hashes into the official emergency briefing PDF/HTML.

---

## 5. Navigation & File Links

- 📖 **Overview & Operational Use Cases:** [`Overview.md`](Overview.md)
- ⚙️ **Mathematical Formulas & Architecture Diagram:** [`Working.md`](Working.md)
- 📊 **Empirical Evaluation & Decision Guide:** [`Evaluation.md`](Evaluation.md)
- 💻 **Standalone Runnable Implementation:** [`Code.py`](Code.py)
