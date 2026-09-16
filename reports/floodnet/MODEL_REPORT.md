# Model Technical Dossier: FloodNet DeepLabV3+
## Tactical Nadir Flood Inundation & Structural Submergence Engine

**Project:** NETRA-D (UKIS-2026 Problem P-008)  
**Beneficiary:** Disaster Mitigation and Management Centre (DMMC), Department of Disaster Management, Government of Uttarakhand  
**Category:** Tactical Aerial Drone Intelligence (Micro Tier)  
**Model Name:** FloodNet DeepLabV3+  
**Architecture:** DeepLabV3+ with ResNet Backbone and Atrous Spatial Pyramid Pooling (ASPP)  
**Checkpoint Path:** [`backend/aerial/checkpoints/floodnet_deeplabv3plus.pth`](../../backend/aerial/)  
**Target Perspectives:** 90° Nadir Orthomosaics, Top-Down UAV Grid Sorties  

---

## 1. Executive Summary & Problem Context

In rapid-onset flash flood disasters (such as cloudbursts in the Mandakini valley or Chamoli river surges), emergency responders need to rapidly isolate flooded road networks and submerged residential clusters.

Satellite observations (10m Sentinel-2) lack the resolving power to verify whether a village culvert is flooded or passable. **FloodNet DeepLabV3+** processes sub-decimeter tactical drone orthomosaics (5cm–10cm GSD) and segments terrain into 4 tactical disaster classes:
1. `Background / Non-Flooded Terrain`
2. `Flooded Building`
3. `Flooded Road`
4. `Water / Submergence`

All class percentages are strictly normalized to **$100.0\%$**, eliminating arithmetic distortion in official DMMC incident briefings.

---

## 2. Technical Architecture & Mathematical Foundation

FloodNet employs Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale inundation features:

$$y[i] = \sum_{k=1}^K x[i + r \cdot k] \cdot w[k], \quad r \in \{6, 12, 18\}$$

Encoder bottleneck features are fused with low-level stem features to recover fine boundary detail around building eaves and road curbs:

$$\mathbf{F}_{\text{fused}} = \text{Concat}\left( \text{Upsample}_{4\times}(\mathbf{Y}_{\text{ASPP}}), \, \text{Conv}_{1 \times 1}(\mathbf{F}_{\text{stem}}) \right)$$

---

## 3. Operational Domain Boundaries & Multi-Model Auditing

1. **The Oblique Sky Boundary:** FloodNet was trained strictly on top-down nadir drone data. When evaluating oblique perspectives with horizon sky, it is audited by **SegFormer B0**, which identifies sky at 99.5% confidence and suppresses false flood positives.
2. **The Landslide Scar Boundary:** FloodNet has zero class representation for bare-soil mountain landslides. When evaluating mountain slope failures, it is paired with **TransLandSeg (SAM ViT-L)**, which detects active landslide scars.

---

## 4. Quantitative Verification Summary

| Metric | Measured Value | Operational Meaning |
| :--- | :---: | :--- |
| **Mean IoU (Nadir Test Set)** | **73.2%** | High-precision nadir segmentation across all 4 classes |
| **Flooded Road Boundary Recall** | **84.5%** | Accurately identifies submerged road corridors |
| **Arithmetic Integrity** | **$\sum P_c \equiv 100.0\%$** | Zero arithmetic error in command reports |
| **CPU Execution Speed (512x512)** | **~650 – 850 ms** | Tactical turnaround on field laptops |

---

## 5. Navigation & File Links

- 📖 **Overview & Operational Use Cases:** [`Overview.md`](Overview.md)
- ⚙️ **Mathematical Formulas & Architecture Diagram:** [`Working.md`](Working.md)
- 📊 **Empirical Evaluation & Failure Audits:** [`Evaluation.md`](Evaluation.md)
- 💻 **Standalone Runnable Implementation:** [`Code.py`](Code.py)
