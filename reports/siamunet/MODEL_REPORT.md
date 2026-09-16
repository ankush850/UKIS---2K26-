# Model Technical Dossier: Microsoft SiamUnet
## Pre/Post Disaster Building Damage Assessment Engine (xBD / HAZUS Standard)

**Project:** NETRA-D (UKIS-2026 Problem P-008)  
**Beneficiary:** Disaster Mitigation and Management Centre (DMMC), Department of Disaster Management, Government of Uttarakhand  
**Category:** Tactical Aerial Drone Intelligence (Micro Tier)  
**Model Name:** Microsoft SiamUnet  
**Architecture:** Siamese Dual-Branch Weight-Shared CNN with Deconvolutional Skip-Connection Decoder  
**Training Standard:** xBD Global Building Damage Benchmark  
**Target Scenarios:** Post-Earthquake, Flash Flood, and Landslide Structural Impact Surveys  

---

## 1. Executive Summary & Problem Context

Following sudden catastrophic events (such as Himalayan cloudbursts or seismic activity), emergency commanders and humanitarian agencies need rapid, objective data on structural integrity.

Standard single-image detection models can identify whether a building exists, but cannot determine whether the building has lost its roof or suffered foundation shifts compared to its pre-disaster state.

**Microsoft SiamUnet** processes paired pre- and post-disaster aerial imagery through a weight-shared Siamese network, classifying individual building footprints into the four international **xBD / FEMA HAZUS damage categories**:
1. `Destroyed`
2. `Major Damage`
3. `Minor Damage`
4. `Intact`

The system calculates a composite **Structural Integrity Score (0% to 100%)** and mathematically normalizes all damaged building percentages to strictly **$100.0\%$**, providing legally defensible evidence for government compensation audits and search-and-rescue dispatch.

---

## 2. Technical Architecture & Mathematical Foundation

SiamUnet passes pre- and post-disaster images through identical encoder branches with shared parameter weights $\mathbf{W}$:

$$\mathbf{F}_{\text{pre}} = \mathcal{E}(\mathbf{I}_{\text{pre}}; \, \mathbf{W}), \quad \mathbf{F}_{\text{post}} = \mathcal{E}(\mathbf{I}_{\text{post}}; \, \mathbf{W})$$

The absolute structural difference vector $\Delta \mathbf{F} = |\mathbf{F}_{\text{pre}} - \mathbf{F}_{\text{post}}|$ is concatenated with both features and decoded:

$$\mathbf{F}_{\text{fused}} = \text{Concat}([\mathbf{F}_{\text{pre}}, \, \mathbf{F}_{\text{post}}, \, \Delta \mathbf{F}])$$

$$\text{Integrity} = \frac{1.0 \cdot N_{\text{intact}} + 0.7 \cdot N_{\text{minor}} + 0.2 \cdot N_{\text{major}} + 0.0 \cdot N_{\text{destroyed}}}{N_{\text{total}}} \times 100.0\%$$

---

## 3. Quantitative Verification Summary

| Metric / Dimension | Value | Operational Context |
| :--- | :---: | :--- |
| **xBD Composite F1 Score** | **80.6%** | Official international damage benchmark accuracy |
| **Building Localization Recall** | **86.4%** | Successfully captures building perimeter vectors |
| **Damage Normalization Integrity**| **$\sum \text{Damaged} \equiv 100.0\%$** | Zero arithmetic drift in DMMC briefings |
| **Execution Speed (CPU, 512x512)**| **~420 – 550 ms** | Fast enough for tactical field operations |

---

## 4. Operational Integration in NETRA-D

1. **Dual Image Comparison Mode:** Triggered in `/api/aerial/custom-inspection` whenever a `pre_image_b64` is provided alongside the post-disaster sortie photo.
2. **Single Image Fallback:** If pre-disaster imagery is unavailable, gracefully shifts to detection-only mode without crashing or halting the pipeline.
3. **Automated Incident Briefing:** Incorporates structural integrity percentages and damage overlays into the official DMMC disaster report.

---

## 5. Navigation & File Links

- 📖 **Overview & Operational Use Cases:** [`Overview.md`](Overview.md)
- ⚙️ **Mathematical Formulas & Architecture Diagram:** [`Working.md`](Working.md)
- 📊 **Empirical Evaluation & Decision Guide:** [`Evaluation.md`](Evaluation.md)
- 💻 **Standalone Runnable Implementation:** [`Code.py`](Code.py)
