# Model Technical Dossier: SegFormer B0 (ADE20K 150-Class Transformer)
## General Scene Understanding & Water/Sky Horizon Disambiguation Engine

**Project:** NETRA-D (UKIS-2026 Problem P-008)  
**Beneficiary:** Disaster Mitigation and Management Centre (DMMC), Department of Disaster Management, Government of Uttarakhand  
**Category:** Tactical Aerial Drone Intelligence (Micro Tier)  
**Model Name:** SegFormer B0  
**Hugging Face Hub ID:** [`nvidia/segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)  
**Target Perspectives:** Oblique UAV Sorties, Horizon Views, Multi-Angle Aerial Photos  

---

## 1. Executive Summary & Problem Context

In disaster response aerial operations, drones frequently capture oblique (tilted camera) imagery where the **horizon and sky** occupy significant portions of the upper frame.

Traditional drone disaster segmentation models (such as FloodNet) were trained strictly on **nadir (top-down 90°)** imagery where sky never appears in frame. Because these models never learned what sky looks like, they default to classifying blue regions as floodwater, producing **catastrophic false flood alerts (e.g. 66.66% floodwater on clear coastal images)**.

**SegFormer B0** directly resolves this failure mode architecturally. Pretrained on the comprehensive **ADE20K dataset**, it contains explicit, separate semantic classes for `sky` (Class 2), `water` (Class 21), `sea` (Class 26), `river` (Class 60), and `lake` (Class 128). Coupled with per-pixel softmax probability estimation, SegFormer reliably confirms sky horizons at **99.5% confidence**, enabling the consensus router to suppress false flood flags before they reach emergency incident commanders.

---

## 2. Technical Architecture & Mathematical Foundation

SegFormer utilizes a hierarchical Mix Transformer (MiT-B0) encoder paired with an All-MLP decoder:

1. **Efficient Spatial Reduction Attention:**
   $$\text{Attention}(\mathbf{Q}, \mathbf{K}', \mathbf{V}') = \text{Softmax}\left(\frac{\mathbf{Q} (\mathbf{K}')^T}{\sqrt{d_k}}\right) \mathbf{V}', \quad \text{where } \mathbf{K}', \mathbf{V}' \text{ are reduced by ratio } R$$
2. **Mix-FFN Positional-Encoding Free Blocks:** Uses $3 \times 3$ depth-wise convolutions to dynamically encode spatial coordinates, eliminating resolution interpolation artifacts.
3. **Softmax Class Distribution:**
   $$P(k \mid x, y) = \frac{e^{\mathbf{Z}_k(x, y)}}{\sum_{j=1}^{150} e^{\mathbf{Z}_j(x, y)}}$$
   - **Sky Mask:** $P(2 \mid x, y) \ge 0.35$
   - **Water Mask:** $\max_{c \in \{21, 26, 60, 128\}} P(c \mid x, y) \ge 0.35$

---

## 3. Quantitative Verification Benchmarks

| Test Scenario | Image Characteristics | FloodNet Standalone | SegFormer B0 Standalone | NETRA-D Consensus Verdict |
| :--- | :--- | :---: | :---: | :--- |
| **Sydney Beach Houses** | Oblique aerial coastal photo; prominent blue sky horizon; dry sunny weather. | ⚠️ **66.66% Floodwater** *(Catastrophic False Positive)* | 🔵 **53.15% Sky (99.5% Conf)**; **0.00% Water** | ✅ **Discrepancy Flagged:** `models_disagree=True`, false flood suppressed, classified as *Stable Baseline*. |
| **Meppadi Landslide** | Steep mountain slope scar; zero sky in frame. | 2.84% Water | **0.00% Sky; 0.00% Water** | ✅ **Ground Confirmed:** TransLandSeg active landslide scar given 100% dominant priority. |
| **Mountain River Corridor** | Fast-flowing riverbed surrounded by dry rocky terrain. | 11.2% Water | **14.2% River/Water (91.8% Conf)** | ✅ **Consensus Flood/Water:** Riverbed confirmed without false sky confusion. |

---

## 4. Operational Integration in NETRA-D

1. **Automated Horizon Auditing:** Operates concurrently alongside FloodNet in `/api/aerial/custom-inspection`.
2. **Discrepancy Detection (`models_disagree`):** When FloodNet water $\ge 10\%$ and SegFormer sky $\ge 10\%$ with near-zero water, the system overrides FloodNet's output and notes the horizon disambiguation in the incident briefing.
3. **Calibrated Severity Calculation:** Effective flood percentage is sanitized before computing the DMMC Disaster Severity Score (0–100).

---

## 5. Navigation & File Links

- 📖 **Overview & Operational Use Cases:** [`Overview.md`](Overview.md)
- ⚙️ **Mathematical Formulas & Architecture Diagram:** [`Working.md`](Working.md)
- 📊 **Empirical Evaluation & Decision Guide:** [`Evaluation.md`](Evaluation.md)
- 💻 **Standalone Runnable Implementation:** [`Code.py`](Code.py)
