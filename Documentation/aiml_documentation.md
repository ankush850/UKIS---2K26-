# UKIS-2026 — AI/ML Architecture & Model Documentation (NETRA-D)

Yeh documentation project me use hue saare **Machine Learning (ML)** aur **Deep Learning (DL)** models, unke workflows, unke training datasets, aur unke specific disaster response use-cases ko detail me explain karti hai.

Project do main operational tiers me divide hai:
1. **Macro Tier (Netra Satellite):** Sentinel-2 satellite imagery (10m resolution) ko super-resolve karke <4m (2.5m) tak le jana + Hallucination-Aware Uncertainty Layer + Polygon Amoy Blockchain Provenance.
2. **Micro Tier (Netra Aerial Drone):** Tactical UAV imagery par **Tri-Model Multi-Hazard Segmentation** (FloodNet + TransLandSeg + SegFormer ADE20K), **Microsoft SiamUnet** building damage assessment (xBD standard), aur real-time **Overpass OSM** road blockage analysis.

---

## 1. Macro Tier: Satellite Super-Resolution (SR) Models (10m → 2.5m)

Image ko 10m se 2.5m ($4\times$ spatial upscaling, $16\times$ pixel density gain) karne ke liye project me state-of-the-art Deep Learning models integrate kiye gaye hain:

### A. HAT (Hybrid Attention Transformer) — Official Production Engine
- **Kya hai:** Transformer-based super-resolution network jo Window-based Multi-Head Self-Attention (W-MSA) aur Channel Attention Blocks ko combine karta hai.
- **Weights:** Trained on **WorldStrat Dataset** (Sentinel-2 L2A $\leftrightarrow$ SPOT 6/7 1.5m paired scenes).
- **Kyu use kiya:** Highest SSIM (**0.1667**) aur lowest spectral distortion (**5.67° SAM**). Remote sensing me narrow rural roads, agricultural plots, aur bridges ki sharp boundaries preserve karne ke liye best model hai.

### B. SRM-Net (Residual Channel Attention + MC-Dropout) — Interactive Trust Engine
- **Kya hai:** Custom 8 Residual Blocks + Squeeze-and-Excitation Channel Attention architecture jisme Test-Time Spatial Dropout ($p=0.20$) embedded hai.
- **Kyu use kiya:** Normal models sirf ek sharp image dete hain, par yeh model real-time me **epistemic uncertainty ($\boldsymbol{\sigma}^2$)** compute karta hai. CPU par 8 Monte-Carlo passes sirf **3.7 seconds** me run ho jate hain.

### C. CARN (Cascading Residual Network) — Low-Power Edge Baseline
- **Kya hai:** ESA (European Space Agency) EvoLand & WorldStrat project ka cascading skip-connection CNN.
- **Kyu use kiya:** Offline disaster field kits aur low-power devices ke liye ultra-fast baseline (**420 ms**, 195 MB RAM).

### D. Real-ESRGAN (RRDBNet Generator) — Hallucination Audit Baseline
- **Kya hai:** Relativistic GAN (RaGAN) based texture generator.
- **Kyu use kiya:** Scientific proof ke liye — yeh demonstrate karta hai ki commercial GANs remote sensing me visually sharp par fake textures (hallucinations) bana dete hain, jisse cycle consistency error spike karta hai.

---

## 2. The USP: Hallucination-Aware Uncertainty Layer

Remote sensing me unconstrained AI hallucinations bohot dangerous ho sakti hain (jaise khet me fake road dikha dena). NETRA isko mathematically audit karta hai:

1. **Monte-Carlo Dropout (MC-Dropout):**  
   Inference ke waqt dropout active rehta hai (`force_dropout=True`). Model ek hi image ko $N=8$ times stochastic passes me process karta hai:
   $$\boldsymbol{\sigma}^2(x, y) = \frac{1}{N} \sum_{i=1}^N \left( \hat{\mathbf{y}}_i(x, y) - \boldsymbol{\mu}_{\text{SR}}(x, y) \right)^2$$
   Agar model confident hai toh variance $\approx 0$, agar hallucinate kar raha hai toh variance spike karega (Heatmap pe Red mask).
2. **ESA `opensr-test` Cycle Consistency:**  
   Super-resolved 2.5m image ko sensor PSF se wapas 10m par downsample kiya jata hai. Agar original Sentinel-2 input se deviation zyada hai, toh confidence penalize hoti hai.
3. **Spectral Angle Mapper (SAM):**  
   Check karta hai ki true-color RGB/NIR multi-spectral bands ke ratios artificially distort toh nahi hue.
4. **Physical Optical Cloud Shield:**  
   ESA SCL layer aur multi-spectral optical reflectance se clouds aur cloud shadows ko detect karta hai, unka confidence **0.0%** clamp karta hai aur fake detections suppress karta hai.

---

## 3. Micro Tier: Tactical Drone Tri-Model Hazard Engine

Standard single-model drone inspection systems real-world field photos me fail ho jate hain. NETRA-D ne is root cause ko solve karne ke liye **Tri-Model Ensemble** banaya hai:

```
+---------------------------------------------------------------------------------------------+
|                          TRI-MODEL AERIAL HAZARD ARCHITECTURE                               |
+---------------------------------------------------------------------------------------------+
|                                  Input Drone Photo                                          |
|                                          |                                                  |
|         +--------------------------------+--------------------------------+                 |
|         |                                |                                |                 |
|         v                                v                                v                 |
|   [ Model 1: FloodNet ]      [ Model 2: TransLandSeg ]       [ Model 3: SegFormer B0 ]      |
|   DeepLabV3+ (4 Classes)     SAM ViT-L (Bijie Dataset)       ADE20K (150 Scene Classes)     |
|   - Flooded Buildings        - Dedicated Landslide Scars     - Explicit Sky Class (Id: 2)   |
|   - Flooded Roads            - Mudflow Debris Fields         - Water/River/Lake Classes     |
|   - Water & Submergence      - Bare-Soil Displacement        - Softmax Confidence Scoring   |
|         |                                |                                |                 |
|         +--------------------------------+--------------------------------+                 |
|                                          |                                                  |
|                                          v                                                  |
|                         [ Intelligent Consensus Router ]                                    |
|                         - Sky Horizon Disambiguation                                        |
|                         - Missing Landslide Scar Resolution                                 |
|                         - Discrepancy Detection (models_disagree)                           |
+---------------------------------------------------------------------------------------------+
```

### A. Model 1: FloodNet DeepLabV3+
- **Kya hai:** ResNet/DeepLabV3+ backbone jo nadir drone photos par trained hai.
- **Classes:** `Background`, `Flooded-Building`, `Flooded-Road`, `Water`.
- **Limitation:** Is model me **landslide ka koi class nahi hai**, aur yeh sirf nadir (seedha 90° neeche) shots par trained hai, isliye oblique shots me aasmaan (sky) ko paani samajh leta hai.

### B. Model 2: TransLandSeg (SAM ViT-L · Bijie Landslide Checkpoint)
- **Kya hai:** Meta ke Segment Anything Model (SAM) ka 304M-parameter Vision Transformer Large (ViT-L) backbone, jo **Bijie Landslide Dataset** par fine-tune kiya gaya hai.
- **Kyu add kiya:** FloodNet me bare-soil landslide scars ka class missing tha. Wayanad ya Chamoli jaise pahadi disaster me TransLandSeg **active landslide scars aur mudflows ko 93%+ confidence ke saath isolate karta hai** (Meppadi benchmark: 28.74% landslide scar detected).

### C. Model 3: SegFormer B0 (ADE20K 150-Class Transformer)
- **Kya hai:** NVIDIA ka lightweight hierarchical Vision Transformer (`nvidia/segformer-b0-finetuned-ade-512-512`), trained on ADE20K scene dataset.
- **Kyu add kiya:** ADE20K me `sky` (Class 2), `water` (Class 21), `sea` (Class 26), `river` (Class 60), aur `lake` (Class 128) alag-alag clearly labeled hain.
- **Role:** Har pixel par softmax probability distribution calculate karta hai. Jab drone oblique angle se photo leta hai aur blue horizon dikhta hai, toh SegFormer usse **99.5% confidence se "Sky"** identify karta hai aur FloodNet ke false water flag ko cancel kar deta hai (Sydney beach benchmark: 53.15% sky, 0.00% water).

### D. Intelligent Hazard Consensus & Discrepancy Router
- **Discrepancy Detection:** Agar FloodNet water $\ge 10\%$, SegFormer water $< 3\%$, aur SegFormer sky $\ge 10\%$ hai, toh system `models_disagree = True` flag karta hai aur briefing me operator ko batata hai ki yeh sky horizon hai, paani nahi!
- **Auto Disaster Mode:** Agar landslide scar $\ge 4\%$ aur dominant hai, toh TransLandSeg ko lead banakar landslide mode activate hota hai. Agar floodwater confirmed hai, toh flood mode activate hota hai.

---

## 4. Micro Tier: Building Damage & Road Passability

### A. Microsoft SiamUnet Pre/Post Damage Inspector (xBD Standard)
- **Kya hai:** Siamese Convolutional Neural Network jo pre-disaster aur post-disaster paired photos ko compare karta hai.
- **Categories:** Global xBD / HAZUS standard:
  - `Destroyed` (Puri tarah barbaad)
  - `Major Damage` (Badi darare/chhat girna)
  - `Minor Damage` (Halka nuksan)
  - `Intact` (Surakshit)
- **Mathematical Integrity:** Saare damage aur intact percentages strictly **$100.0\%$ par mathematically normalize** hote hain.

### B. Live OpenStreetMap Overpass Road Passability
- Custom image ke EXIF GPS coordinates se region of interest nikalta hai.
- Real-time **OpenStreetMap Overpass API** se highway geometries (NH-7, Badrinath, Kedarnath road networks) pull karta hai.
- Detected flood aur debris polygons ke saath vector buffer intersection run karke blocked roads aur critical evacuation chokepoints identify karta hai.

### C. Simulated 60-Frame Drone HUD Sortie Simulator
- Static drone photos par Ken Burns 60-frame trajectory simulate karta hai.
- **Har single frame par live PyTorch inference** run karke bounding boxes, altitude, air speed, aur hazard percentage canvas par render karta hai.

---

## 5. Summary Table of All Models

| Model Name | Backbone Architecture | Parameters | Dataset | Primary Role |
| :--- | :--- | :---: | :--- | :--- |
| **HAT** | Hybrid Attention Transformer | 1.38 M | WorldStrat (Sentinel-2 / SPOT 6/7) | 10m $\to$ 2.5m Super-Resolution |
| **SRM-Net** | Residual Attention + MC-Dropout | 0.93 M | Custom S2 BOA Reflectance | Real-Time Uncertainty & Heatmaps |
| **CARN** | Cascading Residual Network | 0.98 M | ESA EvoLand Benchmark | Low-Power Offline Field Baseline |
| **Real-ESRGAN** | RRDBNet Generator | 16.70 M | Synthetic Degradations + RaGAN | Hallucination Proof Benchmark |
| **FloodNet** | DeepLabV3+ / ResNet | 26.70 M | FloodNet Nadir Drone Dataset | Nadir Floodwater & Inundation |
| **TransLandSeg** | SAM Vision Transformer (ViT-L) | 304.0 M | Bijie Landslide Dataset | Dedicated Mountain Landslide Scars |
| **SegFormer B0** | Hierarchical Transformer (MixFFN) | 3.71 M | ADE20K 150-Class Dataset | Sky vs Water Disambiguation |
| **SiamUnet** | Siamese CNN / Feature Concatenation | 7.80 M | xBD Disaster Building Dataset | Pre/Post Building Damage (4-tier) |

Yeh comprehensive AI architecture ensure karti hai ki NETRA-D satellite wide-area monitoring se lekar drone tactical rescue tak har scale par scientifically accurate aur reliable output provide kare.
