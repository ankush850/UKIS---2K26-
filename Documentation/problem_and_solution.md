# Problem Statement & Solution Documentation
**Project ID:** UKIS-2026  
**Theme:** Space Technology / Earth Observation  
**Project Name:** GEO-SRM (Super Resolution Mapping)

---

## 1. Executive Summary
The rapid monitoring of Earth's surface for agricultural policy, disaster management, and urban planning relies heavily on open-source satellite imagery. However, a significant gap exists between the spatial resolution offered by free public satellites and the resolution required for micro-level geospatial analysis. This project proposes an AI-driven software solution to bridge this gap, transforming medium-resolution imagery into high-resolution, actionable intelligence while introducing a novel "trust layer" to prevent AI hallucinations.

---

## 2. The Problem Statement Analysis (UKIS-2026)
**Target Objective:** *To develop a Super-Resolution Mapping (SRM) software capable of upscaling Sentinel-2 satellite imagery from a 10m Ground Sample Distance (GSD) to sub-4m (<4m) GSD.*

### The Core Challenges:
1. **The Resolution vs. Cost Trade-off:** 
   European Space Agency (ESA) Copernicus Sentinel-2 L2A provides excellent, freely available multi-spectral imagery with a global 5-day revisit cycle. However, its maximum spatial resolution is physically capped at **10 meters** (for RGB & NIR bands). While 10m is adequate for massive regional analysis, it is completely insufficient for:
   *   Mapping individual rural land boundaries and crop parcels.
   *   Detailed post-disaster infrastructure damage assessment (e.g., identifying washed-out rural bridges).
   *   To get sub-4m resolution (like SPOT or Planet Labs imagery), governments and organizations must purchase commercial satellite data, which is prohibitively expensive for continuous, wide-area monitoring.

2. **The "Black Box" Trust Deficit in AI:**
   While standard Deep Learning models (like GANs) can visually upscale images, they suffer from **hallucinations**. A neural network might invent or synthesize features—such as drawing a road or a farm boundary that does not physically exist—just to make the image look sharper. In critical sectors like defense or disaster response, policy decisions cannot be made on "invented" data. 

---

## 3. The Proposed Solution: GEO-SRM
To solve this, we have developed **GEO-SRM**, an end-to-end web platform that dynamically fetches 10m Sentinel-2 data and applies a mathematically rigorous Deep Learning pipeline to achieve a **2.5m GSD (a 4x spatial upscaling / 16x pixel density increase)**, completely satisfying the sub-4m requirement of the problem statement.

### How the Solution Works:
1. **Automated Ingestion:** The user selects a Bounding Box on an interactive web map. The system automatically queries the Copernicus Data Space Ecosystem (CDSE) to fetch the most recent cloud-free 10m imagery for that coordinate.
2. **PyTorch Super-Resolution Inference:** The 10m tile is sliced into patches and fed into a deep Residual Attention Network (SRMNet / SwinIR). The patches are then stitched back together using 2D Cosine Window blending to prevent any visual grid seams.
3. **Automated Feature Extraction:** Once the 2.5m image is generated, downstream computer vision algorithms automatically detect and extract critical infrastructure (roads, bridges, building footprints) as GIS-ready GeoJSON vectors.

---

## 4. Key Innovations & Unique Selling Propositions (USPs)
Our solution goes beyond merely producing a "sharper picture". The defining feature of our software is the **Hallucination-Aware Uncertainty Pipeline**.

To build trust with policy-makers and judges, the system mathematically proves which pixels are physically real and which are synthesized by the AI:
*   **Monte-Carlo Dropout (Epistemic Uncertainty):** During inference, the AI runs $N$ stochastic forward passes (with active dropout). If the AI is "guessing" a texture, the variance across these passes spikes.
*   **Spectral Angle Mapper (SAM):** The system checks if the new 2.5m pixels maintain the exact spectral color signature of the original 10m observation.
*   **The Confidence Heatmap:** The system overlays a transparent heatmap on the super-resolved map. **Green zones (100%)** mean the data is highly trustworthy and anchored in physical observations. **Red zones (0%)** alert the human analyst that the AI has likely hallucinated a feature, advising them to exercise caution before making a decision based on that specific patch of land.

---

## 5. Anticipated Impact & Applications
By converting free 10m satellite data into highly accurate, trustworthy 2.5m data, this software democratizes access to high-resolution geospatial intelligence:
*   **Precision Agriculture:** Enables governments to accurately monitor crop health (NDVI) and demarcate individual farm parcels without paying for commercial satellite passes.
*   **Rapid Disaster Response:** Allows rescue agencies to instantly generate high-resolution maps of flood-hit or earthquake-stricken areas, cleanly identifying blocked roads or damaged bridges.
*   **Cost Efficiency:** Saves immense capital for the government by maximizing the utility of open-source satellite programs (Copernicus) instead of relying on expensive private vendors.
