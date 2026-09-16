# Tech Stack Documentation: NETRA-D (UKIS-2026)
## Dual-Tier Disaster Intelligence & Infrastructure Assessment Platform

This document provides a comprehensive technical breakdown of all libraries, frameworks, architectures, and protocols utilized across the **NETRA-D** platform.

---

## 1. Backend Infrastructure & API Layer

*   **Runtime Environment:** `Python 3.10` / `Python 3.11`
*   **Web Framework:** `FastAPI` (>=0.100.0)
    *   *Role:* High-performance asynchronous REST API routing, WebSocket real-time video streaming, and automatic interactive OpenAPI/Swagger specification (`/docs`).
*   **ASGI Web Server:** `Uvicorn` (>=0.23.0)
    *   *Role:* Asynchronous event-loop execution engine supporting non-blocking concurrent requests.
*   **Data Serialization & Validation:** `Pydantic` (>=2.0.0)
    *   *Role:* Strict type enforcement for GeoJSON structures, bounding box tuples, model request payloads, and telemetry data models.
*   **Multipart Handling:** `python-multipart`
    *   *Role:* Multipart form decoding for multi-megabyte aerial image and GeoTIFF file uploads.

---

## 2. Artificial Intelligence & Deep Learning Models

### 2.1. Frameworks & Neural Compute
*   **PyTorch (>=2.0.0) & TorchVision (>=0.15.0):**
    *   *Role:* Core tensor computation, backpropagation, and stochastic Monte-Carlo dropout execution across CUDA GPUs and x86 CPUs.
*   **Hugging Face Transformers (>=4.30.0):**
    *   *Role:* Powers **SegFormer B0** (`nvidia/segformer-b0-finetuned-ade-512-512`), executing hierarchical vision transformer inference and per-pixel softmax confidence generation.
*   **Timm (PyTorch Image Models):**
    *   *Role:* Vision backbone utilities and feature extraction layers.

### 2.2. Neural Architecture Zoo
1.  **HAT (Hybrid Attention Transformer):**
    *   *Architecture:* Window-based Multi-Head Self-Attention (W-MSA) + Channel Attention Blocks.
    *   *Role:* Macro Tier Sentinel-2 10m $\to$ 2.5m super-resolution ($4\times$ spatial gain).
2.  **SRM-Net (Residual Channel Attention):**
    *   *Architecture:* 8 Residual Blocks + Squeeze-and-Excitation + Spatial Dropout ($p=0.20$).
    *   *Role:* Real-time epistemic uncertainty estimation ($\boldsymbol{\sigma}^2$) via 8-pass Monte-Carlo sampling.
3.  **TransLandSeg (SAM ViT-L Backbone):**
    *   *Architecture:* 304M-parameter Vision Transformer Large fine-tuned on Bijie Landslide Dataset.
    *   *Role:* Dedicated mountain landslide scar, mudflow, and bare-soil displacement segmentation.
4.  **SegFormer B0 (ADE20K Pretrained):**
    *   *Architecture:* Lightweight hierarchical transformer with Mix-FFN decoder.
    *   *Role:* General scene water/sea/river/lake segmentation with explicit sky horizon filtering.
5.  **FloodNet DeepLabV3+:**
    *   *Architecture:* ResNet backbone with Atrous Spatial Pyramid Pooling (ASPP).
    *   *Role:* Nadir drone floodwater and structural inundation segmentation.
6.  **Microsoft SiamUnet:**
    *   *Architecture:* Dual-branch Siamese CNN with feature concatenation and deconvolutional decoder.
    *   *Role:* Pre/post disaster paired building damage classification adhering to the xBD/HAZUS standard.
7.  **Real-ESRGAN (RRDBNet Generator):**
    *   *Architecture:* Residual-in-Residual Dense Blocks trained with RaGAN loss.
    *   *Role:* Hallucination comparative baseline.

---

## 3. Remote Sensing, Geospatial & Mathematical Computing

*   **SentinelHub-Py (>=3.11.0):**
    *   *Role:* Official Python client interfacing with the Copernicus Data Space Ecosystem (CDSE) to fetch live Sentinel-2 L2A tiles.
*   **OpenStreetMap Overpass API:**
    *   *Role:* Real-time retrieval of vector highway geometries for Himalayan corridors (NH-7, Badrinath, Kedarnath).
*   **GDAL, Rasterio & Tifffile:**
    *   *Role:* Multi-band GeoTIFF georeferencing, projection warping (EPSG:4326 to UTM), and 16-bit/32-bit floating-point radiance manipulation.
*   **Shapely & GeoJSON:**
    *   *Role:* 2D vector geometry operations: polygon buffering, line-string intersection, and tactical hazard export.
*   **NumPy & SciPy:**
    *   *Role:* Array manipulation, Monte-Carlo variance, cycle consistency downsampling via `scipy.ndimage.zoom`, and Reinhard color matching.
*   **Scikit-Image & OpenCV:**
    *   *Role:* Image quality metrics (PSNR, SSIM, SAM, ERGAS), morphological thinning (Zhang-Suen), Otsu thresholding, and Douglas-Peucker polygon simplification.

---

## 4. Blockchain & Cryptographic Provenance (NETRA)

*   **Blockchain Network:** **Polygon Amoy Testnet (Chain ID `80002`)**
*   **Smart Contract:** `TileProvenance.sol` (Solidity `0.8.20`)
    *   *Contract Address:* [`0x2287c88b7764A9D386FeE490958e0aF1316b8F10`](https://amoy.polygonscan.com/address/0x2287c88b7764A9D386FeE490958e0aF1316b8F10)
*   **Web3 Client:** `Web3.py` (>=6.0.0)
    *   *Role:* Transaction construction, private-key signing, gas estimation, and RPC event querying.
*   **Cryptographic Primitives:**
    *   *Hashing:* SHA-256 raster fingerprints for bit-level tamper detection.
    *   *Coordinate Precision:* Integer scaling by $10^6$ for sub-11cm geographic anchoring on-chain without floating-point errors.

---

## 5. Frontend & Tactical User Interface

*   **Zero-Build Vanilla Architecture:** HTML5, CSS3, and Modern ES6+ JavaScript. No npm build steps, webpack, or external framework overhead.
*   **Web Mapping Engine:** `Leaflet.js` (v1.9.4) + `Leaflet.draw`
    *   *Role:* Interactive world map navigation, custom Area of Interest (AOI) bounding box drawing, and vector polygon rendering.
*   **High-Resolution Deep Zoom:** `OpenSeadragon` (v4.1.1)
    *   *Role:* Smooth, high-performance rendering of multi-gigapixel super-resolved rasters with synchronized dual viewports.
*   **Aerospace Defense Aesthetics:** Custom dark-mode CSS with glassmorphism, glowing hazard borders, interactive telemetry HUDs, and real-time confidence threshold sliders.
