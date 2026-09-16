# NETRA-D — Master Technical Documentation Portal
## Comprehensive Guide to Architecture, Mathematical Physics, AI/ML Models & Disaster Operations

**Project Identifier:** UKIS-2026 (Problem P-008)  
**Problem Owner:** Disaster Mitigation and Management Centre (DMMC), Department of Disaster Management, Government of Uttarakhand  
**Target Beneficiaries:** DMMC Uttarakhand, SDRF/NDRF, DoNER / NESAC / ISRO  
**Platform Name:** NETRA-D (Neural Enhancement for Terrain & Resolution Amplification + Tactical Drone Assessment)  

---

## 🧭 Master Documentation Index & Navigator

This directory contains the exhaustive technical specification, scientific derivations, operational workflows, and model evaluations for the entire NETRA-D dual-tier intelligence platform.

| Document | Primary Audience | Key Contents & Operational Scope | Link |
| :--- | :--- | :--- | :---: |
| **Master Technical Documentation** | All Evaluators & Architects | Complete 13-section platform manual: Problem context, USP, Satellite SR, Drone Micro Tier, Blockchain, REST API. | [📖 View Master Doc](MASTER_PROJECT_DOCUMENTATION.md) |
| **Tri-Model Aerial Tactical Hazard Engine** | AI/ML Judges & UAV Engineers | Deep dive into FloodNet, TransLandSeg (SAM ViT-L), SegFormer ADE20K, mathematical consensus routing, and sky horizon disambiguation. | [🚁 View Tri-Model Doc](TRI_MODEL_AERIAL_INSPECTION.md) |
| **OpenStreetMap Road Accessibility & Routing** | GIS Specialists & SDRF Planners | Real Overpass OSM query engine, vector polygon buffering, road blockage calculation, and evacuation corridor analysis. | [🛣️ View Road Doc](ROAD_ACCESSIBILITY_AND_OSM_INTEGRATION.md) |
| **Live Drone HUD Simulation & Streaming** | Defense Evaluators & Pilots | 60-frame Ken Burns trajectory generator, per-frame PyTorch inference, WebSocket telemetry protocol, and DGCA flight safety rules. | [🎮 View HUD Stream Doc](LIVE_DRONE_HUD_AND_STREAMING.md) |
| **DMMC Severity Scoring & Incident Briefings** | Emergency Commanders & Magistrates | 0–100 Disaster Severity Index mathematical weighting, 4-tier action directives, automated HTML/PDF briefing generator, and blockchain audit links. | [📋 View Severity Doc](DMMC_INCIDENT_BRIEFING_AND_SEVERITY.md) |
| **Mathematical Formulas & Physics Derivations** | Remote Sensing Scientists & Theorists | Rigorous LaTeX equation derivations: MC-Dropout variance, ESA cycle consistency, SAM angles, ViT patch embeddings, and structural integrity. | [📐 View Math Formulas](FORMULAS_AND_MATHEMATICAL_DERIVATIONS.md) |
| **Model Benchmark & Selection Guide** | ML Researchers & Evaluators | Exhaustive comparative benchmarks across all models (HAT, SRM-Net, CARN, Real-ESRGAN, TransLandSeg, SegFormer, FloodNet, SiamUnet). | [📊 View Model Benchmarks](MODEL_BENCHMARK_AND_SELECTION_GUIDE.md) |
| **AI/ML Architecture & Model Guide (Hindi/Hinglish)** | Team Members & Indian Evaluators | Hindi/Hinglish deep dive explaining every ML model, why it was chosen ("Kyu use kiya"), and how it solves real field failures. | [🇮🇳 View AI/ML Doc](aiml_documentation.md) |
| **System Architecture & Data Flows** | Software Architects & Full-Stack Leads | High-level architectural diagrams, component breakdown, concurrency models, and frontend-backend interactions. | [🏗️ View Architecture Doc](architecture_documentation.md) |
| **Data Flow Diagrams (DFD Levels 0, 1, 2, 3)** | Software Engineers & System Analysts | Multi-level DFDs tracking data from Copernicus satellite ingestion and drone uploads to PyTorch tensors and Leaflet viewports. | [🔄 View DFD Doc](dfd_documentation.md) |
| **Complete Technology Stack Guide** | DevOps & System Administrators | Comprehensive inventory of all libraries (PyTorch, Transformers, Web3, Overpass OSM, GDAL, FastAPI, Leaflet). | [💻 View Tech Stack Doc](tech_stack_documentation.md) |
| **Problem Statement Analysis & Solution Overview** | Hackathon Evaluators & Jury | Executive breakdown of Problem P-008, the Himalayan optical data paradox, and how the dual-tier system bridges the divide. | [🎯 View Problem Doc](problem_and_solution.md) |
| **Implementation Plan & Feature Roadmap** | Project Managers & Developers | Architectural milestones, completed deliverables, and future satellite-drone cross-calibration roadmap. | [📝 View Implementation Plan](implementation_plan.md) |

---

## 🗺️ Reading Pathways by Stakeholder Role

### 1. For Hackathon Judges & Executive Evaluators (10-Minute Overview):
1. Start with [Problem Statement & Solution Overview](problem_and_solution.md) to grasp Problem P-008 and the Himalayan optical paradox.
2. Read [Tri-Model Aerial Tactical Hazard Engine](TRI_MODEL_AERIAL_INSPECTION.md) to understand how we solved the real-world drone failures (Sydney beach sky false alarms and missing landslide scars).
3. Review [DMMC Severity Scoring & Incident Briefings](DMMC_INCIDENT_BRIEFING_AND_SEVERITY.md) to see how intelligence converts into ready-to-print emergency command briefings.

### 2. For AI/ML Researchers & Remote Sensing Scientists:
1. Examine [Mathematical Formulas & Physics Derivations](FORMULAS_AND_MATHEMATICAL_DERIVATIONS.md) for equation derivations of MC-Dropout variance, ESA cycle consistency, and SAM radiance preservation.
2. Consult [Model Benchmark & Selection Guide](MODEL_BENCHMARK_AND_SELECTION_GUIDE.md) for empirical SPOT 6/7 and Bijie ground truth matrices.
3. Open individual model technical dossiers in [`reports/`](../reports/README.md).

### 3. For Field Incident Commanders (DMMC, SDRF, NDRF):
1. Review [OpenStreetMap Road Accessibility & Routing](ROAD_ACCESSIBILITY_AND_OSM_INTEGRATION.md) for highway blockage and chokepoint detection.
2. Review [Live Drone HUD Simulation & Streaming](LIVE_DRONE_HUD_AND_STREAMING.md) for live telemetry and DGCA weather flight envelopes.
3. Review [DMMC Severity Scoring & Incident Briefings](DMMC_INCIDENT_BRIEFING_AND_SEVERITY.md) for evacuation priority directives.
