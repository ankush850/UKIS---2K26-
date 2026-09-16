# DMMC Disaster Severity Scoring & Automated Incident Briefings
## Multi-Hazard Quantitative Risk Indexing, Action Directives & Immutable Verification

**Project Identifier:** NETRA-D (UKIS-2026 Problem P-008)  
**Problem Owner:** Disaster Mitigation and Management Centre (DMMC), Department of Disaster Management, Government of Uttarakhand  
**Component:** Netra Aerial Severity & Reporting Engine (`backend/aerial/severity.py` & `report_generator.py`)  
**Target Beneficiaries:** State & National Disaster Response Forces (SDRF / NDRF), District Emergency Operation Centres (DEOC)  

---

## 1. Executive Summary & Operational Role

During rapid-onset emergencies, raw neural network outputs (segmentation masks and confidence numbers) are difficult for civilian district magistrates and rescue commanders to parse quickly under intense pressure.

Incident commanders require three concrete answers within 60 seconds of a drone flight:
1. **How bad is the situation?** (A standardized, objective 0–100 Disaster Severity Score).
2. **What should we do right now?** (Actionable Priority Directives: dispatch boats, earth-movers, or airdrop food).
3. **Can this assessment be legally and forensically trusted?** (Tamper-proof cryptographic hashes and blockchain provenance).

NETRA-D fulfills these requirements through the **DMMC Disaster Severity Engine** and the **Automated Incident Briefing Generator**.

```
+--------------------------------------------------------------------------------------------------+
|                            DMMC SEVERITY & REPORTING PIPELINE                                    |
+--------------------------------------------------------------------------------------------------+
|   Tri-Model Segmentation Outputs:                                                                |
|   - Effective Flood Percentage (Sanitized by SegFormer)                                          |
|   - Landslide Scar Percentage (TransLandSeg ViT-L)                                               |
|   - xBD Building Damage Breakdown (Microsoft SiamUnet)                                           |
|   - Overpass OSM Blocked Road Count (Corridor Intersections)                                     |
|                               |                                                                  |
|                               v                                                                  |
|   Mathematical Disaster Severity Index Formulation: S in [0, 100]                                |
|   Weighted Composite Function with Non-Linear Road Chokepoint Multipliers                        |
|                               |                                                                  |
|                               v                                                                  |
|   4-Tier Action Directive Assignment:                                                            |
|   - Level 4 [CRITICAL]: Immediate NDRF/SDRF Deployment & Air Evacuation                         |
|   - Level 3 [HIGH]: Secondary Perimeter Evacuation & Road Clearing                               |
|   - Level 2 [MODERATE]: Structural Audit & Drainage Diversion                                    |
|   - Level 1 [STABLE]: Routine Surveillance & Monitoring                                          |
|                               |                                                                  |
|                               v                                                                  |
|   Automated Incident Briefing Generator (HTML / PDF):                                            |
|   - Executive Incident Summary & Sector Coordinates                                              |
|   - Damage Delta & Road Corridor Status Table                                                    |
|   - Polygon Amoy Blockchain SHA-256 Provenance Stamp                                             |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation of the Disaster Severity Score

The composite Disaster Severity Score $\mathcal{S} \in [0, 100]$ integrates structural collapse, flood inundation, mudflow deposition, and road network disruptions:

### 2.1. Component Damage Distribution
$$\mathcal{S}_{\text{base}} = w_{\text{dest}} \cdot P_{\text{dest}} + w_{\text{maj}} \cdot P_{\text{maj}} + w_{\text{flood}} \cdot P_{\text{flood}}^{\text{eff}} + w_{\text{deb}} \cdot P_{\text{deb}} + w_{\text{min}} \cdot P_{\text{min}}$$

Where:
- $P_{\text{dest}}$: Percentage of destroyed building footprints ($w_{\text{dest}} = 0.35$).
- $P_{\text{maj}}$: Percentage of major structural building damage ($w_{\text{maj}} = 0.20$).
- $P_{\text{flood}}^{\text{eff}}$: Effective floodwater percentage sanitized by SegFormer ($w_{\text{flood}} = 0.20$).
- $P_{\text{deb}}$: Active landslide scar & mudflow percentage from TransLandSeg ($w_{\text{deb}} = 0.15$).
- $P_{\text{min}}$: Minor building damage percentage ($w_{\text{min}} = 0.05$).

### 2.2. Road Network Chokepoint Multiplier
In mountainous terrain, a single blocked national highway (e.g. NH-7) can cut off 50,000 people from food and medical supplies. Therefore, road blockages act as a non-linear multiplier:

$$\mu_{\text{road}} = 1.0 + \min(0.30, \, 0.10 \times N_{\text{blocked\_roads}})$$

### 2.3. Model Confidence Weighting
To prevent uncertain predictions from inflating severity scores:

$$\mathcal{S} = \text{Clamp}\left( \mathcal{S}_{\text{base}} \times \mu_{\text{road}} \times C_{\text{model}}, \quad 0, \quad 100 \right)$$

Where $C_{\text{model}} \in [0.70, 1.0]$ is the average confidence score across the ensemble.

---

## 3. Four-Tier Priority Directives

Based on the calibrated severity score $\mathcal{S}$, the engine assigns official DMMC command directives:

| Severity Range | Priority Level | Badge & Color | Command Action Directive |
| :---: | :---: | :---: | :--- |
| **80 – 100** | **LEVEL 4: CRITICAL** | 🔴 `#EF4444` `CRITICAL RESCUE PRIORITY` | **Immediate SDRF/NDRF Search & Rescue deployment.** Dispatch heavy earth-movers to blocked chokepoints; prioritize helicopter airlifts for destroyed residential sectors. |
| **60 – 79** | **LEVEL 3: HIGH** | 🟠 `#F97316` `HIGH PRIORITY EVACUATION` | **Evacuate threatened peripheral zones.** Establish emergency relief camps; deploy emergency temporary Bailey bridges along severed secondary routes. |
| **40 – 59** | **LEVEL 2: MODERATE** | 🟡 `#F59E0B` `MODERATE INFRASTRUCTURE RISK` | **Engineering damage inspection required.** Mobilize local PWD teams for road debris clearance; audit river embankments for secondary failure risks. |
| **0 – 39** | **LEVEL 1: STABLE** | 🟢 `#10B981` `MONITORING / STABLE` | **Normal tactical monitoring.** No immediate evacuation required; maintain routine drone sortie schedule. |

---

## 4. Automated Incident Briefing Generator (HTML / PDF)

The briefing engine (`/api/aerial/report-html`) automatically compiles a comprehensive, ready-to-print official disaster report formatted for direct submission to District Magistrates, the Chief Minister's Relief Cell, and NDRF commanders.

### Key Sections in the Official Briefing:
1. **Header & Metadata:**
   - Official DMMC emblem and State Disaster Management logo.
   - Incident Reference ID, Sector Name (e.g. Kedarnath Valley Sector 4), and Timestamp.
   - Aerial platform details (Sensor GSD, flight altitude, sortie duration).
2. **Executive Severity Summary:**
   - Large prominent priority badge (e.g. `LEVEL 4: CRITICAL RESCUE PRIORITY`).
   - Numerical Severity Score with interactive gauge breakdown.
3. **Multi-Hazard Impact Matrix:**
   - Total building footprints evaluated vs destroyed vs intact (mathematically normalized to $100.0\%$).
   - Total floodwater submergence area ($\text{m}^2$) and active landslide scar area ($\text{m}^2$).
4. **Road Accessibility & Evacuation Corridor Table:**
   - Detailed status of every surveyed OSM highway (NH-7, Badrinath Corridor) with blockage percentages and coordinates.
5. **Cryptographic Provenance Stamp:**
   - SHA-256 digest of the raw aerial imagery and model checkpoints.
   - Smart contract address and clickable PolygonScan verification link on Polygon Amoy.

---

## 5. REST API Specification

### Endpoint: `GET /api/aerial/report-html`
- **Query Parameters:**
  - `preset_id` *(optional)*: E.g., `kedarnath_flood` or `chamoli_landslide`.
  - `zone_name` *(optional)*: Custom name of the surveyed sector.
- **Response:**
  - `Content-Type: text/html`
  - Fully styled, print-optimized HTML document with CSS `@media print` rules, automatically rendering page breaks and high-contrast tables for PDF printing.
