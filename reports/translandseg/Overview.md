# TransLandSeg (SAM ViT-L · Bijie Dataset) — Overview

## 1. Introduction (Model Kya Hai?)

**TransLandSeg** ek dedicated Transformer-based deep learning segmentation model hai jise high-resolution aerial aur drone imagery se **mountain landslides, mudflow debris tongues, aur bare-soil slope failures** ko accurately detect karne ke liye train kiya gaya hai.

Iska core backbone **Meta AI ke Segment Anything Model (SAM) ka Vision Transformer Large (ViT-L)** hai, jise mountain landslide terrain par **Bijie Landslide Dataset** ke zariye fine-tune kiya gaya hai.

Simple shabdon me:
> **TransLandSeg ka kaam hai pahadi ilakon (jaise Uttarakhand, Kedarnath, Wayanad) me barish ya bhukamp ke baad hui zameen khiskan (landslide scars aur mudflows) ko drone/aerial photo me se exact pixel-level boundary ke saath pehchanna.**

### The Problem It Solves (Kyu Zaroorat Padi?):
Disaster response me standard drone models (jaise FloodNet) sirf baadh (flood) ke classes jaante hain: `water`, `flooded-road`, `flooded-building`. Unke paas **mitti, malba ya landslide ka koi class nahi hota**. Is wajah se jab pahad toot kar girta hai (jaise Meppadi, Wayanad ya Chamoli me), toh flood model usse detect hi nahi kar pata (near-zero detection). **TransLandSeg ne is root-cause gap ko 100% solve kiya hai.**

---

## 2. Real-Life Use Cases (Asli Duniya me Kaha Use Hota Hai?)

TransLandSeg ko NETRA-D platform me **Dedicated Landslide & Debris Flow Detection Engine** banaya gaya hai.

### 1. Himalayan Mountain Slope Failure & Mudflow Mapping (DMMC Uttarakhand)
- **Problem:** Monsoon ke dauran Kedarnath, Chamoli, aur Mandakini valley me cloudbursts ke baad massive mudslides aate hain jo pure pahaad ko cheerte hue nikalte hain.
- **TransLandSeg Solution:** Drone sortie image me se red/brown bare-soil mudflow scar ko isolated vector mask me convert karta hai, jisse disaster management teams ko pahaad ke active failure zone ka exact area (sq. meters) pata chalta hai.

### 2. Evacuation Corridor & Blocked Mountain Highway Triage (NH-7 / Badrinath Corridor)
- **Problem:** Pahaad se gira malba jab National Highway par girta hai, toh rescue teams ko turant pata hona chahiye ki kya rasta zameen khiskan se block hua hai ya bas paani bhara hai.
- **TransLandSeg Solution:** Malbe (debris tongue) ke polygon ko OpenStreetMap (OSM) highway geometries ke saath intersect karta hai aur "HIGHWAY BLOCKED BY ACTIVE LANDSLIDE" alert trigger karta hai.

### 3. Debris Fan & River Damming Hazard Identification (GLOF / Flash Flood Warning)
- **Problem:** Landslide ka malba nadi (river) me gir kar temporary dam bana deta hai, jo tootne par aage achanak flash flood la sakta hai (jaise Rishiganga disaster).
- **TransLandSeg Solution:** Nadi ke kinare par gire debris fan ko isolate karta hai aur riparian blockage severity calculate karta hai.

### 4. Post-Disaster Rehabilitation & Compensation Audits (SDRF / NDRF)
- **Problem:** Gaon ke khet ya ghar zameen khiskan me dhab gaye, par unka physical survey karna pahado me bohot khatarnak aur samay-khapau hota hai.
- **TransLandSeg Solution:** Drone orthomosaics se affected slope area calculate karke digital assessment report provide karta hai.

---

## 3. Advantages (Fayde)

- **Dedicated Landslide Class Representation:** FloodNet ke mukable bare-soil displacement aur mudflow scars ko genuinely pehchanta hai (Zero false-zero detection).
- **Vision Transformer (ViT-L) Global Context:** 304M parameters aur large receptive field ke saath yeh chhote rockfalls se lekar 500-meter lambe continuous mudslides ko smoothly capture karta hai.
- **High Confidence in Rocky/Himalayan Terrain:** Meppadi landslide benchmark me **28.74% landslide scar ko 93.1% confidence** ke saath isolate kiya.
- **Robust Against Soil Moisture Variations:** Geeli mitti (wet mud) aur sukhi mitti (dry scree) dono ke spectral patterns ko alag-alag handle karta hai.
- **Seamless Polygonization:** Segmented binary masks turant Douglas-Peucker polygonization se GeoJSON vector format me convert ho jate hain.

---

## 4. Disadvantages (Kamiyan / Limitations)

- **Large Model Footprint (ViT-L Backbone):** SAM ViT-L ek heavy transformer model hai (~304M parameters, checkpoint size ~40MB - 350MB depending on quantization).
- **CPU Inference Latency:** Heavy self-attention matrices ki wajah se CPU par ek single drone tile ko process karne me 1.5–2.5 seconds lagte hain (NVIDIA T4/A100 GPU par <120 ms).
- **Scree vs. Fresh Landslide Ambiguity:** Sukhi pahadi dhalano par purane sukhe pathar (scree/talus) ko kabhi-kabhi low probability par confuse kar sakta hai (isliye 4.0% hazard threshold rakha gaya hai).

---

## 5. Assumptions (Kis Data / Conditions Pe Best Kaam Karta Hai?)

1. **Sub-Decimeter to Low-Altitude Aerial Imagery:** Ground Sample Distance (GSD) 5cm se lekar 50cm per pixel ke beech honi chahiye (standard commercial UAV flight altitude 60m–120m AGL).
2. **Clear Optical Visibility:** Heavy rain ya dense fog me optical contrast reduce hone par confidence drop ho sakti hai.
3. **RGB Color Spectrum:** Standard 3-channel (Red, Green, Blue) optical sensors ke saath natively operate karta hai.
4. **Mountainous & Hilly Topography:** Hilly terrain, steep valleys, aur forested slope cuts me iski accuracy sabse high hoti hai.
