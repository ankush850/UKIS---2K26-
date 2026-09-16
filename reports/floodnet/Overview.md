# FloodNet DeepLabV3+ — Overview

## 1. Introduction (Model Kya Hai?)

**FloodNet DeepLabV3+** ek specialized semantic segmentation deep learning model hai jise low-altitude tactical drone imagery se **baadh ke paani (floodwater), doobi hui sadko (flooded roads), aur doobe hue makanon (flooded buildings)** ko pixel-by-pixel classify karne ke liye banaya gaya hai.

Iska architecture **DeepLabV3+ with ResNet backbone** par adharit hai aur isme **Atrous Spatial Pyramid Pooling (ASPP)** module shamil hai.

Simple shabdon me:
> **FloodNet ka kaam hai drone se li gayi top-down (nadir) baadh ki photo me se ye pehchanna ki paani kitna faila hai, kaunsi sadak doob kar band ho chuki hai, aur kitne ghar paani me doob chuke hain.**

### The Problem It Solves:
Standard optical satellite imagery (10m GSD) me sadak par 1 foot paani bhara hai ya nahi, yeh nahi dikhta. FloodNet drone ke high-resolution (5cm–10cm GSD) data ka use karke microscopic road inundation aur residential submergence ko alag-alag 4 tactical disaster classes me divide karta hai.

---

## 2. Real-Life Use Cases (Asli Duniya me Kaha Use Hota Hai?)

FloodNet DeepLabV3+ ko NETRA-D platform me **Nadir Inundation & Structural Submergence Engine** banaya gaya hai.

### 1. Flash Flood & Urban Drainage Inundation Triage
- **Problem:** Kedarnath ya Dehradun me cloudburst ke baad shahar ki galiyon me paani bhar jata hai. District administration ko immediate priority list chahiye hoti hai ki kis mohalle me boat bhejni hai.
- **FloodNet Solution:** Flooded buildings aur flooded roads ka exact square-meter footprint calculate karta hai, jisse NDRF rescue boats seedha targeted localities me dispatch hoti hain.

### 2. Evacuation Road Submergence Detection
- **Problem:** Ground forces ko pata nahi hota ki aage ki sadak passable hai ya beh chuki hai.
- **FloodNet Solution:** Normal road aur flooded road ke beech ka difference identify karta hai, jisse road accessibility engine blockage alert raise karta hai.

### 3. Structural Submergence vs Intact Buildings
- **Problem:** Compensation aur casualty risk assess karne ke liye pata hona chahiye ki kitne ghar paani ke beech fase hain.
- **FloodNet Solution:** `flooded-building` class se isolated building footprints nikalta hai.

---

## 3. Advantages (Fayde)

- **Dedicated Tactical Disaster Subclasses:** Sirf generic water nahi, balki `flooded-road` aur `flooded-building` jaise operational subclasses deta hai.
- **Multi-Scale Feature Capture (ASPP):** Chhote paani ke puddles se lekar bade lake-level submergence ko multi-scale dilation rates se capture karta hai.
- **High-Resolution Precision:** 5cm–10cm sub-decimeter orthomosaics par sharp structural boundaries banata hai.
- **Deterministic Latency:** 26.7M parameter model hone ke bawajood standard CPU par ~650–850 ms me execute hota hai.

---

## 4. Disadvantages (Kamiyan / Limitations)

- **Zero Representation for Landslides / Mudflows:** Model ke training vocabulary me bare-soil landslide ka class nahi hai. Mountain mudslides ko yeh miss kar deta hai (isliye TransLandSeg ke sath chalaya jata hai).
- **Nadir-Perspective Only (Horizon Blindness):** Straight-down 90° nadir imagery par train hua hai. Oblique shots me blue sky ko floodwater samajh leta hai (isliye SegFormer B0 ke sath ensemble kiya gaya hai).

---

## 5. Assumptions (Kis Data / Conditions Pe Best Kaam Karta Hai?)

1. **Nadir UAV Perspectives:** Drone camera ka pitch angle $-90^\circ$ (vertically downwards) ya near-nadir ($\pm 15^\circ$) hona chahiye.
2. **Sub-Decimeter GSD:** Optimal ground resolution 2cm se 15cm per pixel.
3. **RGB Natural Color Spectrum:** Standard daylight optical imagery.
