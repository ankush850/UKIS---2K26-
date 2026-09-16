# Microsoft SiamUnet (xBD Pre/Post Damage) — Overview

## 1. Introduction (Model Kya Hai?)

**Microsoft SiamUnet** ek dual-branch Siamese Convolutional Neural Network hai jise disaster response me **paired pre-disaster (aapda se pehle) aur post-disaster (aapda ke baad) aerial imagery** ko compare karke buildings ke structural damage ko classify karne ke liye banaya gaya hai.

Yeh model international **xBD / FEMA HAZUS disaster damage standards** ka palan karta hai:
1. 🔴 **Destroyed:** Imarat puri tarah dharashayi / zameen-doz ho chuki hai.
2. 🟠 **Major Damage:** Badi dararein, chhat girna, ya partial structure collapse.
3. 🟡 **Minor Damage:** Halka nuksan, khidkiyan tootna, ya minor roof tile loss.
4. 🟢 **Intact:** Imarat bilkul surakshit hai.

Simple shabdon me:
> **SiamUnet ka kaam hai aapda se pehle aur baad ki do photos ko aapas me match karna aur ye batana ki kitne ghar 100% toot gaye, kitno me dararein hain, aur kitne bach gaye.**

---

## 2. Real-Life Use Cases (Asli Duniya me Kaha Use Hota Hai?)

Microsoft SiamUnet ko NETRA-D platform me **Structural Damage & Rehabilitation Assessment Engine** banaya gaya hai.

### 1. Post-Earthquake & Flood Building Damage Triage (DMMC / SDRF)
- **Problem:** Bhukamp (earthquake) ya flash flood ke baad hazaron ghar prabhavit hote hain. Ground survey teams har gali me turant nahi pahunch sakti.
- **SiamUnet Solution:** Drone se li gayi nayi photo ko purani baseline photo se match karke minutes ke andar destroyed aur major damage buildings ki GeoJSON list generate karta hai.

### 2. Emergency Relief & Evacuation Priority Directives
- **Problem:** Incident commanders ko decide karna hota hai ki search-and-rescue sniffer dogs aur heavy earth-movers kaha bhejein.
- **SiamUnet Solution:** "Destroyed" buildings ke coordinates aur density clusters provide karta hai jaha log malbe me dabe hone ka risk sabse high hota hai.

### 3. Transparent Government Compensation & Insurance Settlement
- **Problem:** Disaster relief funds distribution me bhrashtachar aur vivaad aam baat hoti hai.
- **SiamUnet Solution:** Pre/post structural change ka objective mathematical delta deta hai (e.g., "Building #142: Structural Integrity dropped from 100% to 12%"), jo Polygon blockchain par permanently record ho jata hai.

---

## 3. Advantages (Fayde)

- **International xBD Standard Compliance:** Global disaster agencies (FEMA, UN-SPIDER, HAZUS) dwara manyata prapt 4-tier damage classification.
- **Weight-Shared Siamese Architecture:** Pre aur post images ko ek hi shared feature space me map karta hai, jisse lighting ya seasonal shadow differences par false damage flags nahi bante.
- **Mathematical 100% Normalization:** Damaged aur intact percentages strictly **$100.0\%$ par mathematically normalize** hote hain (zero arithmetic drift).
- **Graceful Single-Image Fallback:** Agar pre-disaster reference image available nahi hai, toh system crash nahi hota; yeh single-image structural detection mode me shift ho jata hai.

---

## 4. Disadvantages (Kamiyan / Limitations)

- **Requires Co-Registration Alignment:** Pre aur post images ka geographic alignment (co-registration) accurate hona chahiye; agar 5-meter shift hua toh border pixels par false damage infer ho sakta hai.
- **Optical Resolution Dependency:** Chhote plaster cracks detect karne ke liye GSD $\le 15\text{ cm}$ hona zaroori hai.

---

## 5. Assumptions (Kis Data / Conditions Pe Best Kaam Karta Hai?)

1. **Paired Optical Imagery:** Pre-disaster baseline reference (Google Earth / Satellite / Earlier Sortie) + Post-disaster drone photo.
2. **Sub-Decimeter Resolution:** 5cm–20cm per pixel GSD.
3. **Aligned Coordinates:** Shared bounding box ya GPS alignment.
