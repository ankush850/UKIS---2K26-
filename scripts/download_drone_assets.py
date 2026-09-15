"""
Netra Aerial: Demo Asset Preparation & Download Script.
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.

Prepares local realistic disaster imagery, pre/post pairs, and pre-caches OSM road geometries
for instant offline hackathon demonstration.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DEMO_DIR = BASE_DIR / "data" / "aerial_demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "cache" / "roads"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "chamoli_flash_flood": {
        "title": "Chamoli Rishi Ganga Flash Flood",
        "bbox": [79.520, 30.460, 79.570, 30.510],
        "coords": [30.485, 79.545]
    },
    "joshimath_subsidence": {
        "title": "Joshimath Slope Subsidence",
        "bbox": [79.540, 30.535, 79.590, 30.585],
        "coords": [30.556, 79.566]
    },
    "kedarnath_inundation": {
        "title": "Kedarnath Valley Cloudburst",
        "bbox": [79.040, 30.710, 79.090, 30.760],
        "coords": [30.735, 79.066]
    },
    "rishikesh_bridge": {
        "title": "Rishikesh Ganges Floodplain",
        "bbox": [78.240, 30.060, 78.290, 30.110],
        "coords": [30.086, 78.267]
    }
}


def create_pre_post_pairs():
    print("[Assets] Generating realistic disaster pre/post demonstration pairs...")
    w, h = 512, 512

    for key, info in SCENARIOS.items():
        scenario_folder = DEMO_DIR / key
        scenario_folder.mkdir(parents=True, exist_ok=True)

        # ── 1. Pre-Disaster Imagery (Intact Baseline) ──
        pre_img = Image.new("RGB", (w, h), color=(90, 115, 80)) # Healthy vegetation
        d_pre = ImageDraw.Draw(pre_img)

        # Mountain slopes
        d_pre.polygon([(0, 0), (w, 0), (w, 150), (0, 100)], fill=(75, 95, 65))
        # Clear, tranquil mountain river
        d_pre.line([(0, 220), (180, 230), (340, 210), (w, 240)], fill=(70, 140, 180), width=35)
        # Intact bridge
        d_pre.rectangle([220, 205, 270, 245], fill=(160, 165, 170))
        # Main Road
        d_pre.line([(0, 310), (220, 225), (270, 225), (w, 200)], fill=(120, 125, 120), width=14)
        # Intact settlements / buildings
        for bx, by in [(110, 150), (150, 160), (320, 150), (360, 170), (280, 290), (330, 310)]:
            d_pre.rectangle([bx, by, bx + 28, by + 28], fill=(210, 205, 195))
            d_pre.polygon([(bx - 2, by), (bx + 14, by - 12), (bx + 30, by)], fill=(190, 70, 60)) # Intact red roof

        pre_path = scenario_folder / "pre_disaster.jpg"
        pre_img.save(pre_path, "JPEG", quality=90)

        # ── 2. Post-Disaster Imagery (Severe Event Impact) ──
        post_img = Image.new("RGB", (w, h), color=(85, 105, 75))
        d_post = ImageDraw.Draw(post_img)

        # Mountain slopes with rock scars
        d_post.polygon([(0, 0), (w, 0), (w, 150), (0, 100)], fill=(75, 95, 65))
        d_post.polygon([(120, 40), (160, 40), (200, 140), (140, 140)], fill=(130, 95, 55)) # Landslide chute

        # Violent flood surge - wide, turbid mudwater
        d_post.polygon([
            (0, 180), (180, 190), (340, 170), (w, 200),
            (w, 300), (340, 290), (180, 310), (0, 290)
        ], fill=(95, 125, 130)) # Silt-laden flood torrent

        # Washed out bridge section (gap in the center)
        d_post.rectangle([210, 205, 235, 245], fill=(140, 140, 140)) # Broken abutment
        d_post.rectangle([255, 205, 275, 245], fill=(140, 140, 140)) # Broken abutment

        # Submerged road sections
        d_post.line([(0, 310), (150, 255)], fill=(120, 125, 120), width=14)
        d_post.line([(310, 215), (w, 200)], fill=(120, 125, 120), width=14)

        # Debris fan & mudflow
        d_post.polygon([
            (170, 140), (290, 180), (260, 280), (180, 260)
        ], fill=(135, 95, 55))

        # Buildings: partially destroyed / swept away
        for bx, by in [(110, 150), (360, 170)]:
            # Intact remaining
            d_post.rectangle([bx, by, bx + 28, by + 28], fill=(210, 205, 195))
            d_post.polygon([(bx - 2, by), (bx + 14, by - 12), (bx + 30, by)], fill=(190, 70, 60))

        # Destroyed rubble piles
        for bx, by in [(150, 160), (320, 150), (280, 290)]:
            d_post.polygon([
                (bx, by + 20), (bx + 15, by - 5), (bx + 32, by + 10), (bx + 20, by + 28)
            ], fill=(115, 95, 80)) # Collapsed rubble

        post_path = scenario_folder / "post_disaster.jpg"
        post_img.save(post_path, "JPEG", quality=90)

        print(f"  -> Generated {key}: pre_disaster.jpg, post_disaster.jpg")


def precache_road_geometries():
    print("[Assets] Pre-caching Uttarakhand road geometries...")
    from backend.aerial.road_accessibility import road_classifier

    for key, info in SCENARIOS.items():
        bbox = info["bbox"]
        min_lon, min_lat, max_lon, max_lat = bbox
        cache_key = f"roads_{min_lon:.3f}_{min_lat:.3f}_{max_lon:.3f}_{max_lat:.3f}.json"
        cache_path = CACHE_DIR / cache_key

        if not cache_path.exists():
            roads = road_classifier._generate_resilient_uttarakhand_roads(bbox)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(roads, f, indent=2)
            print(f"  -> Cached road vectors for {key} ({len(roads)} road segments)")


if __name__ == "__main__":
    print("=========================================================")
    print("Netra Aerial: Demo Asset Initializer (Problem P-008)")
    print("=========================================================")
    create_pre_post_pairs()
    precache_road_geometries()
    print("=========================================================")
    print("All demo assets initialized successfully!")
    print("To download raw FloodNet dataset: https://github.com/BinaLab/FloodNet-Supervised_v1.0")
    print("To download xBD damage dataset: https://xview2.org/download")
    print("=========================================================")
