"""
Fetch Genuine OpenStreetMap Road Networks from Overpass API.
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.
"""

import requests
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache" / "roads"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AOIS = {
    "rishikesh": {
        "bbox": [78.25, 30.05, 78.35, 30.15], # min_lon, min_lat, max_lon, max_lat
        "overpass_bbox": "30.05,78.25,30.15,78.35", # south, west, north, east
        "title": "Rishikesh NH-7 / Badrinath Road Corridor"
    },
    "chamoli": {
        "bbox": [79.52, 30.46, 79.57, 30.51],
        "overpass_bbox": "30.46,79.52,30.51,79.57",
        "title": "Chamoli Rishi Ganga / Tapovan Gorge"
    },
    "joshimath": {
        "bbox": [79.54, 30.535, 79.59, 30.585],
        "overpass_bbox": "30.535,79.54,30.585,79.59",
        "title": "Joshimath Town & Auli Bypass"
    },
    "kedarnath": {
        "bbox": [79.04, 30.71, 79.09, 30.76],
        "overpass_bbox": "30.71,79.04,30.76,79.09",
        "title": "Kedarnath Mandakini Pilgrim Corridor"
    }
}

def fetch_aoi_roads(name, aoi):
    print(f"\n[Overpass API] Fetching real OSM road geometry for {aoi['title']}...")
    query = f"""[out:json][timeout:25];
way["highway"]({aoi['overpass_bbox']});
out geom;"""
    url = "https://overpass-api.de/api/interpreter"
    headers = {
        "User-Agent": "NetraAerial-DMMC-DisasterMonitoring/2.0 (Uttarakhand Government Disaster Response)"
    }
    try:
        resp = requests.post(url, data={"data": query}, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            elements = data.get("elements", [])
            print(f"  -> SUCCESS! Fetched {len(elements)} genuine road ways from OpenStreetMap.")
            
            # Save raw OSM cache
            min_lon, min_lat, max_lon, max_lat = aoi["bbox"]
            cache_file = CACHE_DIR / f"roads_{min_lon:.3f}_{min_lat:.3f}_{max_lon:.3f}_{max_lat:.3f}.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(elements, f, indent=2)
            print(f"  -> Saved to: {cache_file.name}")
            
            # Print road names found
            named_roads = [e.get("tags", {}).get("name") for e in elements if "name" in e.get("tags", {})]
            unique_names = list(dict.fromkeys(named_roads))
            print(f"  -> Real Road Names ({len(unique_names)}): {unique_names[:8]}")
            return elements
        else:
            print(f"  -> HTTP Error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  -> Network request failed: {e}")
    return None

if __name__ == "__main__":
    print("==================================================================")
    print("Fetching Real OpenStreetMap Road Networks for Uttarakhand AOIs")
    print("==================================================================")
    for name, aoi in AOIS.items():
        fetch_aoi_roads(name, aoi)
    print("\n==================================================================")
    print("OSM Road Fetch Complete!")
    print("==================================================================")
