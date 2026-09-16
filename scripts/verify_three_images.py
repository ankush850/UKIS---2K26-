"""
Verification Script for the 3 Test Images:
1. Meppadi Landslide (Real landslide photo)
2. Sydney Houses (Normal non-disaster hillside houses with sky horizon)
3. Mountains Landslide (Real mountain landslide photo)

Extracts and tabulates:
- FloodNet Inundated %
- FloodNet Debris %
- TransLandSeg Landslide Scar %
- SegFormer Water % (Confidence)
- SegFormer Sky % (Confidence)
- Primary Hazard Routing
- Dominant Model Engine & Discrepancy Notes
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.aerial.custom_inspection import run_full_custom_inspection

test_images = [
    {
        "name": "Meppadi Landslide (Chooralmala)",
        "path": "data/real/meppadi-india-an-aerial-view-shows-the-site-of-a-landslide-on-july-31-2024-in-chooralmala (1).webp",
        "description": "Real massive landslide event in Kerala"
    },
    {
        "name": "Sydney Beach Houses",
        "path": "data/real/houses-on-the-hill-beautiful-suburb-of-sydney-palm-beach-background-with-copy-space.webp",
        "description": "Normal non-disaster hillside town with houses, pools, blue sky horizon"
    },
    {
        "name": "Mountains Landslide",
        "path": "data/real/mountains-landslides-due-heavy-rain-260nw-1822899695.webp",
        "description": "Real mountain mudslide/landslide scar cutting through forest"
    }
]

print("=" * 105)
print("RUNNING FULL CUSTOM INSPECTION TRI-MODEL VERIFICATION (FLOODNET + TRANSLANDSEG + SEGFORMER)")
print("=" * 105)

results = []
for item in test_images:
    img_path = WORKSPACE_ROOT / item["path"]
    print(f"\n[Processing] {item['name']}...")
    if not img_path.exists():
        print(f"  ERROR: Image not found at {img_path}")
        continue

    res = run_full_custom_inspection(
        post_image_input=str(img_path),
        zone_name=item["name"],
        disaster_mode="auto"
    )

    fnet = res.get("segmentation", {})
    hz = fnet.get("hazard_summary", {})
    tls = res.get("landslide_segmentation", {})
    sf = res.get("segformer_water", {})
    routing = res.get("routing", {})
    sev = res.get("severity", {})

    info = {
        "name": item["name"],
        "description": item["description"],
        "fnet_inundated_pct": hz.get("flooded_pct", 0.0),
        "fnet_debris_pct": hz.get("debris_pct", 0.0),
        "tls_landslide_pct": tls.get("landslide_pct", 0.0),
        "tls_confidence_pct": round(tls.get("confidence", 0.0) * 100.0, 1),
        "sf_water_pct": sf.get("flood_water_pct", 0.0),
        "sf_water_conf": round(sf.get("water_confidence", 0.0) * 100.0, 1),
        "sf_sky_pct": sf.get("sky_pct", 0.0),
        "sf_sky_conf": round(sf.get("sky_confidence", 0.0) * 100.0, 1),
        "models_disagree": routing.get("models_disagree", False),
        "disagreement_note": routing.get("disagreement_note"),
        "primary_hazard": routing.get("primary_hazard"),
        "primary_hazard_label": routing.get("primary_hazard_label"),
        "dominant_model": routing.get("dominant_model"),
        "routing_synthesis": routing.get("synthesis"),
        "severity_score": sev.get("severity_score"),
        "severity_level": sev.get("level")
    }
    results.append(info)

    print(f"  -> FloodNet: Inundated={info['fnet_inundated_pct']}%, Debris={info['fnet_debris_pct']}%")
    print(f"  -> TransLandSeg: Landslide Scar={info['tls_landslide_pct']}% (Conf: {info['tls_confidence_pct']}%)")
    print(f"  -> SegFormer ADE20K: Water={info['sf_water_pct']}% ({info['sf_water_conf']}% conf) | Sky={info['sf_sky_pct']}% ({info['sf_sky_conf']}% conf)")
    if info['models_disagree']:
        print(f"  -> ⚠️ DISCREPANCY FLAGGED: {info['disagreement_note']}")
    print(f"  -> Routing Decision: {info['primary_hazard_label']} via {info['dominant_model']}")
    print(f"  -> Severity: {info['severity_score']}/100 ({info['severity_level']})")

print("\n" + "=" * 105)
print("SUMMARY COMPARISON TABLE")
print("=" * 105)
header = f"{'Test Image':<32} | {'FloodNet Water':<14} | {'SegFormer Water':<15} | {'SegFormer Sky':<13} | {'Landslide Scar':<14} | {'Routing Decision'}"
print(header)
print("-" * len(header))
for r in results:
    print(
        f"{r['name']:<32} | "
        f"{r['fnet_inundated_pct']:>13.2f}% | "
        f"{r['sf_water_pct']:>14.2f}% | "
        f"{r['sf_sky_pct']:>12.2f}% | "
        f"{r['tls_landslide_pct']:>13.2f}% | "
        f"{r['primary_hazard_label']}"
    )
print("=" * 105)

# Save JSON result
out_json = WORKSPACE_ROOT / "scratch" / "tri_model_verification_results.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"Detailed verification saved to {out_json}")
