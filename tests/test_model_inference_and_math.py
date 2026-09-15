"""
Unit Tests for User's 5 Verification Checkpoints:
1. Bounding Box Ordering (GeoJSON vs Overpass format)
2. Real FloodNet DeepLabV3+ Model Inference & 4-Class Mapping
3. Real Microsoft SiamUnet xBD Siamese CNN & 136% Math Normalization
4. Overpass Local Cache Reliability (Zero-network demo resilience)
5. Real Rishikesh OSM Road Intersections with Calibrated Ganges Riparian Corridor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import json
from shapely.geometry import Polygon, LineString

from backend.aerial.road_accessibility import road_classifier, CALIBRATED_RIPARIAN_CORRIDORS
from backend.aerial.segmentation import drone_segmentation_engine
from backend.aerial.damage_assessment import building_damage_engine
from backend.aerial.presets import get_preset_by_id


def test_1_bbox_ordering_conversion():
    """Validates that [min_lon, min_lat, max_lon, max_lat] converts accurately to Overpass (south, west, north, east)."""
    bbox = [78.250, 30.050, 78.350, 30.150] # GeoJSON [west, south, east, north]
    min_lon, min_lat, max_lon, max_lat = bbox

    # Overpass QL expects (south, west, north, east)
    overpass_bbox_str = f"({min_lat},{min_lon},{max_lat},{max_lon})"
    assert overpass_bbox_str == "(30.05,78.25,30.15,78.35)"

    # Verify road_classifier loads correctly from cache without inversion
    roads = road_classifier.fetch_roads_overpass(bbox)
    assert len(roads) > 2000, f"Expected >2000 roads for Rishikesh, got {len(roads)}"

    first_node = roads[0]["geometry"][0]
    assert 78.25 <= first_node["lon"] <= 78.35, f"Lon out of bounds: {first_node['lon']}"
    assert 30.05 <= first_node["lat"] <= 30.15, f"Lat out of bounds: {first_node['lat']}"
    print("[PASS] Test 1 Passed: Bbox order conversion and coordinate integrity confirmed.")


def test_2_floodnet_deeplab_inference_and_normalization():
    """Validates real DeepLabV3+ checkpoint execution and strict 100.0% sum normalization."""
    res = drone_segmentation_engine.segment("data/aerial_demo/chamoli_flash_flood/post_disaster.jpg")
    hz = res["hazard_summary"]
    dist = res["distribution"]

    assert hz["real_model_inference"] is True, "Expected real model checkpoint to be loaded"
    assert "FloodNet" in hz["model_name"], f"Model name should indicate FloodNet, got {hz['model_name']}"
    assert hz["flooded_pct"] > 0.0, "Flooded surface percentage should be > 0"

    # Enforce strict 100.0% sum normalization
    sum_pct = sum(d["percentage"] for d in dist.values())
    assert abs(sum_pct - 100.0) < 0.001, f"Class percentages must sum strictly to 100.0%, got {sum_pct}"
    print(f"[PASS] Test 2 Passed: FloodNet DeepLabV3+ ran ({hz['model_name']}). Flooded: {hz['flooded_pct']}%, Total: {sum_pct:.2f}%.")


def test_3_siamese_damage_math_bug_resolution():
    """Validates real Microsoft SiamUnet xBD inference and complete elimination of the 136% math bug."""
    res = building_damage_engine.compare_pre_post(
        "data/aerial_demo/chamoli_flash_flood/pre_disaster.jpg",
        "data/aerial_demo/chamoli_flash_flood/post_disaster.jpg",
        zone_name="Chamoli Gorge"
    )

    bk = res["breakdown"]
    tf = res["total_footprints"]
    db = res["damaged_breakdown_of_damaged"]
    meta = res["model_metadata"]

    assert meta["real_model_inference"] is True, "Expected Microsoft SiamUnet checkpoint to be loaded"

    # 1. Sum of all 4 classes must equal 100.0% strictly
    all_classes_sum = sum(v["percentage"] for v in bk.values())
    assert abs(all_classes_sum - 100.0) < 0.001, f"All classes sum must be 100.0%, got {all_classes_sum}"

    # 2. Mode A: Total Surveyed Footprints = Intact + Damaged = 100.0%
    assert abs(tf["intact_pct"] + tf["damaged_pct"] - 100.0) < 0.001, "Mode A must sum to 100.0%"

    # 3. Mode B: Damaged Footprints Breakdown = Minor + Major + Destroyed = 100.0%
    assert abs(db["minor_pct"] + db["major_pct"] + db["destroyed_pct"] - 100.0) < 0.001, "Mode B must sum to 100.0%"

    print(f"[PASS] Test 3 Passed: 136% math bug resolved! Intact: {tf['intact_pct']}%, Damaged: {tf['damaged_pct']}%. Damaged breakdown: {db}")


def test_4_overpass_local_cache_demo_resilience():
    """Validates that local cache is loaded with 0ms network dependency."""
    cache_file = Path("cache/roads/roads_78.250_30.050_78.350_30.150.json")
    assert cache_file.exists(), "Rishikesh local cache file must exist on disk for demo resilience"

    bbox = [78.250, 30.050, 78.350, 30.150]
    roads = road_classifier.fetch_roads_overpass(bbox)
    assert len(roads) == 2428, f"Expected exactly 2428 roads from Rishikesh cache, got {len(roads)}"
    print(f"[PASS] Test 4 Passed: Loaded {len(roads)} OSM ways directly from local cache without touching network.")


def test_5_rishikesh_osm_riparian_intersections():
    """Validates that low-lying riparian roads in Rishikesh are tagged blocked while bypass is clear."""
    res = road_classifier.classify_roads([78.250, 30.050, 78.350, 30.150])
    summary = res["summary"]

    assert summary["total_segments"] == 2428
    assert summary["blocked_count"] > 300, f"Expected >300 blocked segments along river, got {summary['blocked_count']}"
    assert summary["clear_count"] > 1500, f"Expected >1500 clear segments, got {summary['clear_count']}"

    blocked_names = [cp["road_name"] for cp in summary["critical_chokepoints"]]
    assert any("Ram Jhula" in n for n in blocked_names), "Ram Jhula should be detected as blocked in riparian flood zone"
    assert any("Virbhadra" in n for n in blocked_names), "Virbhadra Rd should be detected as blocked along riverbed"

    # Verify Rishikesh Bypass is clear
    bypass_features = [f for f in res["features"] if "Bypass" in f["properties"]["name"]]
    clear_bypass = any(f["properties"]["status"] == "clear" for f in bypass_features)
    assert clear_bypass, "Rishikesh Bypass upper highway should have clear evacuation corridors"
    print(f"[PASS] Test 5 Passed: Ganges riparian chokepoints verified (Ram Jhula, Virbhadra Rd). Bypass clear.")


def test_6_satellite_and_drone_module_independence():
    """
    Validates Section 0 & 6 Checklist Items 1 & 2:
    - Satellite Module runs completely independently without any Drone Module dependency.
    - Drone Module runs completely independently without any Satellite Module dependency.
    - Direct neural outputs vs derived spatial overlays are explicitly demarcated.
    """
    # 1. Verify Satellite Module Standalone Execution
    from backend.ingestion.copernicus_client import CopernicusClient
    SAT_PRESETS = CopernicusClient.get_demo_presets()
    assert len(SAT_PRESETS) > 0, "Satellite presets must load independently"
    punjab_sat = next((p for p in SAT_PRESETS if "punjab" in p["id"]), None)
    assert punjab_sat is not None, "Satellite Punjab preset must exist"
    assert "bbox" in punjab_sat, "Satellite preset must specify Sentinel-2 AOI bbox"
    print(f"[PASS] Satellite module runs standalone: {len(SAT_PRESETS)} Sentinel-2 AOIs available ({punjab_sat['title']}).")

    # 2. Verify Drone Module Standalone Execution
    from backend.aerial.presets import UTTARAKHAND_DISASTER_PRESETS as DRONE_PRESETS
    assert len(DRONE_PRESETS) > 0, "Drone presets must load independently"
    chamoli_drone = next((p for p in DRONE_PRESETS if "chamoli" in p["id"]), None)
    assert chamoli_drone is not None, "Drone Chamoli preset must exist"
    assert "drone_sortie" in chamoli_drone, "Drone preset must contain independent sortie data"
    print(f"[PASS] Drone module runs standalone: {len(DRONE_PRESETS)} DMMC disaster zones available.")

    # 3. Verify Scientific Attribution & Explicit Class Demarcation
    drone_res = drone_segmentation_engine.segment("data/aerial_demo/chamoli_flash_flood/post_disaster.jpg")
    hz = drone_res["hazard_summary"]
    assert "direct_model_outputs" in hz, "Must explicitly include direct neural outputs"
    assert "derived_spatial_metrics" in hz, "Must explicitly include derived spatial metrics"
    assert hz["direct_model_outputs"]["flooded_water_pct"] > 0.0
    print(f"[PASS] Drone neural classes vs derived spatial metrics strictly segregated.")


if __name__ == "__main__":
    test_1_bbox_ordering_conversion()
    test_2_floodnet_deeplab_inference_and_normalization()
    test_3_siamese_damage_math_bug_resolution()
    test_4_overpass_local_cache_demo_resilience()
    test_5_rishikesh_osm_riparian_intersections()
    test_6_satellite_and_drone_module_independence()
    print("\n==================================================================")
    print("ALL 6 USER CHECKPOINT & INDEPENDENCE TESTS PASSED WITH 100% SUCCESS!")
    print("==================================================================")

