"""
Comprehensive Unit & Integration Test Suite for Netra Aerial Pipeline.
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.aerial.segmentation import drone_segmentation_engine, AERIAL_CLASSES
from backend.aerial.damage_assessment import building_damage_engine, DAMAGE_CLASSES
from backend.aerial.road_accessibility import road_classifier
from backend.aerial.severity import severity_scorer
from backend.aerial.fusion import fusion_engine
from backend.aerial.weather import get_weather_forecast
from backend.aerial.report_generator import generate_dmmc_report_html
from backend.aerial.presets import UTTARAKHAND_DISASTER_PRESETS


def test_drone_segmentation():
    print("[TEST 1] Testing Drone Multi-Class Segmentation Pipeline...")
    # Create test aerial frame (320x320)
    img_np = np.zeros((320, 320, 3), dtype=np.uint8)
    img_np[:, :] = [90, 120, 70] # Vegetation
    img_np[100:180, :] = [40, 110, 160] # Water flood
    img_np[190:240, 50:180] = [130, 95, 60] # Mud debris

    res = drone_segmentation_engine.segment(img_np, gsd_m=0.10)
    assert res["status"] == "success", "Segmentation failed"
    assert "distribution" in res, "Missing class distribution"
    assert "hazard_summary" in res, "Missing hazard summary"
    assert "segmentation_mask_b64" in res, "Missing base64 mask"
    assert res["total_area_m2"] > 0, "Total area should be greater than 0"
    
    # Verify all 7 classes are accounted for
    for c_id, info in AERIAL_CLASSES.items():
        assert info["name"] in res["distribution"], f"Class {info['name']} missing from distribution"

    print(f"  -> Success! Total Area: {res['total_area_m2']} m², Flooded: {res['hazard_summary']['flooded_pct']}%")


def test_siamese_damage_assessment():
    print("[TEST 2] Testing Siamese Pre/Post Building Damage Assessment...")
    pre_img = np.ones((256, 256, 3), dtype=np.uint8) * 120
    post_img = np.ones((256, 256, 3), dtype=np.uint8) * 120
    # Simulate disaster impact in post image
    post_img[50:150, 50:150] = [200, 70, 50]

    res = building_damage_engine.compare_pre_post(pre_img, post_img, zone_name="Chamoli Sector")
    assert res["status"] == "success", "Comparison failed"
    assert "structural_integrity_pct" in res, "Missing structural integrity score"
    assert "breakdown" in res, "Missing xBD damage breakdown"
    assert "damage_overlay_b64" in res, "Missing damage overlay"
    assert 0.0 <= res["structural_integrity_pct"] <= 100.0, "Integrity out of bounds"

    print(f"  -> Success! Structural Integrity: {res['structural_integrity_pct']}%, Status: {res['overall_status']}")


def test_road_accessibility_classification():
    print("[TEST 3] Testing Rule-Based Road Accessibility Classifier...")
    bbox = [79.520, 30.460, 79.570, 30.510]
    
    # Create test flood mask
    flood_mask = np.zeros((256, 256), dtype=np.uint8)
    flood_mask[100:180, :] = 1 # Submerge center corridor

    res = road_classifier.classify_roads(bbox=bbox, flood_mask=flood_mask, threshold_pct=15.0)
    assert res["type"] == "FeatureCollection", "Not a valid GeoJSON FeatureCollection"
    assert len(res["features"]) > 0, "No road features returned"
    assert "summary" in res, "Missing summary"

    has_blocked = any(f["properties"]["status"] == "blocked" for f in res["features"])
    has_clear = any(f["properties"]["status"] == "clear" for f in res["features"])
    assert has_blocked, "Should identify blocked road segments from flood mask"

    print(f"  -> Success! Total Segments: {res['summary']['total_segments']}, Blocked: {res['summary']['blocked_count']}")


def test_severity_scoring():
    print("[TEST 4] Testing Severity Scoring & Prioritization Formula...")
    high_dmg = {
        "building_destroyed": 35.0,
        "major_damage": 25.0,
        "flooded": 40.0,
        "debris": 20.0,
        "minor_damage": 10.0
    }
    sev_high = severity_scorer.calculate_severity(
        affected_area_pct=65.0,
        damage_distribution=high_dmg,
        confidence_score=0.92,
        blocked_roads_count=2
    )
    assert sev_high["level"] == "HIGH", f"Expected HIGH, got {sev_high['level']}"
    assert sev_high["severity_score"] >= 70, "Score should be >= 70"
    assert sev_high["action_code"] == "RED-ALPHA", "Action code mismatch"

    low_dmg = {
        "building_destroyed": 0.0,
        "major_damage": 2.0,
        "flooded": 5.0,
        "debris": 2.0,
        "minor_damage": 5.0
    }
    sev_low = severity_scorer.calculate_severity(
        affected_area_pct=8.0,
        damage_distribution=low_dmg,
        confidence_score=0.80,
        blocked_roads_count=0
    )
    assert sev_low["level"] == "LOW", f"Expected LOW, got {sev_low['level']}"
    assert sev_low["severity_score"] < 40, "Score should be < 40"

    print(f"  -> Success! High Severity: {sev_high['severity_score']}/100, Low Severity: {sev_low['severity_score']}/100")


def test_satellite_drone_fusion():
    print("[TEST 5] Testing Satellite-Drone Fusion & Confidence Delta Engine...")
    sat_zones = [
        {
            "id": "chamoli_test",
            "name": "Chamoli Sector",
            "district": "Chamoli",
            "bbox": [79.520, 30.460, 79.570, 30.510],
            "satellite_confidence": 0.65,
            "flag_reason": "NDWI flood surge anomaly",
            "severity": {"severity_score": 85, "level": "HIGH"}
        },
        {
            "id": "uninspected_zone",
            "name": "Remote Hill Zone",
            "district": "Uttarkashi",
            "bbox": [78.400, 30.700, 78.450, 30.750],
            "satellite_confidence": 0.58,
            "flag_reason": "Spectral cloud haze",
            "severity": {"severity_score": 48, "level": "MEDIUM"}
        }
    ]
    drone_sorties = [
        {
            "id": "DRONE-01",
            "bbox": [79.525, 30.465, 79.565, 30.505], # High IoU overlap with chamoli_test
            "drone_confidence": 0.94,
            "timestamp": "Just now",
            "hazard_summary": {"flooded_pct": 30.0},
            "severity": {"severity_score": 88, "level": "HIGH"},
            "blocked_roads": []
        }
    ]

    fused = fusion_engine.fuse_zones(sat_zones, drone_sorties)
    assert len(fused) == 2, "Should return 2 fused zone items"
    
    confirmed = next(z for z in fused if z["id"] == "chamoli_test")
    assert confirmed["status"] == "fused_confirmed", "Chamoli should be fused_confirmed"
    assert confirmed["fusion_metrics"]["confidence_delta_pct"] == 29.0, "Delta should be +29%"

    uninspected = next(z for z in fused if z["id"] == "uninspected_zone")
    assert uninspected["status"] == "satellite_only", "Remote zone should be satellite_only"
    assert "drone_flight_recommendation" in uninspected, "Should provide drone sortie recommendation"

    print(f"  -> Success! Confirmed Delta: {confirmed['fusion_metrics']['confidence_delta_text']}")
    print(f"  -> Success! Satellite-Only status: {uninspected['status_label']}")


def test_weather_and_flight_clearance():
    print("[TEST 6] Testing Open-Meteo Weather & Flight Clearance...")
    w = get_weather_forecast(lat=30.485, lon=79.545)
    assert "temperature_c" in w, "Missing temperature"
    assert "wind_speed_kmh" in w, "Missing wind speed"
    assert "flight_safety" in w, "Missing flight safety"
    assert "safe_to_fly" in w["flight_safety"], "Missing safe_to_fly boolean"

    print(f"  -> Success! Temp: {w['temperature_c']}°C, Wind: {w['wind_speed_kmh']} km/h, Clearance: {w['flight_safety']['badge']}")


def test_dmmc_report_generator():
    print("[TEST 7] Testing DMMC Field Incident Action Report HTML Generator...")
    zone = UTTARAKHAND_DISASTER_PRESETS[0]
    zone_data = {
        "id": zone["id"],
        "name": zone["title"],
        "district": zone["district"],
        "center": zone["coords"],
        "satellite_data": {"confidence_pct": 68.0},
        "drone_data": zone.get("drone_sortie", {}),
        "fusion_metrics": {"confidence_delta_text": "+26% Fidelity Gain"},
        "severity": zone.get("drone_sortie", {}).get("severity", {}),
        "blocked_roads": zone.get("drone_sortie", {}).get("blocked_roads", []),
        "pre_image_b64": "",
        "post_image_b64": ""
    }
    w = get_weather_forecast(lat=zone["coords"][0], lon=zone["coords"][1])
    html = generate_dmmc_report_html(zone_data, w)
    assert "<!DOCTYPE html>" in html, "Invalid HTML"
    assert "DMMC INCIDENT ACTION DIRECTIVE" in html, "Missing DMMC header"
    assert "Chamoli" in html, "Missing zone name"
    assert "@media print" in html, "Missing printable styles"

    print(f"  -> Success! Generated report HTML ({len(html)} bytes)")


if __name__ == "__main__":
    print("==================================================================")
    print("RUNNING FULL NETRA AERIAL TEST SUITE (Problem P-008)")
    print("==================================================================")
    test_drone_segmentation()
    test_siamese_damage_assessment()
    test_road_accessibility_classification()
    test_severity_scoring()
    test_satellite_drone_fusion()
    test_weather_and_flight_clearance()
    test_dmmc_report_generator()
    print("==================================================================")
    print("ALL 7 NETRA AERIAL TESTS PASSED WITH 100% SUCCESS!")
    print("==================================================================")
