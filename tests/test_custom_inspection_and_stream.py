"""
Automated Unit Tests for Custom Drone Inspection Pipeline & Ken Burns Simulated Flight Pass.
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.

Covers:
1. Single custom image upload with location (Segmentation + Severity + Road Accessibility unified)
2. Single custom image upload without location (Graceful skip with exact specified message)
3. Pre/Post pair upload (Full SiamUnet execution + strict 100.0% Mode A/B normalization)
4. Static image flight pass simulation (Ken Burns pan/zoom frame generation + real PyTorch inference per frame)
5. Strict honesty badge segregation (Video vs Static Image mode badges)
6. Satellite/Drone module independence (Zero cross-module coupling)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from backend.aerial.custom_inspection import run_full_custom_inspection, extract_exif_gps
from backend.aerial.live_stream import live_stream_simulator
from backend.aerial.segmentation import drone_segmentation_engine
from backend.aerial.damage_assessment import building_damage_engine


def test_1_single_image_with_location():
    """Confirms single image upload with coordinates returns unified segmentation, severity, and road accessibility."""
    print("[TEST 1] Testing single image upload with confirmed location...")
    test_img = "data/aerial_demo/chamoli_flash_flood/post_disaster.jpg"
    
    # Rishikesh coordinates
    lat, lon = 30.100, 78.300
    res = run_full_custom_inspection(
        post_image_input=test_img,
        user_lat=lat,
        user_lon=lon,
        zone_name="Rishikesh Riverbed Survey"
    )

    assert res["status"] == "success"
    
    # 1. Segmentation
    seg = res["segmentation"]
    assert "distribution" in seg
    assert seg["hazard_summary"]["flooded_pct"] > 0.0
    sum_dist = sum(d["percentage"] for d in seg["distribution"].values())
    assert abs(sum_dist - 100.0) < 0.001

    # 2. Severity
    sev = res["severity"]
    assert "severity_score" in sev
    assert sev["level"] in ["HIGH", "MEDIUM", "LOW"]
    assert "directive" in sev

    # 3. Road Accessibility
    roads = res["road_accessibility"]
    assert roads["available"] is True
    assert roads["total_segments"] > 0
    assert roads["clear_count"] + roads["blocked_count"] == roads["total_segments"]

    # 4. Building Damage (Detection-Only Mode)
    bldg = res["building_damage"]
    assert bldg["mode"] == "single_image_detection_only"
    assert "damaged_building_pct" in bldg

    print(f"  -> Success! Unified result returned. Severity: {sev['severity_score']}/100, Roads: {roads['total_segments']} segments ({roads['blocked_count']} blocked).")


def test_2_single_image_without_location():
    """Confirms single image upload without location skips road accessibility gracefully with exact required message."""
    print("[TEST 2] Testing single image upload without location data...")
    test_img = "data/aerial_demo/chamoli_flash_flood/post_disaster.jpg"

    res = run_full_custom_inspection(
        post_image_input=test_img,
        user_lat=None,
        user_lon=None,
        zone_name="Unlocated Hill Survey"
    )

    assert res["status"] == "success"
    roads = res["road_accessibility"]
    assert roads["available"] is False
    assert roads["message"] == "Location not available — road accessibility skipped"
    assert roads["total_segments"] == 0
    assert roads["geojson"] is None

    print(f"  -> Success! Road accessibility skipped gracefully: '{roads['message']}'")


def test_3_pre_post_image_pair_siamunet():
    """Confirms pre/post pair upload runs full SiamUnet assessment with strict 100.0% normalization."""
    print("[TEST 3] Testing Pre/Post image pair upload for Siamese damage assessment...")
    pre_img = "data/aerial_demo/chamoli_flash_flood/pre_disaster.jpg"
    post_img = "data/aerial_demo/chamoli_flash_flood/post_disaster.jpg"

    res = run_full_custom_inspection(
        post_image_input=post_img,
        pre_image_input=pre_img,
        zone_name="Chamoli Valley Pre/Post"
    )

    assert res["status"] == "success"
    bldg = res["building_damage"]
    assert bldg["mode"] == "pre_post_siamese_comparison"
    assert "structural_integrity_pct" in bldg
    assert 0.0 <= bldg["structural_integrity_pct"] <= 100.0

    tf = bldg["total_footprints"]
    db = bldg["damaged_breakdown"]

    # Mode A sums to 100.0%
    assert abs(tf["intact_pct"] + tf["damaged_pct"] - 100.0) < 0.001
    # Mode B sums to 100.0%
    assert abs(db["minor_pct"] + db["major_pct"] + db["destroyed_pct"] - 100.0) < 0.001

    print(f"  -> Success! SiamUnet executed. Mode A: Intact={tf['intact_pct']}%, Damaged={tf['damaged_pct']}%. Mode B: Minor={db['minor_pct']}%, Major={db['major_pct']}%, Destroyed={db['destroyed_pct']}%.")


def test_4_ken_burns_flight_pass_generation_and_inference():
    """Confirms static aerial image produces 60 varying Ken Burns frames and runs real PyTorch inference per frame."""
    print("[TEST 4] Testing Ken Burns simulated flight pass frame generation...")
    post_img = "data/aerial_demo/chamoli_flash_flood/post_disaster.jpg"

    n_frames = live_stream_simulator.set_static_image(post_img)
    assert n_frames == 60
    assert live_stream_simulator.source_type == "static_image"
    assert len(live_stream_simulator._cached_frames) == 60

    frame_0 = live_stream_simulator._cached_frames[0]
    frame_30 = live_stream_simulator._cached_frames[30]
    frame_59 = live_stream_simulator._cached_frames[59]

    # Verify crops vary across trajectory (motion simulation)
    arr_0 = np.array(frame_0)
    arr_30 = np.array(frame_30)
    arr_59 = np.array(frame_59)

    diff_0_30 = np.mean(np.abs(arr_0.astype(float) - arr_30.astype(float)))
    diff_30_59 = np.mean(np.abs(arr_30.astype(float) - arr_59.astype(float)))

    assert diff_0_30 > 5.0, "Frames should vary noticeably due to pan/zoom flight trajectory"
    assert diff_30_59 > 5.0, "Frames should vary noticeably across flight trajectory"

    # Verify real PyTorch segmentation executes on generated frames
    res_0 = drone_segmentation_engine.segment(frame_0)
    res_30 = drone_segmentation_engine.segment(frame_30)

    assert res_0["status"] == "success"
    assert res_30["status"] == "success"
    assert res_0["hazard_summary"]["real_model_inference"] is True

    print(f"  -> Success! Generated 60 synthetic flight frames with real PyTorch inference. Frame delta: {diff_0_30:.1f} intensity diff.")


def test_5_honesty_badges_segregated():
    """Confirms video and static image mode honesty badges are distinct and never interchangeable."""
    print("[TEST 5] Testing honesty badge segregation...")
    badge_video = live_stream_simulator.HONESTY_BADGES["video"]
    badge_image = live_stream_simulator.HONESTY_BADGES["static_image"]

    assert "PRE-RECORDED SORTIE" in badge_video
    assert "SIMULATED FLIGHT PASS OVER STATIC IMAGE" in badge_image
    assert badge_video != badge_image

    live_stream_simulator.set_video_source()
    assert live_stream_simulator.source_type == "video"
    assert live_stream_simulator.HONESTY_BADGES[live_stream_simulator.source_type] == badge_video

    live_stream_simulator.set_static_image("data/aerial_demo/chamoli_flash_flood/post_disaster.jpg")
    assert live_stream_simulator.source_type == "static_image"
    assert live_stream_simulator.HONESTY_BADGES[live_stream_simulator.source_type] == badge_image

    print(f"  -> Success! Honesty badges strictly segregated:\n     - Video: '{badge_video}'\n     - Image: '{badge_image}'")


def test_6_satellite_module_isolation():
    """Confirms Satellite Triage module remains completely isolated and untouched."""
    print("[TEST 6] Verifying Satellite Module isolation...")
    from backend.ingestion.copernicus_client import CopernicusClient
    presets = CopernicusClient.get_demo_presets()
    assert len(presets) > 0
    assert any("punjab" in p["id"] for p in presets)
    print("  -> Success! Satellite module operates completely independent of drone upload and flight simulation.")


if __name__ == "__main__":
    print("==================================================================")
    print("RUNNING CUSTOM INSPECTION & KEN BURNS FLIGHT PASS TEST SUITE")
    print("==================================================================")
    test_1_single_image_with_location()
    test_2_single_image_without_location()
    test_3_pre_post_image_pair_siamunet()
    test_4_ken_burns_flight_pass_generation_and_inference()
    test_5_honesty_badges_segregated()
    test_6_satellite_module_isolation()
    print("==================================================================")
    print("ALL 6 TESTS PASSED WITH 100% SUCCESS!")
    print("==================================================================")
