"""
Unit & Integration Tests for SegFormer ADE20K Water/Flood & Sky Segmentation.
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import pytest

from backend.aerial.water_detection_segformer import SegFormerWaterDetector, segformer_water_detector
from backend.aerial.custom_inspection import run_full_custom_inspection


def test_segformer_label_mapping_and_sanity_assertion():
    """Confirms ADE20K classes are mapped properly and sky is never in water class IDs."""
    detector = segformer_water_detector
    assert detector.model_loaded is True
    assert detector.sky_class_id == 2
    assert "water" in detector.water_class_ids
    assert "sea" in detector.water_class_ids
    assert "river" in detector.water_class_ids
    assert "lake" in detector.water_class_ids

    # Mandatory sanity check: Sky must NEVER be counted as water
    assert detector.sky_class_id not in detector.flood_water_id_set, "Sky class must never be counted as water"
    assert "sky" not in [name.lower() for name in detector.water_class_ids.keys()]
    print(f"SegFormer Label Mapping verified: Sky={detector.sky_class_id}, Water={detector.water_class_ids}")


def test_segformer_inference_synthetic():
    """Tests forward inference of SegFormer on synthetic image with genuine confidence scores."""
    detector = segformer_water_detector
    # Synthetic blue upper half (sky-like) and blue-green lower half
    test_arr = np.zeros((128, 128, 3), dtype=np.uint8)
    test_arr[:64, :] = [135, 206, 235]  # Sky blue
    test_arr[64:, :] = [30, 144, 255]   # Deep water blue

    res = detector.detect_water(test_arr, gsd_m=0.10)
    assert res["status"] == "success"
    assert "flood_water_pct" in res
    assert "sky_pct" in res
    assert "water_confidence" in res
    assert "sky_confidence" in res
    assert "segmentation_mask_b64" in res
    assert res["segmentation_mask_b64"].startswith("data:image/png;base64,")
    assert isinstance(res["water_confidence"], float)
    assert isinstance(res["sky_confidence"], float)
    assert 0.0 <= res["water_confidence"] <= 1.0
    assert 0.0 <= res["sky_confidence"] <= 1.0


def test_sydney_houses_sky_vs_water_disambiguation():
    """Confirms SegFormer classifies upper region as sky and produces near-zero water on Sydney houses."""
    img_path = Path("data/real/houses-on-the-hill-beautiful-suburb-of-sydney-palm-beach-background-with-copy-space.webp")
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")

    res = segformer_water_detector.detect_water(str(img_path))
    print(f"\nSydney Houses -> Sky: {res['sky_pct']}% (Conf: {res['sky_confidence']}), Water: {res['flood_water_pct']}% (Conf: {res['water_confidence']})")

    # High sky presence (> 40%)
    assert res["sky_pct"] > 40.0, f"Expected >40% sky, got {res['sky_pct']}%"
    assert res["sky_confidence"] > 0.80, f"Expected high sky confidence, got {res['sky_confidence']}"

    # Near-zero water (< 2.0%), directly fixing FloodNet's 66.66% false positive
    assert res["flood_water_pct"] < 2.0, f"Expected near-zero water, got {res['flood_water_pct']}%"
    assert res["is_oblique_view"] is True


def test_custom_inspection_integration_and_discrepancy():
    """Confirms run_full_custom_inspection includes SegFormer output and flags discrepancy on Sydney houses."""
    img_path = Path("data/real/houses-on-the-hill-beautiful-suburb-of-sydney-palm-beach-background-with-copy-space.webp")
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")

    res = run_full_custom_inspection(
        post_image_input=str(img_path),
        zone_name="Sydney Hillside Test",
        disaster_mode="auto"
    )

    assert "segformer_water" in res
    assert res["segformer_water"] is not None
    assert "routing" in res

    routing = res["routing"]
    assert routing["models_disagree"] is True
    assert routing["disagreement_note"] is not None
    assert "FloodNet is nadir-calibrated" in routing["disagreement_note"]
    # Confirms auto-routing avoids classifying Sydney as a flood event
    assert routing["primary_hazard"] != "flood"
    print(f"Routing correctly resolved discrepancy: {routing['primary_hazard_label']}")
