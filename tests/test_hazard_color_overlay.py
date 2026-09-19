"""
Unit Tests for Selective Landslide & Flood Hazard Color Overlay on Real Drone Feed.
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.
"""

import sys
import io
import base64
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.aerial.segmentation import drone_segmentation_engine, AERIAL_CLASSES
from backend.aerial.live_stream import live_stream_simulator


def test_selective_hazard_transparency():
    """Confirms non-hazard ground/structures have alpha=0 (real photo color) and hazards have clear colors."""
    img = Image.open(BASE_DIR / "data" / "aerial_demo" / "chamoli_flash_flood" / "post_disaster.jpg")
    
    res = drone_segmentation_engine.segment(img, hazard_only=True)
    assert res["status"] == "success"
    assert res["hazard_only"] is True

    # Decode overlay mask PNG
    mask_b64 = res["segmentation_mask_b64"].split(",", 1)[1]
    mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
    mask_np = np.array(mask_img)

    raw_mask = res["raw_mask"]
    
    # 1. Non-hazard background pixels MUST have alpha == 0
    non_hazard_pts = (raw_mask == 0)
    assert np.any(non_hazard_pts), "Expected non-hazard pixels in test image"
    assert np.all(mask_np[non_hazard_pts, 3] == 0), "Non-hazard pixels must have alpha=0 so real camera colors show through!"

    # 2. Flood pixels MUST be Cyan [6, 182, 212] with alpha > 0
    flood_pts = (raw_mask == 1)
    if np.any(flood_pts):
        flood_rgba = mask_np[flood_pts]
        assert np.all(flood_rgba[:, 3] > 0), "Flood pixels must have non-zero alpha"
        assert flood_rgba[0, 0] == 6 and flood_rgba[0, 1] == 182 and flood_rgba[0, 2] == 212, "Flood color must be Cyan"

    # 3. Landslide / debris pixels MUST be Amber [245, 158, 11] with alpha > 0
    debris_pts = (raw_mask == 6)
    if np.any(debris_pts):
        debris_rgba = mask_np[debris_pts]
        assert np.all(debris_rgba[:, 3] > 0), "Landslide pixels must have non-zero alpha"
        assert debris_rgba[0, 0] == 245 and debris_rgba[0, 1] == 158 and debris_rgba[0, 2] == 11, "Landslide color must be Amber"

    print("  -> Success! Non-hazard pixels are 100% transparent (alpha=0), flood and landslide highlighted correctly.")


def test_full_semantic_mode_fallback():
    """Confirms hazard_only=False retains full semantic multi-class coloring."""
    img = Image.open(BASE_DIR / "data" / "aerial_demo" / "chamoli_flash_flood" / "post_disaster.jpg")
    
    res = drone_segmentation_engine.segment(img, hazard_only=False)
    assert res["status"] == "success"
    assert res["hazard_only"] is False

    mask_b64 = res["segmentation_mask_b64"].split(",", 1)[1]
    mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
    mask_np = np.array(mask_img)

    raw_mask = res["raw_mask"]
    non_hazard_pts = (raw_mask == 0)
    # In full semantic mode, non-hazard ground has color (alpha > 0)
    assert np.all(mask_np[non_hazard_pts, 3] > 0), "Full semantic mode must render color for non-hazard terrain"
    print("  -> Success! Full semantic mode renders colors across all classes as expected.")


def test_classes_hazard_definitions():
    """Confirms AERIAL_CLASSES specification has valid hazard flags and colors."""
    assert AERIAL_CLASSES[0]["is_hazard"] is False
    assert AERIAL_CLASSES[0]["hazard_color"][3] == 0
    assert AERIAL_CLASSES[1]["is_hazard"] is True
    assert AERIAL_CLASSES[1]["hazard_color"][3] > 0
    assert AERIAL_CLASSES[6]["is_hazard"] is True
    assert AERIAL_CLASSES[6]["hazard_color"][3] > 0
    print("  -> Success! AERIAL_CLASSES hazard flags verified.")


if __name__ == "__main__":
    print("RUNNING HAZARD COLOR OVERLAY TESTS...")
    test_selective_hazard_transparency()
    test_full_semantic_mode_fallback()
    test_classes_hazard_definitions()
    print("ALL TESTS PASSED SUCCESSFULLY!")
