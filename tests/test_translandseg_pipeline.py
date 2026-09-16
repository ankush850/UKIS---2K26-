"""
Unit and Integration Tests for Dedicated Landslide Segmentation (TransLandSeg / Bijie)
and Multi-Model Hazard Routing (FloodNet + TransLandSeg).
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.aerial.landslide_segmentation import LandslideSegmentationEngine, landslide_segmentation_engine
from backend.aerial.custom_inspection import run_full_custom_inspection
from backend.app import app

client = TestClient(app)


def test_translandseg_engine_load_and_fingerprint():
    """Confirms TransLandSeg loads weights with 0 missing and 0 unexpected keys."""
    engine = landslide_segmentation_engine
    assert engine.model_loaded is True
    assert engine.checkpoint_path is not None
    assert engine.checkpoint_hash == "b69f843685fa"
    assert engine.keys_matched == 521
    assert engine.keys_missing == 0
    assert engine.keys_unexpected == 0
    print(f"TransLandSeg verified: {engine.keys_matched}/521 keys strictly loaded.")


def test_translandseg_inference_synthetic():
    """Tests forward inference of TransLandSeg on a synthetic image."""
    engine = landslide_segmentation_engine
    test_arr = np.zeros((256, 256, 3), dtype=np.uint8)
    test_arr[100:150, 100:150] = [180, 120, 80] # brown soil block

    res = engine.segment(test_arr, gsd_m=0.10)
    assert res["status"] == "success"
    assert "landslide_pct" in res
    assert "landslide_area_m2" in res
    assert "segmentation_mask_b64" in res
    assert res["segmentation_mask_b64"].startswith("data:image/png;base64,")
    assert res["real_model_inference"] is True
    assert res["checkpoint_fingerprint"] == "b69f843685fa"


def test_custom_inspection_dual_model_routing():
    """Confirms run_full_custom_inspection executes both models and routes properly."""
    test_img = "data/aerial_demo/chamoli_flash_flood/post_disaster.jpg"

    # 1. Auto mode
    res_auto = run_full_custom_inspection(
        post_image_input=test_img,
        zone_name="Test Zone",
        disaster_mode="auto"
    )
    assert res_auto["status"] == "success"
    assert "routing" in res_auto
    assert "landslide_segmentation" in res_auto
    assert "segmentation" in res_auto
    assert res_auto["routing"]["disaster_mode"] == "auto"
    assert res_auto["routing"]["primary_hazard"] in ["flood", "landslide", "baseline"]
    assert "dominant_model" in res_auto["routing"]

    # 2. Manual Landslide Override
    res_ls = run_full_custom_inspection(
        post_image_input=test_img,
        zone_name="Test Zone",
        disaster_mode="landslide"
    )
    assert res_ls["routing"]["primary_hazard"] == "landslide"
    assert "Landslide Scenario" in res_ls["routing"]["synthesis"]

    # 3. Manual Flood Override
    res_fl = run_full_custom_inspection(
        post_image_input=test_img,
        zone_name="Test Zone",
        disaster_mode="flood"
    )
    assert res_fl["routing"]["primary_hazard"] == "flood"
    assert "Flood Scenario" in res_fl["routing"]["synthesis"]


def test_api_inspect_upload_with_disaster_mode():
    """Tests /api/aerial/inspect-upload API endpoint with disaster_mode param."""
    # Generate small 64x64 base64 JPEG
    import io, base64
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 140, 90)).save(buf, format="JPEG")
    b64_str = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    response = client.post(
        "/api/aerial/inspect-upload",
        json={
            "image_b64": b64_str,
            "zone_name": "API Test Zone",
            "disaster_mode": "auto"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "routing" in data
    assert "landslide_segmentation" in data
    assert "segmentation" in data
    assert data["routing"]["disaster_mode"] == "auto"
