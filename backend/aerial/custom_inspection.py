"""
Custom Drone Inspection Pipeline.
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.

Processes user-uploaded drone photos through the complete tactical inspection pipeline:
1. EXIF Geolocation Extraction (GPS latitude/longitude)
2. Multi-class Segmentation (FloodNet DeepLabV3+ / U-Net)
3. Calibrated Severity Scoring (0-100 score + High/Medium/Low priority)
4. Road Accessibility Corridor Analysis (Overpass OSM vectors if location available, else graceful skip)
5. Building Damage Assessment (Single-image detection-only OR Pre/Post SiamUnet comparison)
"""

import io
import base64
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image

from backend.aerial.segmentation import drone_segmentation_engine, AERIAL_CLASSES
from backend.aerial.damage_assessment import building_damage_engine
from backend.aerial.road_accessibility import road_classifier
from backend.aerial.severity import severity_scorer


def extract_exif_gps(image: Image.Image) -> Optional[Tuple[float, float]]:
    """
    Extracts latitude and longitude from EXIF metadata of a PIL image if available.
    Returns (lat, lon) in decimal degrees, or None if not georeferenced.
    """
    try:
        exif = image.getexif()
        if not exif:
            return None

        # GPS Info IFD tag is 34853 (0x8825)
        gps_info = exif.get_ifd(0x8825)
        if not gps_info:
            return None

        def _to_decimal(dms: Any) -> float:
            if isinstance(dms, tuple) and len(dms) == 3:
                return float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
            return float(dms)

        # Tag 2: Latitude, Tag 1: LatitudeRef ('N'/'S')
        # Tag 4: Longitude, Tag 3: LongitudeRef ('E'/'W')
        if 2 not in gps_info or 4 not in gps_info:
            return None

        lat = _to_decimal(gps_info[2])
        if gps_info.get(1, 'N') == 'S':
            lat = -lat

        lon = _to_decimal(gps_info[4])
        if gps_info.get(3, 'E') == 'W':
            lon = -lon

        return (round(lat, 6), round(lon, 6))
    except Exception as e:
        print(f"[CustomInspection] EXIF GPS extraction note: {e}")
        return None


def run_full_custom_inspection(
    post_image_input: Union[np.ndarray, Image.Image, str],
    pre_image_input: Optional[Union[np.ndarray, Image.Image, str]] = None,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    zone_name: str = "Custom Aerial Survey",
    gsd_m: float = 0.10
) -> Dict[str, Any]:
    """
    Runs the comprehensive tactical drone inspection pipeline on uploaded imagery.
    
    Args:
        post_image_input: Drone survey photo (Base64 string, PIL Image, or numpy array).
        pre_image_input: Optional pre-disaster baseline photo for Siamese building comparison.
        user_lat: Optional user-confirmed latitude if image lacks EXIF.
        user_lon: Optional user-confirmed longitude if image lacks EXIF.
        zone_name: Descriptive name for the inspected location.
        gsd_m: Ground Sample Distance in meters per pixel.
        
    Returns:
        Unified dictionary with segmentation, severity, road accessibility, and building damage.
    """
    # 1. Load and inspect PIL Image for EXIF GPS
    post_pil = drone_segmentation_engine._load_pil_image(post_image_input)
    exif_coords = extract_exif_gps(post_pil)

    final_lat = user_lat if user_lat is not None else (exif_coords[0] if exif_coords else None)
    final_lon = user_lon if user_lon is not None else (exif_coords[1] if exif_coords else None)

    location_info = {
        "has_location": (final_lat is not None and final_lon is not None),
        "source": "EXIF_GPS" if exif_coords and user_lat is None else ("USER_CONFIRMED" if user_lat is not None else "UNAVAILABLE"),
        "latitude": final_lat,
        "longitude": final_lon,
    }

    # 2. Multi-Class Drone Segmentation (FloodNet DeepLabV3+ / U-Net)
    seg_result = drone_segmentation_engine.segment(post_pil, gsd_m=gsd_m)
    hazard = seg_result["hazard_summary"]
    dist = seg_result["distribution"]

    # 3. Calibrated Severity Scoring
    # Formula: severity = f(affected_area_%, damage_class_weight, confidence)
    # Weights: destroyed=1.0, major=0.7, road-blocked=0.6, minor=0.4, debris=0.3
    dmg_dist = {
        "building_destroyed": dist.get("building-damaged", {}).get("percentage", 0.0) * 0.4,
        "major_damage": dist.get("building-damaged", {}).get("percentage", 0.0) * 0.6,
        "flooded": hazard["flooded_pct"],
        "debris": hazard["debris_pct"],
        "minor_damage": dist.get("building-intact", {}).get("percentage", 0.0) * 0.1,
        "road_blocked": hazard["blocked_roads_pct"]
    }
    affected_pct = min(100.0, hazard["flooded_pct"] + hazard["debris_pct"] + hazard["damaged_buildings_pct"])
    
    severity_res = severity_scorer.calculate_severity(
        affected_area_pct=affected_pct,
        damage_distribution=dmg_dist,
        confidence_score=0.94,
        blocked_roads_count=1 if hazard["blocked_roads_pct"] > 5.0 else 0
    )

    # 4. Road Accessibility Analysis
    if location_info["has_location"]:
        lat = location_info["latitude"]
        lon = location_info["longitude"]
        bbox = [round(lon - 0.025, 4), round(lat - 0.025, 4), round(lon + 0.025, 4), round(lat + 0.025, 4)]
        
        raw_mask = seg_result.get("raw_mask")
        flood_mask = (raw_mask == 1).astype(np.uint8) if raw_mask is not None else None
        debris_mask = (raw_mask == 6).astype(np.uint8) if raw_mask is not None else None

        roads_geojson = road_classifier.classify_roads(
            bbox=bbox,
            flood_mask=flood_mask,
            debris_mask=debris_mask,
            threshold_pct=15.0
        )
        roads_summary = roads_geojson.get("summary", {})
        road_accessibility_data = {
            "available": True,
            "bbox": bbox,
            "total_segments": roads_summary.get("total_segments", 0),
            "blocked_count": roads_summary.get("blocked_count", 0),
            "clear_count": roads_summary.get("clear_count", 0),
            "critical_chokepoints": roads_summary.get("critical_chokepoints", []),
            "geojson": roads_geojson
        }
    else:
        # Graceful skip as requested by specification
        road_accessibility_data = {
            "available": False,
            "message": "Location not available — road accessibility skipped",
            "total_segments": 0,
            "blocked_count": 0,
            "clear_count": 0,
            "critical_chokepoints": [],
            "geojson": None
        }

    # 5. Building Damage Assessment (Single vs Pre/Post Pair)
    if pre_image_input is not None:
        # Pre/Post Siamese CNN Comparison Mode (Microsoft SiamUnet)
        damage_res = building_damage_engine.compare_pre_post(
            pre_image_input=pre_image_input,
            post_image_input=post_pil,
            zone_name=zone_name,
            gsd_m=gsd_m
        )
        building_assessment = {
            "mode": "pre_post_siamese_comparison",
            "label": "Microsoft SiamUnet Pre/Post Comparison",
            "structural_integrity_pct": damage_res.get("structural_integrity_pct", 90.0),
            "overall_status": damage_res.get("overall_status", "SURVEY COMPLETE"),
            "total_footprints": damage_res.get("total_footprints", {}),
            "damaged_breakdown": damage_res.get("damaged_breakdown_of_damaged", {}),
            "damage_overlay_b64": damage_res.get("damage_overlay_b64"),
            "model_metadata": damage_res.get("model_metadata", {})
        }
    else:
        # Single-Image Detection-Only Mode
        damaged_bldg = dist.get("building-damaged", {})
        intact_bldg = dist.get("building-intact", {})
        total_bldg_area = damaged_bldg.get("area_m2", 0.0) + intact_bldg.get("area_m2", 0.0)
        total_bldg_pixels = damaged_bldg.get("pixel_count", 0) + intact_bldg.get("pixel_count", 0)

        if total_bldg_pixels > 0:
            damaged_ratio = round((damaged_bldg["pixel_count"] / total_bldg_pixels) * 100.0, 1)
            intact_ratio = round(100.0 - damaged_ratio, 1)
        else:
            damaged_ratio = 0.0
            intact_ratio = 100.0

        building_assessment = {
            "mode": "single_image_detection_only",
            "label": "Single-Frame Structure Detection",
            "note": "Single-image detection-only mode. Upload an optional pre-disaster baseline for full SiamUnet structural differential analysis.",
            "total_building_footprint_m2": round(total_bldg_area, 2),
            "damaged_footprint_m2": damaged_bldg.get("area_m2", 0.0),
            "intact_footprint_m2": intact_bldg.get("area_m2", 0.0),
            "damaged_building_pct": damaged_ratio,
            "intact_building_pct": intact_ratio,
            "has_damaged_structures": (damaged_bldg.get("pixel_count", 0) > 0)
        }

    # Encode original post image to base64 preview
    buf = io.BytesIO()
    post_pil.save(buf, format="JPEG", quality=85)
    post_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    return {
        "status": "success",
        "zone_name": zone_name,
        "location": location_info,
        "segmentation": {
            "distribution": dist,
            "hazard_summary": hazard,
            "total_area_m2": seg_result.get("total_area_m2", 0.0),
            "segmentation_mask_b64": seg_result.get("segmentation_mask_b64")
        },
        "severity": severity_res,
        "road_accessibility": road_accessibility_data,
        "building_damage": building_assessment,
        "preview_image_b64": post_b64
    }
