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
from backend.aerial.landslide_segmentation import landslide_segmentation_engine
from backend.aerial.water_detection_segformer import segformer_water_detector
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
    gsd_m: float = 0.10,
    disaster_mode: str = "auto"  # "auto", "flood", "landslide"
) -> Dict[str, Any]:
    """
    Runs the comprehensive tactical drone inspection pipeline on uploaded imagery:
    1. EXIF GPS extraction + Himalayan bounding-box domain guard
    2. Dual ML Segmentation:
       - FloodNet DeepLabV3+ (4 classes: background, flooded-building, flooded-road, water)
       - TransLandSeg (SAM ViT-L · Bijie-trained dedicated landslide scar detector)
    3. Intelligent Multi-Model Hazard Routing (Auto or Manual Override)
    4. Calibrated Severity Scoring (0-100 score + High/Medium/Low priority)
    5. Road Accessibility Corridor Analysis (Overpass OSM vectors)
    6. Building Damage Assessment (Single vs Siamese SiamUnet Pre/Post)
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

    # Calibrated geographic domain check (Uttarakhand / Himalayan disaster corridor)
    # Bounding box: Lat [28.5, 31.8], Lon [77.4, 81.3]
    in_domain = None
    domain_warning = None
    if final_lat is not None and final_lon is not None:
        if (28.5 <= final_lat <= 31.8) and (77.4 <= final_lon <= 81.3):
            in_domain = True
            domain_warning = None
        else:
            in_domain = False
            domain_warning = "This image is outside the model's calibrated geographic domain — results may be unreliable."
    else:
        in_domain = None
        domain_warning = "Location not available (no EXIF GPS) — geographic domain guard cannot verify regional calibration."

    domain_guard = {
        "calibrated_region": "Uttarakhand / Garhwal & Kumaon Himalayas",
        "calibrated_bbox": [77.4, 28.5, 81.3, 31.8],
        "in_domain": in_domain,
        "warning": domain_warning,
        "domain_notes": "FloodNet calibrated on nadir flood imagery; TransLandSeg calibrated on mountainous Bijie landslide terrain."
    }

    # 2. Multi-Model ML Inference
    # Model A: FloodNet DeepLabV3+
    seg_result = drone_segmentation_engine.segment(post_pil, gsd_m=gsd_m)
    hazard = seg_result["hazard_summary"]
    dist = seg_result["distribution"]

    # Model B: TransLandSeg (Dedicated Bijie-trained Landslide Detector)
    landslide_res = None
    try:
        landslide_res = landslide_segmentation_engine.segment(post_pil, gsd_m=gsd_m)
    except Exception as e:
        print(f"[CustomInspection] TransLandSeg inference note: {e}")

    # Model C: SegFormer ADE20K (General-Scene Water & Sky Disambiguator)
    segformer_res = None
    try:
        segformer_res = segformer_water_detector.detect_water(post_pil, gsd_m=gsd_m)
    except Exception as e:
        print(f"[CustomInspection] SegFormer water detector note: {e}")

    # 3. Routing Logic (Auto vs Manual Override)
    flood_water_pct = float(hazard.get("flooded_pct", 0.0))
    landslide_scar_pct = float(landslide_res.get("landslide_pct", 0.0)) if landslide_res else 0.0

    sf_water_pct = float(segformer_res.get("flood_water_pct", 0.0)) if segformer_res else 0.0
    sf_sky_pct = float(segformer_res.get("sky_pct", 0.0)) if segformer_res else 0.0
    sf_water_conf = float(segformer_res.get("water_confidence", 0.0)) if segformer_res else 0.0
    sf_sky_conf = float(segformer_res.get("sky_confidence", 0.0)) if segformer_res else 0.0

    # Discrepancy Detection: FloodNet nadir-bias misclassifying sky/horizon as water
    models_disagree = False
    disagreement_note = None
    if flood_water_pct >= 10.0 and sf_water_pct < 3.0 and sf_sky_pct >= 10.0:
        models_disagree = True
        disagreement_note = (
            f"Models disagree: FloodNet detected {flood_water_pct}% flood/water, but SegFormer identified "
            f"{sf_sky_pct}% sky ({round(sf_sky_conf * 100, 1)}% confidence) and near-zero water ({sf_water_pct}%). "
            f"FloodNet is nadir-calibrated and may misread sky as water in oblique shots; "
            f"SegFormer's general scene understanding is likely more reliable here."
        )

    if disaster_mode == "landslide":
        primary_hazard = "landslide"
        routing_label = "Landslide / Debris Flow"
        routing_synthesis = (
            f"Manual Override (Landslide Scenario): TransLandSeg detected {landslide_scar_pct}% landslide scar. "
            f"FloodNet detected {flood_water_pct}% flood/debris indicators."
        )
    elif disaster_mode == "flood":
        primary_hazard = "flood"
        routing_label = "Flood / Riparian Inundation"
        routing_synthesis = (
            f"Manual Override (Flood Scenario): FloodNet detected {flood_water_pct}% floodwater. "
            f"SegFormer confirmed {sf_water_pct}% water ({round(sf_water_conf * 100, 1)}% conf). "
            f"TransLandSeg detected {landslide_scar_pct}% landslide scar."
        )
    else:  # auto
        if models_disagree:
            if landslide_scar_pct >= 4.0:
                primary_hazard = "landslide"
                routing_label = "Landslide / Debris Flow"
                routing_synthesis = (
                    f"TransLandSeg detected {landslide_scar_pct}% landslide scar (Dominant). "
                    f"Note: FloodNet's {flood_water_pct}% flood reading was resolved as sky horizon by SegFormer ({sf_sky_pct}% sky)."
                )
            else:
                primary_hazard = "baseline"
                routing_label = "Baseline Terrain / Stable"
                routing_synthesis = (
                    f"Scene classified as Stable Baseline Terrain. "
                    f"FloodNet's {flood_water_pct}% flood reading was resolved as sky horizon by SegFormer ({sf_sky_pct}% sky, 0% water)."
                )
        elif landslide_scar_pct >= 4.0 and landslide_scar_pct > max(flood_water_pct, sf_water_pct):
            primary_hazard = "landslide"
            routing_label = "Landslide / Debris Flow"
            routing_synthesis = (
                f"Flood indicators: low ({flood_water_pct}%) — Landslide indicators: high ({landslide_scar_pct}%) — "
                f"classified as Landslide Event (TransLandSeg Dominant)."
            )
        elif (flood_water_pct >= 4.0 and not models_disagree) or sf_water_pct >= 4.0:
            primary_hazard = "flood"
            routing_label = "Flood / Riparian Inundation"
            routing_synthesis = (
                f"Flood indicators: confirmed ({max(flood_water_pct, sf_water_pct)}%) — "
                f"Landslide indicators: low ({landslide_scar_pct}%) — classified as Flood/Inundation Event."
            )
        else:
            primary_hazard = "baseline"
            routing_label = "Baseline Terrain / Stable"
            routing_synthesis = (
                f"Both hazard indicators low: FloodNet water {flood_water_pct}%, SegFormer water {sf_water_pct}%, "
                f"TransLandSeg scar {landslide_scar_pct}% — classified as Stable Baseline Terrain."
            )

    routing_info = {
        "disaster_mode": disaster_mode,
        "primary_hazard": primary_hazard,
        "primary_hazard_label": routing_label,
        "synthesis": routing_synthesis,
        "flood_indicators_pct": flood_water_pct,
        "landslide_indicators_pct": landslide_scar_pct,
        "segformer_water_pct": sf_water_pct,
        "segformer_water_confidence": sf_water_conf,
        "segformer_sky_pct": sf_sky_pct,
        "segformer_sky_confidence": sf_sky_conf,
        "models_disagree": models_disagree,
        "disagreement_note": disagreement_note,
        "dominant_model": (
            "TransLandSeg (Bijie ViT-L)" if primary_hazard == "landslide"
            else ("FloodNet & SegFormer Dual-Water" if primary_hazard == "flood"
            else ("SegFormer Horizon Disambiguated" if models_disagree else "Multi-Model Consensus (Baseline)"))
        )
    }

    # 4. Calibrated Severity Scoring (0-100)
    effective_flood_pct = sf_water_pct if models_disagree else flood_water_pct
    effective_debris_pct = max(hazard.get("debris_pct", 0.0), landslide_scar_pct)
    dmg_dist = {
        "building_destroyed": dist.get("building-damaged", {}).get("percentage", 0.0) * 0.4,
        "major_damage": dist.get("building-damaged", {}).get("percentage", 0.0) * 0.6,
        "flooded": effective_flood_pct,
        "debris": effective_debris_pct,
        "minor_damage": dist.get("building-intact", {}).get("percentage", 0.0) * 0.1,
        "road_blocked": hazard.get("blocked_roads_pct", 0.0)
    }
    affected_pct = min(100.0, effective_flood_pct + effective_debris_pct + hazard.get("damaged_buildings_pct", 0.0))
    
    severity_res = severity_scorer.calculate_severity(
        affected_area_pct=affected_pct,
        damage_distribution=dmg_dist,
        confidence_score=0.94,
        blocked_roads_count=1 if hazard["blocked_roads_pct"] > 5.0 else 0
    )

    # 5. Road Accessibility Analysis
    if location_info["has_location"]:
        lat = location_info["latitude"]
        lon = location_info["longitude"]
        bbox = [round(lon - 0.025, 4), round(lat - 0.025, 4), round(lon + 0.025, 4), round(lat + 0.025, 4)]
        
        raw_mask = seg_result.get("raw_mask")
        flood_mask = (raw_mask == 1).astype(np.uint8) if raw_mask is not None else None
        
        # Prefer dedicated landslide mask for road blockage if available
        if landslide_res and "raw_mask" in landslide_res:
            debris_mask = landslide_res["raw_mask"].astype(np.uint8)
        else:
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
        "domain_guard": domain_guard,
        "routing": routing_info,
        "segmentation": {
            "distribution": dist,
            "hazard_summary": hazard,
            "total_area_m2": seg_result.get("total_area_m2", 0.0),
            "segmentation_mask_b64": seg_result.get("segmentation_mask_b64")
        },
        "landslide_segmentation": {
            "status": landslide_res.get("status"),
            "model_name": landslide_res.get("model_name"),
            "checkpoint_source": landslide_res.get("checkpoint_source"),
            "checkpoint_fingerprint": landslide_res.get("checkpoint_fingerprint"),
            "real_model_inference": landslide_res.get("real_model_inference", False),
            "landslide_pct": landslide_res.get("landslide_pct", 0.0),
            "landslide_area_m2": landslide_res.get("landslide_area_m2", 0.0),
            "non_landslide_pct": landslide_res.get("non_landslide_pct", 100.0),
            "confidence": landslide_res.get("confidence", 0.0),
            "has_active_landslide": landslide_res.get("has_active_landslide", False),
            "segmentation_mask_b64": landslide_res.get("segmentation_mask_b64"),
            "summary": landslide_res.get("summary")
        } if landslide_res else None,
        "segformer_water": {
            "status": segformer_res.get("status"),
            "model_name": segformer_res.get("model_name"),
            "architecture": segformer_res.get("architecture"),
            "benchmark_dataset": segformer_res.get("benchmark_dataset"),
            "real_model_inference": segformer_res.get("real_model_inference", True),
            "flood_water_pct": segformer_res.get("flood_water_pct", 0.0),
            "flood_water_area_m2": segformer_res.get("flood_water_area_m2", 0.0),
            "water_confidence": segformer_res.get("water_confidence", 0.0),
            "sky_pct": segformer_res.get("sky_pct", 0.0),
            "sky_confidence": segformer_res.get("sky_confidence", 0.0),
            "is_oblique_view": segformer_res.get("is_oblique_view", False),
            "class_breakdown": segformer_res.get("class_breakdown", {}),
            "segmentation_mask_b64": segformer_res.get("segmentation_mask_b64"),
            "summary": segformer_res.get("summary")
        } if segformer_res else None,
        "severity": severity_res,
        "road_accessibility": road_accessibility_data,
        "building_damage": building_assessment,
        "preview_image_b64": post_b64
    }
