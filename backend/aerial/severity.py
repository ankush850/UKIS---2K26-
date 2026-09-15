"""
Severity Scoring and Tactical Prioritization Module.
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.

Formula:
  severity = f(affected_area_%, damage_class_weight, confidence_score)

Weights:
  - building-destroyed: 1.0
  - major-damage: 0.7
  - road-blocked: 0.6
  - flooded: 0.5
  - minor-damage: 0.4
  - debris: 0.3

Buckets:
  - High Priority (Critical): 70 - 100 (Red #EF4444)
  - Medium Priority (Moderate): 40 - 69 (Amber #F59E0B)
  - Low Priority (Stable): 0 - 39 (Emerald #10B981)
"""

from typing import Dict, Any, List, Optional
import numpy as np


class SeverityScorer:
    WEIGHTS = {
        "building_destroyed": 1.0,
        "major_damage": 0.7,
        "road_blocked": 0.6,
        "flooded": 0.5,
        "minor_damage": 0.4,
        "debris": 0.3
    }

    @classmethod
    def calculate_severity(
        cls,
        affected_area_pct: float,
        damage_distribution: Dict[str, float],
        confidence_score: float = 0.85,
        blocked_roads_count: int = 0
    ) -> Dict[str, Any]:
        """
        Computes calibrated 0–100 severity score and returns tactical emergency recommendations.
        
        Args:
            affected_area_pct: Percentage of zone land affected (0.0 to 100.0).
            damage_distribution: Dictionary of class percentages or weights:
                - building_destroyed (pct)
                - major_damage (pct)
                - flooded (pct)
                - debris (pct)
                - minor_damage (pct)
            confidence_score: Model confidence score (0.0 to 1.0, e.g. from MC-Dropout or Drone U-Net).
            blocked_roads_count: Total count of blocked transport links.
        """
        # 1. Base Impact Component from affected land ratio (up to 40 points)
        base_area_pts = min(40.0, (affected_area_pct / 100.0) * 45.0)

        # 2. Weighted Damage Composition Component (up to 45 points)
        comp_sum = 0.0
        norm_factor = 0.0

        for key, weight in cls.WEIGHTS.items():
            pct = damage_distribution.get(key, 0.0)
            comp_sum += (pct / 100.0) * weight * 45.0
            norm_factor += weight

        damage_comp_pts = min(45.0, comp_sum)

        # 3. Transportation Chokepoint Penalty (up to 15 points)
        # Blocked mountain roads in Uttarakhand trap entire valleys
        road_penalty = min(15.0, blocked_roads_count * 5.0)

        # 4. Raw Severity
        raw_score = base_area_pts + damage_comp_pts + road_penalty

        # 5. Honest Confidence Calibration
        # We scale raw score by confidence function (0.7 + 0.3 * confidence)
        # High confidence confirms the disaster severity; low confidence tempers the metric slightly
        calibrated_score = raw_score * (0.70 + 0.30 * float(np.clip(confidence_score, 0.0, 1.0)))
        final_score = int(round(max(0.0, min(100.0, calibrated_score))))

        # 6. Priority Bucketing & Tactical Directive
        if final_score >= 70:
            level = "HIGH"
            badge = "CRITICAL RESCUE PRIORITY"
            color_hex = "#EF4444"
            action_code = "RED-ALPHA"
            directive = (
                "IMMEDIATE SEARCH & RESCUE DEPLOYMENT: Deploy NDRF / SDRF teams with high-lift helicopters "
                "and heavy debris-clearing bulldozers. Establish air-bridge for medical evacuation. "
                "Main access corridors are severed."
            )
        elif final_score >= 40:
            level = "MEDIUM"
            badge = "MODERATE SECONDARY TRIAGE"
            color_hex = "#F59E0B"
            action_code = "AMBER-BRAVO"
            directive = (
                "TACTICAL CLEARANCE & STABILIZATION: Dispatch state road clearing teams to remove debris "
                "and restore single-lane emergency traffic. Coordinate water pumping and structural inspection "
                "for partially compromised dwellings."
            )
        else:
            level = "LOW"
            badge = "ROUTINE MONITORING"
            color_hex = "#10B981"
            action_code = "GREEN-CHARLIE"
            directive = (
                "LOGISTICAL MONITORING: Primary infrastructure intact. Continue periodic aerial patrol "
                "to ensure receding water levels. Normal emergency vehicle access maintained."
            )

        return {
            "severity_score": final_score,
            "level": level,
            "badge": badge,
            "color_hex": color_hex,
            "action_code": action_code,
            "confidence_score": round(confidence_score, 2),
            "affected_area_pct": round(affected_area_pct, 1),
            "blocked_roads_count": blocked_roads_count,
            "components": {
                "area_impact_points": round(base_area_pts, 1),
                "damage_weighted_points": round(damage_comp_pts, 1),
                "transport_chokepoint_penalty": round(road_penalty, 1),
            },
            "directive": directive
        }


# Global singleton
severity_scorer = SeverityScorer()
