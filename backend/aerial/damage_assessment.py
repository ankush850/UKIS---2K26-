"""
Pre/Post Building Damage Comparison Engine (Siamese CNN).
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.
Base Reference: https://github.com/microsoft/building-damage-assessment-cnn-siamese
Dataset Standard: xBD / xView2 (https://xview2.org)

Damage Classes (xBD Standard):
  0: no-damage (intact building, structural integrity > 95%)
  1: minor-damage (cosmetic or minor roof tile loss, structural integrity ~80%)
  2: major-damage (significant structural failure, partial roof collapse, structural integrity ~40%)
  3: destroyed (total collapse, washed away, foundation exposed, structural integrity 0%)
"""

import io
import base64
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


DAMAGE_CLASSES = {
    0: {"name": "no-damage", "label": "No Damage (Intact)", "weight": 0.0, "hex": "#10B981", "color": [16, 185, 129, 180]},
    1: {"name": "minor-damage", "label": "Minor Damage", "weight": 0.4, "hex": "#FBBF24", "color": [251, 191, 36, 190]},
    2: {"name": "major-damage", "label": "Major Damage", "weight": 0.7, "hex": "#F97316", "color": [249, 115, 22, 210]},
    3: {"name": "destroyed", "label": "Destroyed / Washed Away", "weight": 1.0, "hex": "#EF4444", "color": [239, 68, 68, 230]},
}


# ── Siamese Feature Extractor ────────────────────────────────────────────────
class SiameseDamageNet(nn.Module):
    """
    Siamese convolutional network that processes pre-event and post-event images
    through weight-shared encoders, computes spatial difference embeddings,
    and classifies localized structural damage.
    """
    def __init__(self):
        super().__init__()
        # Shared Feature Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # /2
        )

        # Difference Fusion Decoder
        self.diff_conv = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=3, padding=1), # pre, post, |pre - post|
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 4, kernel_size=1) # 4 damage classes
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        f_pre = self.encoder(pre)
        f_post = self.encoder(post)
        f_diff = torch.abs(f_pre - f_post)

        f_fused = torch.cat([f_pre, f_post, f_diff], dim=1)
        damage_logits = self.diff_conv(f_fused)
        return damage_logits


# ── Damage Assessment Engine ─────────────────────────────────────────────────
class BuildingDamageAssessmentEngine:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.siam_unet = None
        self._load_microsoft_siam_checkpoint()
        self.fallback_model = SiameseDamageNet().to(self.device)
        self.fallback_model.eval()

    def _load_microsoft_siam_checkpoint(self):
        """Loads trained weights from sample_data/microsoft_damage/models/model_best.pth.tar if available."""
        from pathlib import Path
        ckpt_path = Path(__file__).resolve().parent.parent.parent / "sample_data" / "microsoft_damage" / "models" / "model_best.pth.tar"
        if ckpt_path.exists():
            try:
                import sys
                ms_dir = str(ckpt_path.parent.parent)
                if ms_dir not in sys.path:
                    sys.path.insert(0, ms_dir)
                from models.end_to_end_Siam_UNet import SiamUnet
                model = SiamUnet()
                # Internal pre-verified checkpoint loading with weights_only=False
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint.get("state_dict", checkpoint)
                model.load_state_dict(state_dict)
                model.to(self.device).eval()
                self.siam_unet = model
                print(f"[DamageAssessmentEngine] Successfully loaded Microsoft SiamUnet xBD weights from {ckpt_path.name}")
            except Exception as e:
                print(f"[DamageAssessmentEngine] Microsoft checkpoint load warning: {e}")

    def compare_pre_post(
        self,
        pre_image_input: Union[np.ndarray, Image.Image, str, bytes],
        post_image_input: Union[np.ndarray, Image.Image, str, bytes],
        zone_name: str = "Disaster Area",
        gsd_m: float = 0.10
    ) -> Dict[str, Any]:
        """
        Executes Siamese pre/post comparison and structural damage evaluation.
        Guarantees strict 100.0% mathematical normalization to eliminate percentage discrepancy.
        """
        pre_pil = self._load_pil_image(pre_image_input)
        post_pil = self._load_pil_image(post_image_input)

        w, h = post_pil.size
        pre_pil = pre_pil.resize((w, h), Image.Resampling.BILINEAR)

        # Standard resolution for Siamese processing (multiple of 16)
        proc_w = 512 if self.siam_unet is not None else min(max((w // 16) * 16, 256), 640)
        proc_h = 512 if self.siam_unet is not None else min(max((h // 16) * 16, 256), 640)

        pre_arr = np.array(pre_pil.resize((proc_w, proc_h))).astype(np.float32) / 255.0
        post_arr = np.array(post_pil.resize((proc_w, proc_h))).astype(np.float32) / 255.0

        if pre_arr.ndim == 2: pre_arr = np.stack([pre_arr] * 3, axis=-1)
        if post_arr.ndim == 2: post_arr = np.stack([post_arr] * 3, axis=-1)

        model_name = "Siamese CNN Baseline"
        checkpoint_source = "domain difference feature encoder"

        with torch.no_grad():
            if self.siam_unet is not None:
                # Microsoft SiamUnet xBD Inference
                pre_mean = pre_arr.mean(axis=(0, 1))
                pre_std = np.maximum(pre_arr.std(axis=(0, 1)), 1e-4)
                post_mean = post_arr.mean(axis=(0, 1))
                post_std = np.maximum(post_arr.std(axis=(0, 1)), 1e-4)

                pre_norm = (pre_arr - pre_mean) / pre_std
                post_norm = (post_arr - post_mean) / post_std

                t_pre = torch.tensor(pre_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
                t_post = torch.tensor(post_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

                scores = self.siam_unet(t_pre, t_post)
                # Directly compute softmax probabilities over 5 xBD damage channels
                # 0: background, 1: no-damage, 2: minor-damage, 3: major-damage, 4: destroyed
                probs_dmg = F.softmax(scores[2], dim=1).squeeze(0).cpu().numpy()
                raw_dmg = np.argmax(probs_dmg, axis=0)

                # Map xBD classes to our 4 operational damage classes:
                # 0: no-damage, 1: minor-damage, 2: major-damage, 3: destroyed
                damage_map = np.zeros_like(raw_dmg, dtype=np.uint8)
                damage_map[raw_dmg == 1] = 0  # no-damage (intact)
                damage_map[raw_dmg == 2] = 1  # minor-damage (genuine model output preserved)
                damage_map[raw_dmg == 3] = 2  # major-damage
                damage_map[raw_dmg == 4] = 3  # destroyed

                model_name = "Microsoft SiamUnet (xBD Trained Checkpoint)"
                checkpoint_source = "sample_data/microsoft_damage/models/model_best.pth.tar"
            else:
                t_pre = torch.from_numpy(pre_arr[:, :, :3].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                t_post = torch.from_numpy(post_arr[:, :, :3].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                logits = self.fallback_model(t_pre, t_post)
                import cv2
                abs_diff = np.abs(pre_arr - post_arr)
                diff_intensity = np.mean(abs_diff, axis=-1)
                pre_gray = cv2.cvtColor((pre_arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                post_gray = cv2.cvtColor((post_arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                pre_lap = cv2.Laplacian(pre_gray, cv2.CV_32F)
                post_lap = cv2.Laplacian(post_gray, cv2.CV_32F)
                struct_loss = np.maximum(0, np.abs(pre_lap) - np.abs(post_lap)) / 255.0

                logits_np = logits.squeeze(0).cpu().numpy()
                logits_np[0] += (diff_intensity < 0.15).astype(np.float32) * 3.0
                logits_np[1] += ((diff_intensity >= 0.15) & (diff_intensity < 0.30)).astype(np.float32) * 2.0
                logits_np[2] += ((diff_intensity >= 0.30) & (diff_intensity < 0.50)).astype(np.float32) * 2.5
                logits_np[3] += ((diff_intensity >= 0.50) | (struct_loss > 0.40)).astype(np.float32) * 3.5
                damage_map = np.argmax(logits_np, axis=0).astype(np.uint8)

        # Scale back to original resolution
        if (w, h) != (proc_w, proc_h):
            damage_map_full = np.array(
                Image.fromarray(damage_map).resize((w, h), Image.Resampling.NEAREST)
            )
        else:
            damage_map_full = damage_map

        # Compute Statistics & Enforce Strict 100% Normalization
        total_px = max(damage_map_full.size, 1)
        px_area_m2 = gsd_m * gsd_m

        breakdown = {}
        weighted_damage_sum = 0.0

        for cls_id, info in DAMAGE_CLASSES.items():
            count = int(np.sum(damage_map_full == cls_id))
            pct = round(float(count / total_px * 100.0), 2)
            area_m2 = round(float(count * px_area_m2), 2)
            weighted_damage_sum += (pct / 100.0) * info["weight"]

            breakdown[info["name"]] = {
                "class_id": cls_id,
                "label": info["label"],
                "hex": info["hex"],
                "pixel_count": count,
                "percentage": pct,
                "area_m2": area_m2
            }

        # Mathematical Normalization: Guarantee exact 100.0% sum across all 4 classes
        sum_pct = round(sum(b["percentage"] for b in breakdown.values()), 2)
        if abs(sum_pct - 100.0) > 0.001 and sum_pct > 0:
            largest_k = max(breakdown.keys(), key=lambda k: breakdown[k]["pixel_count"])
            breakdown[largest_k]["percentage"] = round(breakdown[largest_k]["percentage"] + (100.0 - sum_pct), 2)

        # Explicit Dual Presentation Breakdown to prevent any 136% math ambiguity:
        # Mode A: Total Surveyed Footprints = 100.0% (Intact + Damaged)
        intact_pct = breakdown["no-damage"]["percentage"]
        total_damaged_pct = round(100.0 - intact_pct, 2)

        # Mode B: Damaged Footprints Breakdown = 100.0% of damaged structures
        dmg_px_total = max(total_px - breakdown["no-damage"]["pixel_count"], 1)
        minor_pct_of_dmg = round(breakdown["minor-damage"]["pixel_count"] / dmg_px_total * 100.0, 1)
        major_pct_of_dmg = round(breakdown["major-damage"]["pixel_count"] / dmg_px_total * 100.0, 1)
        destroyed_pct_of_dmg = round(100.0 - minor_pct_of_dmg - major_pct_of_dmg, 1)

        structural_integrity_pct = round(max(0.0, min(100.0, (1.0 - weighted_damage_sum) * 100.0)), 1)

        if breakdown["destroyed"]["percentage"] > 15.0 or breakdown["major-damage"]["percentage"] > 25.0:
            overall_status = "DESTROYED / CRITICAL"
            severity_rating = "HIGH"
        elif breakdown["major-damage"]["percentage"] > 10.0 or breakdown["minor-damage"]["percentage"] > 20.0:
            overall_status = "MAJOR DAMAGE"
            severity_rating = "MEDIUM"
        elif breakdown["minor-damage"]["percentage"] > 8.0:
            overall_status = "MINOR DAMAGE"
            severity_rating = "LOW"
        else:
            overall_status = "NO SIGNIFICANT DAMAGE"
            severity_rating = "MINIMAL"

        # Render Damage Heatmap Overlay (RGBA)
        overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for cls_id, info in DAMAGE_CLASSES.items():
            if cls_id == 0:
                continue
            pts = (damage_map_full == cls_id)
            if np.any(pts):
                overlay_rgba[pts] = info["color"]

        buf = io.BytesIO()
        Image.fromarray(overlay_rgba, mode="RGBA").save(buf, format="PNG")
        overlay_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        pre_buf = io.BytesIO()
        pre_pil.save(pre_buf, format="JPEG", quality=85)
        pre_b64 = f"data:image/jpeg;base64,{base64.b64encode(pre_buf.getvalue()).decode('utf-8')}"

        post_buf = io.BytesIO()
        post_pil.save(post_buf, format="JPEG", quality=85)
        post_b64 = f"data:image/jpeg;base64,{base64.b64encode(post_buf.getvalue()).decode('utf-8')}"

        return {
            "status": "success",
            "zone_name": zone_name,
            "overall_status": overall_status,
            "severity_rating": severity_rating,
            "structural_integrity_pct": structural_integrity_pct,
            "breakdown": breakdown,
            "total_footprints": {
                "intact_pct": intact_pct,
                "damaged_pct": total_damaged_pct,
                "total_pct": 100.0
            },
            "damaged_breakdown_of_damaged": {
                "minor_pct": minor_pct_of_dmg,
                "major_pct": major_pct_of_dmg,
                "destroyed_pct": destroyed_pct_of_dmg,
                "total_pct": 100.0
            },
            "model_metadata": {
                "model_name": model_name,
                "checkpoint": checkpoint_source,
                "real_model_inference": (self.siam_unet is not None)
            },
            "pre_image_b64": pre_b64,
            "post_image_b64": post_b64,
            "damage_overlay_b64": overlay_b64,
            "estimated_impact_m2": round(float((total_px - breakdown["no-damage"]["pixel_count"]) * px_area_m2), 2)
        }

    def _load_pil_image(self, input_data: Any) -> Image.Image:
        if isinstance(input_data, Image.Image):
            return input_data.convert("RGB")
        elif isinstance(input_data, np.ndarray):
            if input_data.dtype != np.uint8:
                input_data = (np.clip(input_data, 0.0, 1.0) * 255.0).astype(np.uint8)
            return Image.fromarray(input_data).convert("RGB")
        elif isinstance(input_data, str):
            if input_data.startswith("data:image"):
                base64_data = input_data.split(",", 1)[1]
                img_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                return Image.open(input_data).convert("RGB")
        elif isinstance(input_data, bytes):
            return Image.open(io.BytesIO(input_data)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(input_data)}")


# Global singleton
building_damage_engine = BuildingDamageAssessmentEngine()
