"""
Dedicated Landslide Semantic Segmentation Pipeline (TransLandSeg / Bijie).
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.
Base Reference: https://github.com/JunchuanYu/TransLandSeg (SAM ViT-L Transfer Learning on Bijie Landslide Dataset)

Detects active landslide scars, mudslides, and destabilized slip debris
independently of FloodNet floodwater classes.
"""

import io
import os
import base64
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import yaml

from backend.aerial.translandseg_models import models


def checkpoint_fingerprint(path: Path):
    """Calculates MD5 hash and file size for verification."""
    with open(path, "rb") as f:
        data = f.read()
        return hashlib.md5(data).hexdigest()[:12], len(data)


class LandslideSegmentationEngine:
    """
    Dedicated landslide segmentation engine utilizing TransLandSeg
    trained on the 770-image Bijie Landslide Dataset with a SAM ViT-L backbone.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model = None
        self.model_loaded = False
        self.checkpoint_path = None
        self.checkpoint_hash = None
        self.checkpoint_size = 0
        self.keys_matched = 0
        self.keys_missing = 0
        self.keys_unexpected = 0
        self.inp_size = 1024

        self.transform = transforms.Compose([
            transforms.Resize((self.inp_size, self.inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._load_translandseg_checkpoint()

    def _load_translandseg_checkpoint(self):
        """Loads TransLandSeg checkpoint from checkpoint/Bijie.pth.tar."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        ckpt_path = base_dir / "checkpoint" / "Bijie.pth.tar"
        yaml_path = Path(__file__).resolve().parent / "translandseg.yaml"

        if not ckpt_path.exists():
            print(f"[LandslideSegmentationEngine] Warning: Checkpoint not found at {ckpt_path}")
            return

        try:
            self.checkpoint_hash, self.checkpoint_size = checkpoint_fingerprint(ckpt_path)
            self.checkpoint_path = str(ckpt_path)

            with open(yaml_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            model = models.make(config['model'])
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            self.keys_matched = len(state_dict) - len(unexpected)
            self.keys_missing = len(missing)
            self.keys_unexpected = len(unexpected)

            model.to(self.device).eval()
            self.model = model
            self.model_loaded = True
            print(
                f"[LandslideSegmentationEngine] Successfully loaded TransLandSeg (Bijie) "
                f"from {ckpt_path.name} ({self.checkpoint_size / 1024 / 1024:.1f} MB, MD5: {self.checkpoint_hash}). "
                f"Verified {self.keys_matched}/{len(state_dict)} keys, eval mode confirmed."
            )
        except Exception as e:
            print(f"[LandslideSegmentationEngine] CRITICAL Checkpoint load error: {e}")
            raise RuntimeError(f"Failed to load TransLandSeg checkpoint: {e}") from e

    def segment(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
        gsd_m: float = 0.10
    ) -> Dict[str, Any]:
        """
        Executes dedicated landslide semantic segmentation.
        
        Args:
            image_input: NumPy RGB array, PIL Image, or Base64 data URL string.
            gsd_m: Ground Sample Distance in meters per pixel.
            
        Returns:
            Dictionary containing:
                - landslide_pct: % of frame classified as landslide scar / debris
                - landslide_area_m2: ground footprint in square meters
                - non_landslide_pct: baseline terrain %
                - confidence: mean probability over predicted landslide pixels
                - segmentation_mask_b64: base64 RGBA overlay
                - raw_mask: 2D numpy binary array (1=landslide, 0=non-landslide)
                - model_provenance: model name, checkpoint info, and verification
        """
        pil_img = self._load_pil_image(image_input)
        orig_w, orig_h = pil_img.size

        if not self.model_loaded or self.model is None:
            raise RuntimeError("TransLandSeg model is not loaded.")

        # 1. Preprocess for SAM ViT-L backbone (1024x1024)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 2. Forward Inference
        with torch.no_grad():
            pred_mask = self.model.infer(tensor)
            probs = torch.sigmoid(pred_mask).squeeze().cpu().numpy() # [1024, 1024]
            binary_1024 = (probs > 0.5).astype(np.uint8)

        # 3. Resize mask back to original image dimensions
        if (orig_w, orig_h) != (self.inp_size, self.inp_size):
            binary_orig = np.array(
                Image.fromarray(binary_1024).resize((orig_w, orig_h), Image.Resampling.NEAREST)
            )
            probs_orig = np.array(
                Image.fromarray((probs * 255.0).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            ).astype(np.float32) / 255.0
        else:
            binary_orig = binary_1024
            probs_orig = probs

        # 4. Metrics & Statistics
        total_pixels = max(binary_orig.size, 1)
        landslide_pixels = int(np.sum(binary_orig == 1))
        landslide_pct = round(float(landslide_pixels / total_pixels * 100.0), 2)
        non_landslide_pct = round(100.0 - landslide_pct, 2)

        pixel_area_m2 = gsd_m * gsd_m
        total_area_m2 = round(float(total_pixels * pixel_area_m2), 2)
        landslide_area_m2 = round(float(landslide_pixels * pixel_area_m2), 2)
        non_landslide_area_m2 = round(max(0.0, total_area_m2 - landslide_area_m2), 2)

        if landslide_pixels > 0:
            mean_conf = round(float(np.mean(probs_orig[binary_orig == 1])), 3)
        else:
            mean_conf = round(float(np.mean(probs_orig)), 3)

        # 5. Render Color Overlay (RGBA)
        # Landslide Scar Color: High-contrast Amber / Orange (#F59E0B) with 210 alpha
        overlay_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        ls_pts = (binary_orig == 1)
        if np.any(ls_pts):
            overlay_rgba[ls_pts] = [245, 158, 11, 215]

        buf = io.BytesIO()
        Image.fromarray(overlay_rgba, mode="RGBA").save(buf, format="PNG")
        mask_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return {
            "status": "success",
            "model_name": "TransLandSeg (SAM ViT-L · Bijie Landslide Benchmark)",
            "checkpoint_source": self.checkpoint_path,
            "checkpoint_fingerprint": self.checkpoint_hash,
            "real_model_inference": True,
            "dimensions": {"width": orig_w, "height": orig_h},
            "gsd_m": gsd_m,
            "total_area_m2": total_area_m2,
            "landslide_pct": landslide_pct,
            "landslide_area_m2": landslide_area_m2,
            "non_landslide_pct": non_landslide_pct,
            "non_landslide_area_m2": non_landslide_area_m2,
            "confidence": mean_conf,
            "has_active_landslide": (landslide_pct >= 2.0),
            "segmentation_mask_b64": mask_b64,
            "raw_mask": binary_orig,
            "summary": f"TransLandSeg detected {landslide_pct}% active landslide scar ({landslide_area_m2} m²) with {round(mean_conf*100, 1)}% confidence."
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
landslide_segmentation_engine = LandslideSegmentationEngine()
