"""
Dedicated General-Scene Water & Flood Semantic Segmentation (SegFormer ADE20K).
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.

Model: nvidia/segformer-b0-finetuned-ade-512-512 (Hugging Face, Apache-2.0)
Architecture: Transformer-based Hierarchical Semantic Segmentation (SegFormer)
Training Dataset: ADE20K (150 scene classes including sky, water, sea, river, lake)

Solves the nadir vs. oblique aerial horizon challenge:
- FloodNet only learned straight-down nadir photos where sky never appears.
- SegFormer ADE20K explicitly separates 'sky' (class 2) from 'water'/'sea'/'river'/'lake'.
- Computes genuine per-pixel softmax confidence scores for detected water bodies.
"""

import io
import base64
from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class SegFormerWaterDetector:
    """
    SegFormer ADE20K general-scene water and sky detection engine with
    pixel-level softmax confidence scoring.
    """
    MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.processor = None
        self.model = None
        self.model_loaded = False

        # ADE20K Class Index Mappings
        self.sky_class_id = None
        self.water_class_ids: Dict[str, int] = {}
        self.flood_water_id_set = set()

        self._load_segformer_model()

    def _load_segformer_model(self):
        """Initializes SegFormer processor and weights, logging ADE20K label IDs."""
        try:
            print(f"[SegFormerWaterDetector] Loading {self.MODEL_ID} on {self.device}...")
            self.processor = SegformerImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.MODEL_ID)
            self.model.to(self.device).eval()

            # Confirm ADE20K label list
            id2label = self.model.config.id2label

            # Find 'sky'
            for idx, label in id2label.items():
                clean_lbl = label.lower().strip()
                if clean_lbl == "sky":
                    self.sky_class_id = int(idx)
                    break

            # Find water-related classes: water, sea, river, lake
            target_water_names = ["water", "sea", "river", "lake"]
            for idx, label in id2label.items():
                clean_lbl = label.lower().strip()
                for target in target_water_names:
                    # Match exact or primary class tokens (avoiding 'seat', 'skyscraper')
                    tokens = [t.strip() for t in clean_lbl.split(",")]
                    if target in tokens:
                        self.water_class_ids[target] = int(idx)
                        self.flood_water_id_set.add(int(idx))

            # Mandatory architectural sanity check: sky must NEVER be counted as water
            assert self.sky_class_id is not None, "Sky class not found in ADE20K id2label"
            assert self.sky_class_id not in self.flood_water_id_set, "CRITICAL: Sky class must never be counted as water"

            self.model_loaded = True
            print(
                f"[SegFormerWaterDetector] Successfully initialized {self.MODEL_ID}. "
                f"Sky index: {self.sky_class_id} ('{id2label[self.sky_class_id]}') | "
                f"Water indices: {self.water_class_ids} | "
                f"Evaluation mode active."
            )
        except Exception as e:
            print(f"[SegFormerWaterDetector] Initialization error: {e}")
            raise RuntimeError(f"Failed to load SegFormer model: {e}") from e

    def detect_water(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
        gsd_m: float = 0.10
    ) -> Dict[str, Any]:
        """
        Executes SegFormer ADE20K inference on the input image.
        
        Args:
            image_input: NumPy RGB array, PIL Image, or Base64 data URL string.
            gsd_m: Ground Sample Distance in meters per pixel.
            
        Returns:
            Dictionary containing:
                - flood_water_pct: Combined percentage of water + sea + river + lake (sky excluded)
                - flood_water_area_m2: Estimated water ground footprint
                - water_confidence: Mean softmax probability across pixels classified as water (0.0 to 1.0)
                - sky_pct: Percentage of image occupied by sky
                - sky_confidence: Mean softmax probability across pixels classified as sky
                - class_breakdown: Individual percentages & confidences for each water sub-class
                - is_oblique_view: True if sky coverage > 5.0%
                - segmentation_mask_b64: Color-coded RGBA overlay (Cyan for water, Soft Azure for sky)
                - raw_mask: 2D numpy array (1=water, 2=sky, 0=other)
        """
        pil_img = self._load_pil_image(image_input)
        orig_w, orig_h = pil_img.size
        total_pixels = orig_w * orig_h

        if not self.model_loaded or self.model is None:
            raise RuntimeError("SegFormer model is not loaded.")

        # 1. Preprocess & Forward Inference
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, 150, H_feat, W_feat]

            # 2. Upsample logits to original image resolution (bilinear)
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )  # [1, 150, orig_h, orig_w]

            # 3. Softmax across class dimension
            probs = torch.softmax(upsampled, dim=1)[0]  # [150, orig_h, orig_w]

            # 4. Argmax winner per pixel
            pred = probs.argmax(dim=0)  # [orig_h, orig_w]

        # Move tensors to CPU numpy for metric calculations
        pred_np = pred.cpu().numpy()
        probs_cpu = probs.cpu()

        # 5. Sky Detection & Confidence
        sky_mask_t = (pred == self.sky_class_id)
        sky_pixels = int(sky_mask_t.sum().item())
        sky_pct = round(float(sky_pixels / total_pixels * 100.0), 2)
        if sky_pixels > 0:
            sky_conf = round(float(probs_cpu[self.sky_class_id][sky_mask_t].mean().item()), 3)
        else:
            sky_conf = 0.0

        # 6. Water Detection & Confidence (Combining water, sea, river, lake)
        water_breakdown = {}
        water_mask_t = torch.zeros((orig_h, orig_w), dtype=torch.bool, device=pred.device)
        water_probs_combined = []

        for name, cid in self.water_class_ids.items():
            cls_mask = (pred == cid)
            px_count = int(cls_mask.sum().item())
            cls_pct = round(float(px_count / total_pixels * 100.0), 2)

            if px_count > 0:
                cls_probs = probs_cpu[cid][cls_mask]
                cls_conf = round(float(cls_probs.mean().item()), 3)
                water_probs_combined.append(cls_probs)
            else:
                cls_conf = None

            water_breakdown[name] = {
                "class_id": cid,
                "pixel_count": px_count,
                "percentage": cls_pct,
                "confidence": cls_conf
            }
            water_mask_t |= cls_mask

        combined_water_pixels = int(water_mask_t.sum().item())
        flood_water_pct = round(float(combined_water_pixels / total_pixels * 100.0), 2)

        if len(water_probs_combined) > 0:
            all_w_probs = torch.cat(water_probs_combined)
            water_conf = round(float(all_w_probs.mean().item()), 3)
        else:
            water_conf = 0.0

        # Physical Ground Area ($m^2$)
        pixel_area_m2 = gsd_m * gsd_m
        flood_water_area_m2 = round(float(combined_water_pixels * pixel_area_m2), 2)

        # 7. Render Color Overlay (RGBA)
        # Water/Flood: Vivid Cyan [6, 182, 212, 215]
        # Sky: Soft Sky Blue [56, 189, 248, 120]
        water_mask_np = water_mask_t.cpu().numpy()
        sky_mask_np = sky_mask_t.cpu().numpy()

        overlay_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        # Water pixels
        overlay_rgba[water_mask_np] = [6, 182, 212, 215]
        # Sky pixels (rendered with gentle opacity so user sees horizon separation)
        overlay_rgba[sky_mask_np] = [56, 189, 248, 110]

        buf = io.BytesIO()
        Image.fromarray(overlay_rgba, mode="RGBA").save(buf, format="PNG")
        mask_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        # Raw mask representation: 1=Water, 2=Sky, 0=Other
        raw_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        raw_mask[water_mask_np] = 1
        raw_mask[sky_mask_np] = 2

        is_oblique = (sky_pct >= 5.0)

        return {
            "status": "success",
            "model_name": "SegFormer (nvidia/segformer-b0-finetuned-ade-512-512)",
            "architecture": "Hierarchical Vision Transformer",
            "benchmark_dataset": "ADE20K (150 Scene Classes)",
            "real_model_inference": True,
            "dimensions": {"width": orig_w, "height": orig_h},
            "gsd_m": gsd_m,
            "flood_water_pct": flood_water_pct,
            "flood_water_area_m2": flood_water_area_m2,
            "water_confidence": water_conf,
            "sky_pct": sky_pct,
            "sky_confidence": sky_conf,
            "is_oblique_view": is_oblique,
            "class_breakdown": water_breakdown,
            "classes_mapped": {
                "sky_id": self.sky_class_id,
                "water_ids": self.water_class_ids
            },
            "segmentation_mask_b64": mask_b64,
            "raw_mask": raw_mask,
            "summary": (
                f"SegFormer ADE20K detected {flood_water_pct}% water/flood ({flood_water_area_m2} m²) "
                f"with {round(water_conf * 100, 1)}% confidence. Sky coverage: {sky_pct}% ({round(sky_conf * 100, 1)}% confidence)."
            )
        }

    def _load_pil_image(self, input_data: Any) -> Image.Image:
        """Utility to convert ndarray, b64, or Image to PIL RGB Image."""
        if isinstance(input_data, Image.Image):
            return input_data.convert("RGB")
        elif isinstance(input_data, np.ndarray):
            if input_data.dtype != np.uint8:
                input_data = (np.clip(input_data, 0.0, 1.0) * 255.0).astype(np.uint8)
            return Image.fromarray(input_data).convert("RGB")
        elif isinstance(input_data, str):
            if input_data.startswith("data:image"):
                base64_data = input_data.split(",", 1)[1]
                image_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            else:
                return Image.open(input_data).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(input_data)}")


# Global Singleton Instance
segformer_water_detector = SegFormerWaterDetector()
