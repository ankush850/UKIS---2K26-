"""
SegFormer B0: General-Scene Water & Sky Disambiguator (ADE20K Pretrained).
UKIS Hackathon 2026 - Problem P-008 | DMMC Uttarakhand.

Standalone runnable implementation demonstrating:
1. Loading SegFormer B0 model from Hugging Face / Transformers
2. Forward inference on RGB aerial image
3. Multi-class Softmax probability distribution extraction
4. Dedicated separation of Sky (Class 2) vs Water (Classes 21, 26, 60, 128)
5. Calculating per-class coverage and mean softmax confidence
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

try:
    from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ADE20K 150-class indices of operational interest
ADE20K_SKY_CLASS = 2
ADE20K_WATER_CLASSES = [21, 26, 60, 128]  # water, sea, river, lake


def run_segformer_demo():
    print("=" * 70)
    print("🛰️ SegFormer B0: Water & Sky Horizon Disambiguation Demo")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running on: {device}")

    if not TRANSFORMERS_AVAILABLE:
        print("[Error] transformers package is not installed. Please install it with: pip install transformers")
        return

    model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
    print(f"[Model] Loading pretrained checkpoint: {model_id}")

    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
        model.eval()
        print("[Model] Successfully loaded SegFormer B0 architecture.")
    except Exception as e:
        print(f"[Model] Could not load online checkpoint ({e}). Exiting demo.")
        return

    # Create a synthetic 512x512 RGB test image
    # Simulate an image with an upper blue sky region and a lower green/brown terrain
    img_np = np.zeros((512, 512, 3), dtype=np.uint8)
    img_np[:256, :] = [135, 206, 235]  # Sky Blue upper half
    img_np[256:, :] = [34, 139, 34]    # Forest Green lower half
    test_image = Image.fromarray(img_np)
    print("[Input] Created synthetic 512x512 image (Upper: Blue Sky, Lower: Terrain)")

    # Preprocess
    inputs = processor(images=test_image, return_tensors="pt").to(device)

    # Forward Pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, 150, 128, 128)

        # Upscale logits to match original image dimensions
        upscaled_logits = F.interpolate(
            logits,
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        )
        # Compute Softmax probabilities across all 150 classes
        probs = F.softmax(upscaled_logits, dim=1)  # shape: (1, 150, 512, 512)

    probs_np = probs.squeeze(0).cpu().numpy()  # (150, 512, 512)

    # Extract Sky channel (Class 2)
    sky_probs = probs_np[ADE20K_SKY_CLASS]
    sky_mask = (sky_probs >= 0.35).astype(np.uint8)
    sky_pct = (np.sum(sky_mask) / sky_mask.size) * 100.0
    sky_conf = float(np.mean(sky_probs[sky_mask > 0])) if np.sum(sky_mask) > 0 else 0.0

    # Extract Water channels (21: water, 26: sea, 60: river, 128: lake)
    water_probs = np.max([probs_np[c] for c in ADE20K_WATER_CLASSES], axis=0)
    water_mask = (water_probs >= 0.35).astype(np.uint8)
    water_pct = (np.sum(water_mask) / water_mask.size) * 100.0
    water_conf = float(np.mean(water_probs[water_mask > 0])) if np.sum(water_mask) > 0 else 0.0

    print("-" * 70)
    print("📊 Softmax Probability Results:")
    print(f" - Sky Coverage Detected   : {sky_pct:.2f}% (Confidence: {sky_conf * 100:.1f}%)")
    print(f" - Water Coverage Detected : {water_pct:.2f}% (Confidence: {water_conf * 100:.1f}%)")

    # Simulate Discrepancy Detection against a hypothetical nadir flood model
    hypothetical_floodnet_water = 66.66  # e.g., FloodNet misclassifying the blue sky
    models_disagree = (hypothetical_floodnet_water >= 10.0 and water_pct < 3.0 and sky_pct >= 10.0)

    print("-" * 70)
    print(f"🤖 Tri-Model Consensus Simulation:")
    print(f" - Hypothetical FloodNet Reading : {hypothetical_floodnet_water:.2f}% Water")
    print(f" - SegFormer Sky Reading         : {sky_pct:.2f}% Sky")
    print(f" - Discrepancy Detected (Flag)   : {models_disagree}")
    if models_disagree:
        print("   >>> ACTION: Suppressed false flood positive; identified blue horizon as Sky!")
    print("=" * 70)
    print("✅ SegFormer B0 demo completed successfully!")


if __name__ == "__main__":
    run_segformer_demo()
