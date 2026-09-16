import os
import sys
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure workspace and flood_vision paths are available
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
fv_dir = str(WORKSPACE_ROOT / "sample_data" / "flood_vision")
if fv_dir not in sys.path:
    sys.path.insert(0, fv_dir)

from backend.aerial.segmentation import drone_segmentation_engine, DroneSegmentationEngine, AERIAL_CLASSES
from models.deeplab_model import get_deeplab_model

def checkpoint_fingerprint(path):
    with open(path, "rb") as f:
        data = f.read()
        return hashlib.md5(data).hexdigest()[:12], len(data)

def check_model_eval_state(model):
    is_training = model.training
    active_dropouts = []
    active_bns = []
    for name, m in model.named_modules():
        if m.training:
            if "dropout" in m.__class__.__name__.lower():
                active_dropouts.append(name)
            if "batchnorm" in m.__class__.__name__.lower():
                active_bns.append(name)
    return {
        "is_training": is_training,
        "active_dropouts": active_dropouts,
        "active_bns": active_bns
    }

def main():
    print("=" * 80)
    print("STEP 1: SEGMENTATION MODEL DIAGNOSTIC SCRIPT")
    print("=" * 80)

    # 1. Checkpoint Check
    ckpt_path = WORKSPACE_ROOT / "sample_data" / "flood_vision" / "checkpoint_deeplab_4class.pth"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint does not exist at {ckpt_path}")
        return

    md5_hash, size_bytes = checkpoint_fingerprint(ckpt_path)
    print(f"Checkpoint Path : {ckpt_path}")
    print(f"File Size       : {size_bytes:,} bytes ({size_bytes / 1024 / 1024:.2f} MB)")
    print(f"MD5 Fingerprint : {md5_hash}")

    # 2. Load Model Exactly As In Backend
    device = torch.device("cpu")
    model = get_deeplab_model(num_classes=4, pretrained=False)
    weights = torch.load(ckpt_path, map_location=device)
    
    # Check weights structure
    if isinstance(weights, dict):
        if "model_state_dict" in weights:
            state_dict = weights["model_state_dict"]
            print("Note: weights is a dict with 'model_state_dict'")
        elif "state_dict" in weights:
            state_dict = weights["state_dict"]
            print("Note: weights is a dict with 'state_dict'")
        else:
            state_dict = weights
            print(f"Note: weights is a raw state_dict with {len(weights)} keys")
    else:
        state_dict = weights

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict keys: Matched={len(state_dict) - len(unexpected)}, Missing={len(missing)}, Unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"First 5 missing keys: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"First 5 unexpected keys: {unexpected[:5]}")

    model.to(device).eval()

    # Eval state check
    eval_state = check_model_eval_state(model)
    print(f"Model eval() mode confirmed: training={eval_state['is_training']}, "
          f"active_dropouts={len(eval_state['active_dropouts'])}, active_bns={len(eval_state['active_bns'])}")

    # Engine status check
    print(f"drone_segmentation_engine.deeplab_model is loaded: {drone_segmentation_engine.deeplab_model is not None}")

    # 3. Define Test Inputs
    test_inputs = {}

    # (A) Real Flood Scenes
    flood_path_1 = WORKSPACE_ROOT / "data" / "aerial_demo" / "chamoli_flash_flood" / "post_disaster.jpg"
    if flood_path_1.exists():
        test_inputs["1. Real Flood (Chamoli)"] = Image.open(flood_path_1).convert("RGB")

    flood_path_2 = WORKSPACE_ROOT / "data" / "aerial_demo" / "kedarnath_inundation" / "post_disaster.jpg"
    if flood_path_2.exists():
        test_inputs["1b. Real Flood (Kedarnath)"] = Image.open(flood_path_2).convert("RGB")

    # (B) Normal Sunny Hillside Town
    hillside_path = WORKSPACE_ROOT / "data" / "real" / "houses-on-the-hill-beautiful-suburb-of-sydney-palm-beach-background-with-copy-space.webp"
    if hillside_path.exists():
        test_inputs["2. Hillside Town (Palm Beach)"] = Image.open(hillside_path).convert("RGB")

    # (C) Solid Color Images (White & Green)
    solid_white = Image.new("RGB", (512, 512), color=(255, 255, 255))
    test_inputs["3a. Solid White (512x512)"] = solid_white

    solid_green = Image.new("RGB", (512, 512), color=(34, 139, 34))
    test_inputs["3b. Solid Green (512x512)"] = solid_green

    # (D) Completely Different Random Photo / Graphic
    arch_path = WORKSPACE_ROOT / "Assests_and_Report-img" / "arch.png"
    if arch_path.exists():
        test_inputs["4. Architecture Diagram (arch.png)"] = Image.open(arch_path).convert("RGB")

    # (E) Additional Real Aerial Landslide Photo
    landslide_path = WORKSPACE_ROOT / "data" / "real" / "aerial-view-landslide-impacting-hillside-village-in-dense-forest.webp"
    if landslide_path.exists():
        test_inputs["5. Landslide Village (real photo)"] = Image.open(landslide_path).convert("RGB")

    # 4. Run Inference & Collect Detailed Statistics
    results = []

    CLASS_NAMES_4 = ["0: Background", "1: Bldg Flooded", "2: Road Flooded", "3: Water"]

    for name, img in test_inputs.items():
        print("\n" + "-" * 70)
        print(f"TEST INPUT: {name} (Original Size: {img.size})")

        # Step 4a: Preprocessing exactly as in backend
        resized_img = img.resize((512, 512), Image.Resampling.BILINEAR)
        img_np = np.array(resized_img).astype(np.float32) / 255.0
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        img_norm = (img_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # Step 4b: Model inference (raw logits)
        with torch.no_grad():
            logits = model(tensor) # Shape: [1, 4, 512, 512]
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy() # [4, 512, 512]
            pred_4cls = np.argmax(probs, axis=0) # [512, 512]
            logits_np = logits.squeeze(0).cpu().numpy() # [4, 512, 512]

        total_px = pred_4cls.size
        pct_4cls = [float(np.sum(pred_4cls == c) / total_px * 100.0) for c in range(4)]

        print("  Raw Logits per Channel [min, max, mean, std]:")
        for c in range(4):
            ch_logits = logits_np[c]
            print(f"    Class {CLASS_NAMES_4[c]:<17}: min={ch_logits.min():.3f}, max={ch_logits.max():.3f}, "
                  f"mean={ch_logits.mean():.3f}, std={ch_logits.std():.3f} | ArgMax Pct={pct_4cls[c]:.2f}%")

        # Step 4c: Run through the full backend drone_segmentation_engine.segment()
        app_res = drone_segmentation_engine.segment(img)
        app_dist = app_res["distribution"]
        app_hazard = app_res["hazard_summary"]

        results.append({
            "name": name,
            "logits_stats": [(logits_np[c].min(), logits_np[c].max(), logits_np[c].mean(), logits_np[c].std()) for c in range(4)],
            "raw_4cls_pct": pct_4cls,
            "app_water_pct": app_hazard["flooded_pct"],
            "app_debris_pct": app_hazard["debris_pct"],
            "app_bldg_dmg_pct": app_hazard["damaged_buildings_pct"],
            "app_road_blk_pct": app_hazard["blocked_roads_pct"],
            "app_safe_ground_pct": app_dist["non-flooded"]["percentage"],
            "app_model_name": app_hazard["model_name"]
        })

    # 5. Print Side-by-Side Comparison Table
    print("\n" + "=" * 95)
    print("SIDE-BY-SIDE COMPARISON TABLE")
    print("=" * 95)
    
    header = f"{'Input Image':<32} | {'Raw Water%':<10} | {'App Flood%':<10} | {'App Debris%':<11} | {'App Safe%':<9} | {'Bldg Dmg%':<9}"
    print(header)
    print("-" * 95)

    for r in results:
        line = (f"{r['name'][:32]:<32} | "
                f"{r['raw_4cls_pct'][3]:>9.2f}% | "
                f"{r['app_water_pct']:>9.2f}% | "
                f"{r['app_debris_pct']:>10.2f}% | "
                f"{r['app_safe_ground_pct']:>8.2f}% | "
                f"{r['app_bldg_dmg_pct']:>8.2f}%")
        print(line)

    print("=" * 95)

if __name__ == "__main__":
    main()
