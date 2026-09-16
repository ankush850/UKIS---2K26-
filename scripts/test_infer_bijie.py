import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import yaml
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
trans_dir = WORKSPACE_ROOT / "scratch" / "TransLandSeg"
sys.path.insert(0, str(trans_dir))

import models

with open(trans_dir / "translandseg.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print("Creating SAM model...")
model = models.make(config['model'])

ckpt_path = WORKSPACE_ROOT / "checkpoint" / "Bijie.pth.tar"
print("Loading checkpoint from:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=True)
model.eval()
print("Model loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_images = [
    ("meppadi", "data/real/meppadi-india-an-aerial-view-shows-the-site-of-a-landslide-on-july-31-2024-in-chooralmala (1).webp"),
    ("sydney_houses", "data/real/houses-on-the-hill-beautiful-suburb-of-sydney-palm-beach-background-with-copy-space.webp"),
    ("mountains_landslide", "data/real/mountains-landslides-due-heavy-rain-260nw-1822899695.webp")
]

for name, rel_path in test_images:
    p = WORKSPACE_ROOT / rel_path
    if not p.exists():
        print(f"Missing file: {p}")
        continue
    img = Image.open(p).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    print(f"\n--- Testing {name} ({img.size}) ---")
    t0 = time.time()
    with torch.no_grad():
        pred_mask = model.infer(tensor) # Shape [1, 1, 1024, 1024]
    elapsed = time.time() - t0
    print(f"Inference time: {elapsed:.2f}s, pred_mask shape: {pred_mask.shape}")
    probs = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
    binary_mask = (probs > 0.5).astype(np.uint8)
    landslide_pct = (binary_mask.sum() / binary_mask.size) * 100.0
    print(f"Probabilities: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
    print(f"Landslide Area %: {landslide_pct:.2f}%")
