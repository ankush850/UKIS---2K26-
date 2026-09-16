import sys
from pathlib import Path
import yaml
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
trans_dir = WORKSPACE_ROOT / "scratch" / "TransLandSeg"
sys.path.insert(0, str(trans_dir))

import models

with open(trans_dir / "translandseg.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print("Creating SAM model...")
model = models.make(config['model'])
print("SAM model instantiated successfully!")

ckpt_path = WORKSPACE_ROOT / "checkpoint" / "Bijie.pth.tar"
print("Loading checkpoint from:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Key match check: Total in state_dict={len(state_dict)}, Missing={len(missing)}, Unexpected={len(unexpected)}")
if missing:
    print("Sample missing:", missing[:5])
if unexpected:
    print("Sample unexpected:", unexpected[:5])
