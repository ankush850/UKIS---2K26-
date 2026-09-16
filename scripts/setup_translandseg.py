import os
import shutil
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
src_dir = WORKSPACE / "scratch" / "TransLandSeg"
dst_dir = WORKSPACE / "backend" / "aerial" / "translandseg_models"

os.makedirs(dst_dir / "mmseg" / "models" / "sam", exist_ok=True)

# Copy config
shutil.copy(src_dir / "translandseg.yaml", WORKSPACE / "backend" / "aerial" / "translandseg.yaml")

# Copy core model files
shutil.copy(src_dir / "models" / "models.py", dst_dir / "models.py")
shutil.copy(src_dir / "models" / "sam.py", dst_dir / "sam.py")
shutil.copy(src_dir / "models" / "iou_loss.py", dst_dir / "iou_loss.py")
shutil.copy(src_dir / "models" / "__init__.py", dst_dir / "__init__.py")

# Create mmseg init files
with open(dst_dir / "mmseg" / "__init__.py", "w") as f:
    f.write('__version__ = "1.0.0"\n')

with open(dst_dir / "mmseg" / "models" / "__init__.py", "w") as f:
    f.write('__all__ = []\n')

# Copy SAM backbone files
src_sam = src_dir / "models" / "mmseg" / "models" / "sam"
dst_sam = dst_dir / "mmseg" / "models" / "sam"
for f in os.listdir(src_sam):
    if f.endswith(".py"):
        shutil.copy(src_sam / f, dst_sam / f)

print("TransLandSeg models copied to backend/aerial/translandseg_models successfully!")
