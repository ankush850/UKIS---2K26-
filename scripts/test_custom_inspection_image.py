import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.aerial.custom_inspection import run_full_custom_inspection
from PIL import Image

p = 'data/real/houses-on-the-hill-beautiful-suburb-of-sydney-palm-beach-background-with-copy-space.webp'
res = run_full_custom_inspection(p)
print('Keys:', list(res.keys()))
for k in res:
    if k != 'segmentation':
        print(f'{k}: {res[k]}')
print('\nHazard Summary:')
for k, v in res['segmentation']['hazard_summary'].items():
    print(f'  {k}: {v}')
print('\nDistribution:')
for k, v in res['segmentation']['distribution'].items():
    if v['percentage'] > 0:
        print(f"  {k}: {v['percentage']}% ({v['area_m2']} m2)")
