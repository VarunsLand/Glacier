import sys
from pathlib import Path

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
sys.path.insert(0, str(ROOT))

from src.dataset import DL4GAMDataset

ds = DL4GAMDataset(root=ROOT, fold="fold_test")
print(f"Total test patches: {len(ds)}", flush=True)

for i in range(15):
    gid, p = ds.patch_list[i]
    _, m, meta = ds.read_netcdf_patch(p)
    print(f"[{i:02d}] GID: {gid} | File: {p.name} | Ratio: {meta['glacier_pixel_ratio']:.4f}", flush=True)
