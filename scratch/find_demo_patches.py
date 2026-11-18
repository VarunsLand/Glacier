import sys
from pathlib import Path

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
sys.path.insert(0, str(ROOT))

from src.dataset import DL4GAMDataset

ds = DL4GAMDataset(root=ROOT, fold="fold_test")
print("Total test patches:", len(ds))

ratios = []
for i in range(min(120, len(ds))):
    gid, p = ds.patch_list[i]
    _, m, meta = ds.read_netcdf_patch(p)
    ratios.append((gid, p.name, str(p), meta["glacier_pixel_ratio"]))

ratios.sort(key=lambda x: x[3])

print("\n--- Low Ratio (Bedrock / Edge) ---")
for r in ratios[:4]:
    print(f"ID: {r[0]}, File: {r[1]}, Ratio: {r[3]:.4f}")

print("\n--- Medium Ratio (Mixed Alpine Terrain) ---")
mid = len(ratios) // 2
for r in ratios[mid:mid+4]:
    print(f"ID: {r[0]}, File: {r[1]}, Ratio: {r[3]:.4f}")

print("\n--- High Ratio (Large Glacier Body) ---")
for r in ratios[-4:]:
    print(f"ID: {r[0]}, File: {r[1]}, Ratio: {r[3]:.4f}")
