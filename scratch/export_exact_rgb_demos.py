import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
sys.path.insert(0, str(ROOT))

from src.dataset import DL4GAMDataset

DEMO_DIR = ROOT / "outputs" / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

targets = [
    # 2 Glaciers
    ("demo_glacier_1_clean_ice.png", ROOT / "datasets/dl4gam/extracted/g_4367/g_4367_patch_0_xc_256_yc_256.nc"),
    ("demo_glacier_2_valley_ice.png", ROOT / "datasets/dl4gam/extracted/g_3560/g_3560_patch_0_xc_256_yc_256.nc"),
    # 2 Bedrock / Non-glacier
    ("demo_bedrock_1_alpine_ridge.png", ROOT / "datasets/dl4gam/extracted/g_2300/g_2300_patch_0_xc_158_yc_155.nc"),
    ("demo_bedrock_2_rocky_slope.png", ROOT / "datasets/dl4gam/extracted/g_2427/g_2427_patch_0_xc_142_yc_153.nc"),
]

for out_name, path in targets:
    if path.exists():
        raw_bands, mask, meta = DL4GAMDataset.read_netcdf_patch(path)
        rgb = np.stack([raw_bands[3], raw_bands[2], raw_bands[1]], axis=-1)
        p2, p98 = np.nanpercentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        out_path = DEMO_DIR / out_name
        img.save(out_path)
        print(f"Exported: {out_name} (glacier ratio: {meta['glacier_pixel_ratio']:.2%})", flush=True)

print("Export complete!", flush=True)
