import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
sys.path.insert(0, str(ROOT))

from src.dataset import DL4GAMDataset

DEMO_DIR = ROOT / "outputs" / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

ds_test = DL4GAMDataset(root=ROOT, fold="fold_test")

glacier_found = 0
bedrock_found = 0

print(f"Scanning test patches to export demo RGB samples...")
for gid, p_path in ds_test.patch_list:
    try:
        raw_bands, mask, meta = ds_test.read_netcdf_patch(p_path)
        ratio = meta["glacier_pixel_ratio"]

        # High glacier ice sample (ratio > 0.60)
        if ratio > 0.60 and glacier_found < 2:
            rgb = np.stack([raw_bands[3], raw_bands[2], raw_bands[1]], axis=-1)
            p2, p98 = np.nanpercentile(rgb, (2, 98))
            rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
            img = Image.fromarray((rgb * 255).astype(np.uint8))
            out_file = DEMO_DIR / f"demo_glacier_sample_{glacier_found+1}_{gid}.png"
            img.save(out_file)
            print(f"Saved glacier demo: {out_file.name} (ratio: {ratio:.1%})")
            glacier_found += 1

        # Non-glacier bedrock sample (ratio < 0.005)
        elif ratio < 0.005 and bedrock_found < 2:
            rgb = np.stack([raw_bands[3], raw_bands[2], raw_bands[1]], axis=-1)
            p2, p98 = np.nanpercentile(rgb, (2, 98))
            rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
            img = Image.fromarray((rgb * 255).astype(np.uint8))
            out_file = DEMO_DIR / f"demo_bedrock_sample_{bedrock_found+1}_{gid}.png"
            img.save(out_file)
            print(f"Saved bedrock demo: {out_file.name} (ratio: {ratio:.1%})")
            bedrock_found += 1

        if glacier_found >= 2 and bedrock_found >= 2:
            break
    except Exception as e:
        continue

print("Demo samples generation finished!")
