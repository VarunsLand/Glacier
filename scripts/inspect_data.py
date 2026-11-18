from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

ROOT = Path.home() / "Desktop" / "GlacierResearch"

print("\n========== DL4GAM BAND INSPECTION ==========")

files = sorted(
    (ROOT / "datasets/dl4gam/extracted").rglob("*.nc")
)

with xr.open_dataset(files[0]) as ds:

    print("\nBand coordinates:")
    print(ds["band"].values)

    print("\nBand coordinate attributes:")
    print(ds["band"].attrs)

    print("\nSpectral band statistics:")

    for i in range(ds.sizes["band"]):
        a = ds["band_data"].isel(band=i).values
        print(
            f"Band {i}: min={np.nanmin(a):.4f}, "
            f"max={np.nanmax(a):.4f}, "
            f"mean={np.nanmean(a):.4f}"
        )

    print("\nMask statistics:")

    names = [
        "mask_all_g_id",
        "mask_crt_g",
        "mask_crt_g_b-20",
        "mask_crt_g_b-10",
        "mask_crt_g_b10",
        "mask_crt_g_b20",
        "mask_crt_g_b50",
        "mask_debris"
    ]

    for name in names:
        a = ds[name].values
        values, counts = np.unique(
            a[np.isfinite(a)],
            return_counts=True
        )

        print("\n", name)
        print("Unique values:", len(values))
        print("Most frequent:", sorted(
            zip(values.tolist(), counts.tolist()),
            key=lambda x: -x[1]
        )[:10])

    print("\nSpatial reference:")
    print(ds["spatial_ref"].attrs)

print("\n========== FOLD INSPECTION ==========")

csv_path = next(
    (ROOT / "datasets/dl4gam").rglob(
        "map_all_splits_all_folds.csv"
    )
)

df = pd.read_csv(csv_path)

for col in [f"split_{i}" for i in range(1, 6)]:
    print("\n", col)
    print(df[col].value_counts().to_dict())

print("\nTotal glacier IDs:", len(df))

print("\n========== LANDSAT MASK INSPECTION ==========")

label_files = sorted(
    (ROOT / "datasets/landsat/labels").glob("*.png")
)

for path in label_files[:3]:
    a = np.array(Image.open(path))
    print(
        path.name,
        "shape:", a.shape,
        "values:", np.unique(a, return_counts=True)
    )

print("\n========== INSPECTION COMPLETE ==========")
