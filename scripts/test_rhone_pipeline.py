import numpy as np
import rasterio
from rasterio.io import MemoryFile
from pathlib import Path

# Load file bytes
tif_path = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch/datasets/objective4/sentinel2_rhone/rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif")
file_bytes = tif_path.read_bytes()

with MemoryFile(file_bytes) as memfile:
    with memfile.open() as ds:
        arr = ds.read().astype(np.float32)
        count = ds.count
        w, h = ds.width, ds.height
        crs = ds.crs
        transform = ds.transform
        res = ds.res
        nodatas = ds.nodatavals
        descriptions = ds.descriptions
        tags = ds.tags()

print(f"Loaded: count={count}, dims={w}x{h}, crs={crs}, res={res}")
print(f"Descriptions: {descriptions}")

# Resolve band indices
band_names = list(descriptions)
print("Band names:", band_names)

def resolve_s2_band_indices(b_names):
    res = {"blue": None, "green": None, "red": None, "nir": None, "swir1": None}
    lower = [b.lower() for b in b_names]
    for i, name in enumerate(lower):
        if name in ["b02", "b2", "blue"]:
            res["blue"] = i
        elif name in ["b03", "b3", "green"]:
            res["green"] = i
        elif name in ["b04", "b4", "red"]:
            res["red"] = i
        elif name in ["b08", "b8", "b8a", "nir"]:
            res["nir"] = i
        elif name in ["b11", "b11a", "swir1", "swir 1", "swir"]:
            res["swir1"] = i
    return res

indices = resolve_s2_band_indices(band_names)
print("Resolved indices:", indices)

assert indices["blue"] == 0, f"Expected blue=0, got {indices['blue']}"
assert indices["green"] == 1, f"Expected green=1, got {indices['green']}"
assert indices["red"] == 2, f"Expected red=2, got {indices['red']}"
assert indices["nir"] == 3, f"Expected nir=3, got {indices['nir']}"
assert indices["swir1"] == 4, f"Expected swir1=4, got {indices['swir1']}"

# Test RGB generation: Red = B04 (idx 2), Green = B03 (idx 1), Blue = B02 (idx 0)
r = arr[indices["red"]]
g = arr[indices["green"]]
b = arr[indices["blue"]]
raw_rgb = np.stack([r, g, b], axis=-1)
p1 = np.nanpercentile(raw_rgb, 2.0)
p2 = np.nanpercentile(raw_rgb, 98.0)
rgb_stretched = np.clip((raw_rgb - p1) / max(p2 - p1, 1e-6), 0.0, 1.0)
rgb_uint8 = (rgb_stretched * 255.0).astype(np.uint8)
print(f"RGB uint8 shape: {rgb_uint8.shape}, dtype: {rgb_uint8.dtype}, mean: {rgb_uint8.mean():.2f}")

# Test SWIR generation: Red = B11 (idx 4), Green = B08 (idx 3), Blue = B04 (idx 2)
swir_r = arr[indices["swir1"]]
swir_g = arr[indices["nir"]]
swir_b = arr[indices["red"]]
raw_swir = np.stack([swir_r, swir_g, swir_b], axis=-1)
p1_s = np.nanpercentile(raw_swir, 2.0)
p2_s = np.nanpercentile(raw_swir, 98.0)
swir_stretched = np.clip((raw_swir - p1_s) / max(p2_s - p1_s, 1e-6), 0.0, 1.0)
swir_uint8 = (swir_stretched * 255.0).astype(np.uint8)
print(f"SWIR uint8 shape: {swir_uint8.shape}, dtype: {swir_uint8.dtype}, mean: {swir_uint8.mean():.2f}")

# Test NDSI: (B03 - B11) / (B03 + B11)
denom = g + swir_r
valid = (denom > 1e-6) & np.isfinite(g) & np.isfinite(swir_r)
ndsi = np.zeros_like(g, dtype=np.float32)
ndsi[valid] = (g[valid] - swir_r[valid]) / denom[valid]
ndsi[~valid] = np.nan
ndsi = np.clip(ndsi, -1.0, 1.0)
mask = (ndsi >= 0.40).astype(np.uint8)
print(f"NDSI min: {np.nanmin(ndsi):.3f}, max: {np.nanmax(ndsi):.3f}, mean: {np.nanmean(ndsi):.3f}")
print(f"Snow/ice pixels at 0.40 threshold: {mask.sum():,} ({mask.mean()*100:.2f}%)")
print("ALL TESTS PASSED SUCCESSFULLY!")
