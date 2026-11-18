import xarray as xr
from pathlib import Path
ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
nc = ROOT / "datasets" / "dl4gam" / "extracted" / "g_4367" / "g_4367_patch_0_xc_256_yc_256.nc"
ds = xr.open_dataset(nc)
for k, v in ds.attrs.items():
    print(f"{k}: {v}")
print("Coords:", {c: ds.coords[c].values.shape for c in ds.coords})
print("Vars:", {v: ds.data_vars[v].values.shape for v in ds.data_vars})
