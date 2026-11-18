import xarray as xr
from pathlib import Path
ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
nc = ROOT / "datasets" / "dl4gam" / "extracted" / "g_4367" / "g_4367_patch_0_xc_256_yc_256.nc"
ds = xr.open_dataset(nc)
print("spatial_ref attrs:", dict(ds.spatial_ref.attrs))
print("x range:", ds.x.values.min(), ds.x.values.max())
print("y range:", ds.y.values.min(), ds.y.values.max())
