import rasterio
from pathlib import Path
ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
tif_p = ROOT / "datasets" / "objective4" / "sentinel2_rhone" / "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
with rasterio.open(tif_p) as src:
    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("Bounds:", src.bounds)
    print("Width, Height:", src.width, src.height)
    print("Count:", src.count)
