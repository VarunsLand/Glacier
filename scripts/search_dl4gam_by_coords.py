"""
search_dl4gam_by_coords.py
Check if any DL4GAM patches match the coordinates of Great Aletsch, Unteraar, or Gorner.
"""
from pathlib import Path
import sys
ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
sys.path.insert(0, str(ROOT))
import xarray as xr
import pyproj
extracted = ROOT / "datasets" / "dl4gam" / "extracted"

BENCHMARKS = {
    "Rhone": {"lat": 46.56, "lon": 8.38, "utm_x": 452400, "utm_y": 5157000, "crs": "EPSG:32632"},
    "Great Aletsch": {"lat": 46.50, "lon": 8.03, "utm_x": 425600, "utm_y": 5150500, "crs": "EPSG:32632"},
    "Unteraar": {"lat": 46.57, "lon": 8.19, "utm_x": 437600, "utm_y": 5157800, "crs": "EPSG:32632"},
    "Gorner": {"lat": 45.97, "lon": 7.80, "utm_x": 407000, "utm_y": 5091000, "crs": "EPSG:32632"},
}

def main():
    print("Searching DL4GAM patches...")
    # Sample 100 patches across different directories
    patch_dirs = list(extracted.glob("g_*"))
    print(f"Total glacier folders: {len(patch_dirs)}")

    # Check the 5 featured patches in visual_analysis.py
    from src.visual_analysis import get_featured_dl4gam_patches
    featured = get_featured_dl4gam_patches()
    for fp in featured:
        nc_p = fp["path"]
        if nc_p.exists():
            ds = xr.open_dataset(nc_p)
            wkt = ds.spatial_ref.attrs.get("crs_wkt", "")
            epsg = "32632" if "UTM zone 32N" in wkt else ("32633" if "UTM zone 33N" in wkt else "Unknown")
            x_min, x_max = float(ds.x.min()), float(ds.x.max())
            y_min, y_max = float(ds.y.min()), float(ds.y.max())
            print(f"Featured {fp['glacier_id']} ({fp['title']}):")
            print(f"  EPSG: {epsg}, X: [{x_min:.1f}, {x_max:.1f}], Y: [{y_min:.1f}, {y_max:.1f}]")
            print(f"  Area attr: {ds.attrs.get('glacier_area')} km²")
            # Convert to lat/lon
            if epsg != "Unknown":
                transformer = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
                lon_min, lat_min = transformer.transform(x_min, y_min)
                lon_max, lat_max = transformer.transform(x_max, y_max)
                print(f"  Lat/Lon: Lat [{lat_min:.4f}, {lat_max:.4f}], Lon [{lon_min:.4f}, {lon_max:.4f}]")

if __name__ == "__main__":
    main()
