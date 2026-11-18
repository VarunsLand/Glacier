"""
find_benchmark_satellite_imagery.py
Search for satellite imagery matching the coordinates of the 4 benchmark glaciers:
- Rhone: (46.56, 8.38)
- Great Aletsch: (46.50, 8.03)
- Unteraar: (46.57, 8.19)
- Gorner: (45.97, 7.80)
"""
import sys
from pathlib import Path
import json
import numpy as np

ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")

# 1. Check DL4GAM patches to see if any have coordinates in the NetCDF attributes
def check_dl4gam():
    import xarray as xr
    extracted = ROOT / "datasets" / "dl4gam" / "extracted"
    print("Checking DL4GAM extracted patches for geospatial metadata...")
    nc_files = list(extracted.rglob("*.nc"))[:50]
    for nc in nc_files:
        try:
            ds = xr.open_dataset(nc)
            attrs = dict(ds.attrs)
            coords = list(ds.coords.keys())
            vars = list(ds.data_vars.keys())
            print(f"Sample {nc.name}:")
            print(f"  Coords: {coords}")
            print(f"  Attrs: {list(attrs.keys())[:10]}")
            if "lat" in coords or "latitude" in coords or "x" in coords:
                print(f"  Spatial coords: {[c for c in coords if c in ('lat','lon','x','y')]}")
            break
        except Exception as e:
            continue

# 2. Check if there are other files in datasets/objective4 or datasets/
def check_objective4():
    print("\nChecking datasets/objective4/...")
    obj4 = ROOT / "datasets" / "objective4"
    if obj4.exists():
        for f in obj4.rglob("*"):
            if f.is_file():
                print(f"  {f.relative_to(obj4)} ({f.stat().st_size:,} bytes)")

# 3. Check outputs/figures for publication figures
def check_figures():
    print("\nChecking outputs/figures for benchmark figures...")
    figs = ROOT / "outputs" / "figures"
    for f in sorted(figs.glob("glims_boundary_*.png")):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")

if __name__ == "__main__":
    check_dl4gam()
    check_objective4()
    check_figures()
