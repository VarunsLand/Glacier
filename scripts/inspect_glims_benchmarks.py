"""
inspect_glims_benchmarks.py
Inspect the actual GLIMS shapefile records for the 4 benchmark glaciers:
Rhone, Great Aletsch, Unteraar, Gorner.
"""
import sys
from pathlib import Path
import pyogrio
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
SHP_PATH = ROOT / "datasets" / "glims" / "extracted" / "glims_download_52211" / "glims_polygons.shp"

BENCHMARKS = [
    ("Rhone", "G008379E46648N"),
    ("Great Aletsch", "G008032E46504N"),
    ("Unteraar", "G008187E46569N"),
    ("Gorner", "G007800E45965N"),
]

def main():
    print("Reading GLIMS shapefile schema...")
    meta = pyogrio.read_info(SHP_PATH)
    print(f"Driver: {meta['driver']}, CRS: {meta['crs']}, Features: {meta['features']}")

    for name, gid in BENCHMARKS:
        print(f"\n{'='*60}\n{name} ({gid})\n{'='*60}")
        where = f"glac_id = '{gid}'"
        gdf = pyogrio.read_dataframe(SHP_PATH, where=where)
        print(f"Total rows: {len(gdf)}")
        print(f"Columns: {list(gdf.columns)}")
        
        # Display summary of rows
        cols_to_show = [c for c in ['anlys_id', 'glac_name', 'src_date', 'line_type', 'area', 'db_area', 'primary_class'] if c in gdf.columns]
        df_summary = pd.DataFrame(gdf[cols_to_show])
        print(df_summary.to_string())

if __name__ == "__main__":
    main()
