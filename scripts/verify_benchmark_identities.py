"""Re-calculate exact geometric areas and spatial differences in EPSG:32632.
"""
from pathlib import Path
import pyogrio
import geopandas as gpd
from shapely.ops import unary_union
import pyproj
from shapely.ops import transform

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
SHP_PATH = ROOT / "datasets" / "glims" / "extracted" / "glims_download_52211" / "glims_polygons.shp"

benchmarks = [
    {"id": "G008379E46648N", "name": "Rhone", "crs": "EPSG:32632", "e_date": "1973-09-01", "l_date": "2015-08-29"},
    {"id": "G008032E46504N", "name": "Great Aletsch", "crs": "EPSG:32632", "e_date": "1973-09-01", "l_date": "2015-08-29"},
    {"id": "G008187E46569N", "name": "Unteraar", "crs": "EPSG:32632", "e_date": "1973-09-01", "l_date": "2015-08-29"},
]

for b in benchmarks:
    gid = b["id"]
    name = b["name"]
    where_clause = f"glac_id = '{gid}' AND line_type = 'glac_bound'"
    gdf = pyogrio.read_dataframe(SHP_PATH, where=where_clause)
    
    # Filter earlier and later by src_date
    gdf["_sdate"] = gdf["src_date"].str[:10]
    e_df = gdf[gdf["_sdate"] == b["e_date"]]
    l_df = gdf[gdf["_sdate"] == b["l_date"]]
    
    # Unary union
    geom_e = unary_union(e_df.geometry.values)
    geom_l = unary_union(l_df.geometry.values)
    
    # Reproject to EPSG:32632
    project = pyproj.Transformer.from_crs("EPSG:4326", b["crs"], always_xy=True).transform
    geom_e_utm = transform(project, geom_e)
    geom_l_utm = transform(project, geom_l)
    
    area_e = geom_e_utm.area / 1e6
    area_l = geom_l_utm.area / 1e6
    
    # Retained, Lost, Gained
    retained = geom_e_utm.intersection(geom_l_utm)
    lost = geom_e_utm.difference(geom_l_utm)
    gained = geom_l_utm.difference(geom_e_utm)
    
    area_ret = retained.area / 1e6
    area_lost = lost.area / 1e6
    area_gain = gained.area / 1e6
    
    net_loss = area_lost - area_gain
    net_loss_direct = area_e - area_l
    pct_loss = (net_loss / area_e) * 100.0
    
    # Elapsed years: from 1973-09-01 to 2015-08-29
    # In exact days / 365.25:
    import pandas as pd
    d1 = pd.to_datetime(b["e_date"])
    d2 = pd.to_datetime(b["l_date"])
    elapsed_years_exact = (d2 - d1).days / 365.2425
    elapsed_years_integer = 2015 - 1973
    annual_rate_exact = net_loss / elapsed_years_exact
    annual_rate_int = net_loss / elapsed_years_integer
    
    print(f"\n=======================================================")
    print(f"BENCHMARK: {name} ({gid})")
    print(f"=======================================================")
    print(f"Earlier Date: {b['e_date']} (polygons: {len(e_df)})")
    print(f"Later Date:   {b['l_date']} (polygons: {len(l_df)})")
    print(f"Earlier Area (A_earlier): {area_e:.6f} km² -> {area_e:.3f}")
    print(f"Modern Area  (A_later):   {area_l:.6f} km² -> {area_l:.3f}")
    print(f"Retained Area (A_ret):    {area_ret:.6f} km² -> {area_ret:.3f}")
    print(f"Lost Area     (A_lost):   {area_lost:.6f} km² -> {area_lost:.3f}")
    print(f"Gained Area   (A_gain):   {area_gain:.6f} km² -> {area_gain:.3f}")
    print(f"Net Loss (A_lost - A_gain): {net_loss:.6f} km² -> {net_loss:.3f}")
    print(f"Direct Net (A_e - A_l):     {net_loss_direct:.6f} km² -> {net_loss_direct:.3f}")
    print(f"Percentage Loss:            {pct_loss:.4f}% -> {pct_loss:.2f}%")
    print(f"Elapsed Years (integer): {elapsed_years_integer} yr | Annual Rate: {annual_rate_int:.4f} km²/yr -> {annual_rate_int:.3f}")
    print(f"Elapsed Years (exact):   {elapsed_years_exact:.2f} yr | Annual Rate: {annual_rate_exact:.4f} km²/yr -> {annual_rate_exact:.3f}")
    
    # Check identities
    id1_diff = abs(area_e - (area_ret + area_lost))
    id2_diff = abs(area_l - (area_ret + area_gain))
    id3_diff = abs(net_loss - (area_lost - area_gain))
    id4_diff = abs(net_loss - net_loss_direct)
    print(f"\nIdentity Checks:")
    print(f"  1. A_earlier = A_ret + A_lost : diff = {id1_diff:.2e} ({'EXACT' if id1_diff < 1e-9 else 'DIFF'})")
    print(f"  2. A_later   = A_ret + A_gain : diff = {id2_diff:.2e} ({'EXACT' if id2_diff < 1e-9 else 'DIFF'})")
    print(f"  3. Net Loss  = A_lost - A_gain: diff = {id3_diff:.2e} ({'EXACT' if id3_diff < 1e-9 else 'DIFF'})")
    print(f"  4. Net Loss  = A_e - A_l      : diff = {id4_diff:.2e} ({'EXACT' if id4_diff < 1e-9 else 'DIFF'})")
