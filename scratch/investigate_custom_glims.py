import os
from pathlib import Path
import pyogrio
import pandas as pd
import geopandas as gpd
import pyproj
from shapely.ops import transform, unary_union

shp_path = Path("datasets/glims/extracted/glims_download_52211/glims_polygons.shp")
where_clause = "glac_id = 'G010725E46789N' AND line_type = 'glac_bound'"
df = pyogrio.read_dataframe(shp_path, where=where_clause)
print(f"Total matching polygons: {len(df)}")
for idx, r in df.iterrows():
    print(dict(
        anlys_id=r.get("anlys_id"),
        glac_id=r.get("glac_id"),
        glac_name=r.get("glac_name"),
        src_date=str(r.get("src_date"))[:10],
        db_area=r.get("db_area"),
        geom_type=r.get("geometry").geom_type
    ))

# Check the two analysis dates from glims_retreat_assessment_validated.csv
# earlier_analysis_id: 1031480, later_analysis_id: 709596
# earlier_date: 2000-08-22, later_date: 2016-09-29
e_df = df[df["anlys_id"] == 1031480]
l_df = df[df["anlys_id"] == 709596]
print("Earlier polygon count:", len(e_df))
print("Later polygon count:", len(l_df))

geom_e = unary_union(e_df.geometry.values)
geom_l = unary_union(l_df.geometry.values)

# Let's project to EPSG:3035 (European Equal-Area) and EPSG:32632 (UTM 32N)
for crs_name in ["EPSG:3035", "EPSG:32632"]:
    project = pyproj.Transformer.from_crs("EPSG:4326", crs_name, always_xy=True).transform
    geom_e_proj = transform(project, geom_e)
    geom_l_proj = transform(project, geom_l)

    area_e = geom_e_proj.area / 1e6
    area_l = geom_l_proj.area / 1e6
    net_loss = area_e - area_l
    pct_loss = (net_loss / area_e) * 100.0

    print(f"\nProjection {crs_name}:")
    print(f"  Earlier Area ({str(e_df.iloc[0]['src_date'])[:10]}): {area_e:.6f} km²")
    print(f"  Later Area ({str(l_df.iloc[0]['src_date'])[:10]}): {area_l:.6f} km²")
    print(f"  Net Loss (unrounded): {net_loss:.6f} km²")
    print(f"  Net Loss (rounded 3 dec): {net_loss:.3f} km²")
    print(f"  Percentage Loss (unrounded): {pct_loss:.4f}%")
    print(f"  Percentage Loss (rounded 2 dec): {pct_loss:.2f}%")
