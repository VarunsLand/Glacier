import pyogrio
import pyproj
from pathlib import Path
from shapely.ops import transform, unary_union

ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
SHP_PATH = ROOT / "datasets" / "glims" / "extracted" / "glims_download_52211" / "glims_polygons.shp"

gdf = pyogrio.read_dataframe(SHP_PATH, where="glac_id = 'G008379E46648N' AND line_type = 'glac_bound'")
gdf["_date"] = gdf["src_date"].str[:10]

project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform

e_df = gdf[gdf["_date"] == "1973-09-01"]
l_df = gdf[gdf["_date"] == "2015-08-29"]

geom_e_utm = transform(project, unary_union(e_df.geometry.values))
geom_l_utm = transform(project, unary_union(l_df.geometry.values))

print("1973 Rhone Bounds (UTM 32N):", geom_e_utm.bounds)
print("2015 Rhone Bounds (UTM 32N):", geom_l_utm.bounds)

# Check if inside GeoTIFF
# GeoTIFF: left=448620.0, bottom=5155320.0, right=457880.0, top=5164280.0
gt_minx, gt_miny, gt_maxx, gt_maxy = 448620.0, 5155320.0, 457880.0, 5164280.0

for name, g in [("1973", geom_e_utm), ("2015", geom_l_utm)]:
    bx = g.bounds
    inside = gt_minx <= bx[0] and bx[2] <= gt_maxx and gt_miny <= bx[1] and bx[3] <= gt_maxy
    print(f"Is {name} completely inside Rhone GeoTIFF? {inside}")
