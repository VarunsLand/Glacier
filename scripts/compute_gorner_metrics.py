import pyogrio, pyproj
from shapely.ops import unary_union, transform
from pathlib import Path

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
shp = ROOT / "datasets" / "glims" / "extracted" / "glims_download_52211" / "glims_polygons.shp"
gdf = pyogrio.read_dataframe(shp, where="glac_id = 'G007800E45965N' AND line_type = 'glac_bound'")
gdf["_date"] = gdf["src_date"].str[:10]
e_df = gdf[gdf["_date"] == "1973-09-01"]
l_df = gdf[gdf["_date"] == "2015-08-29"]
geom_e = unary_union(e_df.geometry.values)
geom_l = unary_union(l_df.geometry.values)
project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform
geom_e_utm = transform(project, geom_e)
geom_l_utm = transform(project, geom_l)
ae = geom_e_utm.area / 1e6
al = geom_l_utm.area / 1e6
ret = geom_e_utm.intersection(geom_l_utm).area / 1e6
lost = geom_e_utm.difference(geom_l_utm).area / 1e6
gain = geom_l_utm.difference(geom_e_utm).area / 1e6
print("Gorner UTM EPSG:32632:")
print(f"Earlier: {ae:.3f}")
print(f"Later:   {al:.3f}")
print(f"Retained:{ret:.3f}")
print(f"Lost:    {lost:.3f}")
print(f"Gained:  {gain:.3f}")
print(f"Net:     {ae-al:.3f}")
print(f"Pct:     {(ae-al)/ae*100:.2f}%")
print(f"Cons 1:  {abs(ae - (ret + lost)):.2e}")
print(f"Cons 2:  {abs(al - (ret + gain)):.2e}")
