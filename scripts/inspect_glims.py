from pathlib import Path
import pyogrio
import geopandas as gpd

ROOT = Path.home() / "Desktop" / "GlacierResearch"

shp = (
    ROOT / "datasets" / "glims" / "extracted" /
    "glims_download_52211" / "glims_polygons.shp"
)

print("========== GLIMS DATA AUDIT ==========")

info = pyogrio.read_info(shp)

print("\nTotal features:", info["features"])
print("CRS:", info["crs"])
print("Geometry:", info["geometry_type"])
print("\nAvailable fields:")

for name, dtype in zip(info["fields"], info["dtypes"]):
    print(name, ":", dtype)

gdf = gpd.read_file(shp, rows=10)

print("\n========== SAMPLE RECORDS ==========")
print(gdf.drop(columns="geometry").to_string(index=False))

print("\n========== GEOMETRY TYPES ==========")
print(gdf.geom_type.value_counts())

print("\n========== AUDIT COMPLETE ==========")
