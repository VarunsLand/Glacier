"""Test multi-regional equal-area projections and WGS84 geodesic area calculations.
"""
import pyogrio
import geopandas as gpd
import pyproj
from shapely.ops import transform, unary_union
import numpy as np

shp_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/datasets/glims/extracted/glims_download_52211/glims_polygons.shp"
geod = pyproj.Geod(ellps="WGS84")

# Representative glaciers from 4 distinct global glaciological regions:
REGIONAL_TESTS = [
    {
        "region": "European Alps",
        "glacier_name": "Rhone Glacier (Rhonegletscher)",
        "glac_id": "G008379E46648N",
        "regional_crs": "EPSG:3035",  # ETRS89-LAEA (European Equal Area)
        "year": 1973
    },
    {
        "region": "European Alps",
        "glacier_name": "Great Aletsch Glacier",
        "glac_id": "G008032E46504N",
        "regional_crs": "EPSG:3035",  # ETRS89-LAEA
        "year": 1973
    },
    {
        "region": "Alaska / St. Elias",
        "glacier_name": "Malaspina Glacier Complex",
        "glac_id": "G219787E60289N",
        "regional_crs": "EPSG:3338",  # Alaska Albers Equal Area
        "year": 2000
    },
    {
        "region": "Canadian Arctic",
        "glacier_name": "Axel Heiberg Ice Cap",
        "glac_id": "G284432E82157N",
        "regional_crs": "EPSG:3413",  # Arctic Polar Stereographic
        "year": 1999
    }
]

print("=" * 80)
print("MULTI-REGIONAL EQUAL-AREA PROJECTION & GEODESIC VALIDATION")
print("=" * 80)

for test in REGIONAL_TESTS:
    gid = test["glac_id"]
    where_c = f"glac_id = '{gid}' AND line_type = 'glac_bound'"
    df = pyogrio.read_dataframe(shp_path, where=where_c)
    df['_year'] = df['src_date'].str[:4].astype(int)
    
    # Pick target year
    sub = df[df['_year'] == test["year"]]
    if sub.empty:
        # pick closest year
        sub = df[df['_year'] == df['_year'].min()]
    
    actual_year = sub['_year'].iloc[0]
    db_area_sum = sub['db_area'].sum()
    geom_wgs = unary_union(sub['geometry'].values)
    
    # 1. Geodesic WGS84 area calculation (ellipsoidal, no planar projection distortion)
    geo_area_m2, _ = geod.geometry_area_perimeter(geom_wgs)
    geo_area_km2 = abs(geo_area_m2) / 1e6
    
    # 2. Regional Equal-Area Projection area calculation
    proj_to_equal_area = pyproj.Transformer.from_crs("EPSG:4326", test["regional_crs"], always_xy=True).transform
    geom_proj = transform(proj_to_equal_area, geom_wgs)
    proj_area_km2 = geom_proj.area / 1e6
    
    diff_geo_db = abs(geo_area_km2 - db_area_sum)
    diff_geo_db_pct = (diff_geo_db / db_area_sum) * 100.0
    
    diff_proj_geo = abs(proj_area_km2 - geo_area_km2)
    diff_proj_geo_pct = (diff_proj_geo / geo_area_km2) * 100.0

    print(f"\nRegion:           {test['region']}")
    print(f"Glacier:          {test['glacier_name']} ({gid}) | Epoch: {actual_year}")
    print(f"Regional CRS:     {test['regional_crs']}")
    print(f"NSIDC db_area:    {db_area_sum:.4f} km^2")
    print(f"Geodesic WGS84:   {geo_area_km2:.4f} km^2  (Diff vs db_area: {diff_geo_db:.4f} km^2, {diff_geo_db_pct:.2f}%)")
    print(f"Regional Proj:    {proj_area_km2:.4f} km^2  (Diff vs Geodesic: {diff_proj_geo:.4f} km^2, {diff_proj_geo_pct:.2f}%)")
