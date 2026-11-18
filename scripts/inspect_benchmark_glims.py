import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

shp_path = 'datasets/glims/extracted/glims_download_52211/glims_polygons.shp'
benchmarks = ['G008379E46648N', 'G008032E46504N', 'G008187E46569N']

where_clause = "glac_id IN ('G008379E46648N', 'G008032E46504N', 'G008187E46569N')"
gdf = gpd.read_file(shp_path, where=where_clause)
print(f"Total matching polygon features: {len(gdf)}")

cols = ['glac_id', 'glac_name', 'anlys_id', 'anlys_time', 'src_date', 'db_area', 'line_type']
print("\n--- ALL RECORDS FOR BENCHMARK GLACIERS ---")
print(gdf[cols].to_string())

# Now let's calculate projected area in EPSG:32632 for each (glac_id, anlys_id) and (glac_id, src_date)
gdf_utm = gdf.to_crs(epsg=32632)

print("\n--- DETAILED ANALYSIS PER GLACIER AND OBSERVATION ---")
for gid in benchmarks:
    sub = gdf_utm[gdf_utm['glac_id'] == gid]
    print(f"\n=======================================================")
    print(f"GLACIER: {gid} ({sub['glac_name'].iloc[0]})")
    print(f"=======================================================")
    
    # Group by src_date
    for s_date, grp in sub.groupby('src_date'):
        anlys_ids = grp['anlys_id'].unique().tolist()
        anlys_times = grp['anlys_time'].unique().tolist()
        line_types = grp['line_type'].unique().tolist()
        n_polys = len(grp)
        sum_db_area = grp['db_area'].sum()
        
        # Merge geometries
        union_geom = unary_union(grp.geometry)
        geom_area_km2 = union_geom.area / 1e6
        
        # Individual geometry sum without union
        sum_geom_area_km2 = grp.geometry.area.sum() / 1e6
        
        print(f"src_date: {s_date} | anlys_time: {anlys_times} | anlys_id: {anlys_ids}")
        print(f"  Count: {n_polys} polygons | line_types: {line_types}")
        print(f"  sum(db_area): {sum_db_area:.6f} km²")
        print(f"  unary_union geom area (EPSG:32632): {geom_area_km2:.6f} km²")
        print(f"  sum(individual geom area): {sum_geom_area_km2:.6f} km²")
