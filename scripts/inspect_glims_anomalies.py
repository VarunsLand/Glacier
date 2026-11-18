import pyogrio
import geopandas as gpd

shp_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/datasets/glims/extracted/glims_download_52211/glims_polygons.shp"
where_clause = "glac_id = 'G277056E81417N'"
df = pyogrio.read_dataframe(shp_path, where=where_clause)
print("Found rows:", len(df))
for idx, row in df.iterrows():
    print(f"Index: {idx}, anlys_id: {row['anlys_id']}, date: {row['src_date']}, line_type: {row['line_type']}, db_area: {row['db_area']}, area: {row['area']}, geom_type: {row['geometry'].geom_type}")
