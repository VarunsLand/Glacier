import pyogrio

shp_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/datasets/glims/extracted/glims_download_52211/glims_polygons.shp"
where_clause = "anlys_id = 1097447"
df = pyogrio.read_dataframe(shp_path, where=where_clause)
print("Found rows for anlys_id 1097447:", len(df))
print(df[['glac_id', 'glac_name', 'anlys_id', 'src_date', 'line_type', 'db_area', 'area']])
