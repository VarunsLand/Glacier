import pyogrio

shp_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/datasets/glims/extracted/glims_download_52211/glims_polygons.shp"
df = pyogrio.read_dataframe(shp_path, where="glac_id = 'G277056E81417N'")
for idx, row in df.iterrows():
    bounds = row['geometry'].bounds
    print(f"anlys_id: {row['anlys_id']}, date: {row['src_date']}, db_area: {row['db_area']}, area: {row['area']}, bounds: {bounds}")
