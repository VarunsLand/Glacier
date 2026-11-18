import pyogrio
import pandas as pd
from pathlib import Path

shp_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/datasets/glims/extracted/glims_download_52211/glims_polygons.shp"
csv_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/glims_verified_repeated_observations.csv"

df = pd.read_csv(csv_path, low_memory=False)
gb = df[df['line_type'] == 'glac_bound'].copy()
gb['_date'] = pd.to_datetime(gb['src_date'], errors='coerce', utc=True)
gb = gb[gb['_date'].notna()]
gb['year'] = gb['_date'].dt.year

# Filter for satellite era, >= 1 km^2
gb['db_area'] = pd.to_numeric(gb['db_area'], errors='coerce').fillna(0)
large = gb[(gb['db_area'] >= 1.0) & (gb['year'] >= 1980)].copy()

# Find glaciers with at least 2 observations separated by >= 5 years
counts = large.groupby('glac_id').agg(
    n_obs=('year', 'nunique'),
    min_year=('year', 'min'),
    max_year=('year', 'max'),
    mean_area=('db_area', 'mean')
)
candidates = counts[(counts['n_obs'] >= 2) & (counts['max_year'] - counts['min_year'] >= 5)].sort_values('mean_area', ascending=False)
print("Top candidates with multi-temporal geometries:")
print(candidates.head(20))
