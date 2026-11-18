import pandas as pd
import numpy as np

p = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/glims_verified_repeated_observations.csv"
df = pd.read_csv(p, low_memory=False)

print(f"Total rows in CSV: {len(df):,}")
print("Line type counts:")
print(df['line_type'].value_counts())

# Filter for glac_bound
gb = df[df['line_type'] == 'glac_bound'].copy()
print(f"glac_bound rows: {len(gb):,}")

# Parse dates
gb['_date'] = pd.to_datetime(gb['src_date'], errors='coerce', utc=True)
gb['_year'] = gb['_date'].dt.year

print(f"Valid dates: {gb['_date'].notna().sum():,} / {len(gb):,}")
print(f"Year range: {gb['_year'].min()} to {gb['_year'].max()}")

# Unique glaciers
print(f"Unique glaciers: {gb['glac_id'].nunique():,}")

# How many glaciers have multiple unique dates?
date_counts = gb.groupby('glac_id')['_date'].nunique()
print(f"Glaciers with >= 2 distinct dates: {(date_counts >= 2).sum():,}")
print(f"Glaciers with >= 2 distinct years: {(gb.groupby('glac_id')['_year'].nunique() >= 2).sum():,}")
