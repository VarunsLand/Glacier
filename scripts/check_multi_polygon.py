import pandas as pd
import numpy as np

p = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/glims_verified_repeated_observations.csv"
df = pd.read_csv(p, low_memory=False)
gb = df[df['line_type'] == 'glac_bound'].copy()
gb['_date'] = pd.to_datetime(gb['src_date'], errors='coerce', utc=True)
gb['_db_area'] = pd.to_numeric(gb['db_area'], errors='coerce')
gb['_area'] = pd.to_numeric(gb['area'], errors='coerce')

# Check multi-polygon observations:
# Group by (glac_id, _date)
counts_per_date = gb.groupby(['glac_id', '_date']).agg(
    n_rows=('anlys_id', 'count'),
    n_anlys=('anlys_id', 'nunique'),
    sum_db_area=('_db_area', 'sum'),
    max_db_area=('_db_area', 'max'),
    min_db_area=('_db_area', 'min'),
    mean_area=('_area', 'mean'),
    max_area=('_area', 'max')
).reset_index()

multi_row = counts_per_date[counts_per_date['n_rows'] > 1]
print(f"Total (glacier, date) pairs: {len(counts_per_date):,}")
print(f"Pairs with multiple rows: {len(multi_row):,} ({len(multi_row)/len(counts_per_date)*100:.1f}%)")
print(f"Pairs with multiple anlys_id: {(counts_per_date['n_anlys'] > 1).sum():,}")

print("\nSample of pairs with multiple anlys_id on the same date:")
sample_multi = multi_row[multi_row['n_anlys'] > 1].head(10)
print(sample_multi[['glac_id', '_date', 'n_rows', 'n_anlys', 'sum_db_area', 'max_db_area', 'max_area']])
