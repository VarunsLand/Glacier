"""Test scientifically robust GLIMS retreat calculation.
"""
import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/glims_verified_repeated_observations.csv")
print("Loading CSV...")
df = pd.read_csv(csv_path, low_memory=False)

# 1. glac_bound only
gb = df[df['line_type'] == 'glac_bound'].copy()

# 2. Parse dates
gb['_date'] = pd.to_datetime(gb['src_date'], errors='coerce', utc=True)
gb = gb[gb['_date'].notna()]
gb['_year'] = gb['_date'].dt.year

# Plausible satellite era (1970 - 2024), exclude unverified Jan 1 placeholders
gb = gb[gb['_year'].between(1970, 2024)]
gb = gb[~((gb['_date'].dt.month == 1) & (gb['_date'].dt.day == 1))]

# Convert numeric
gb['db_area'] = pd.to_numeric(gb['db_area'], errors='coerce').fillna(0.0)
gb['area'] = pd.to_numeric(gb['area'], errors='coerce').fillna(0.0)

# Filter out rows with zero or negative db_area
gb = gb[gb['db_area'] > 0.0001]

print(f"Filtered boundary rows: {len(gb):,}")

# 3. Aggregate polygon parts belonging to the same observation
# An observation is uniquely identified by (glac_id, src_date, anlys_time)
obs = gb.groupby(['glac_id', '_date', 'anlys_time'], as_index=False).agg(
    n_parts=('db_area', 'count'),
    total_db_area=('db_area', 'sum'),
    rep_area=('area', 'max'),
    anlys_id=('anlys_id', 'first')
)

print(f"Aggregated observations: {len(obs):,}")

# Filter out incomplete fragments:
# If rep_area > 0 and total_db_area < 0.2 * rep_area, the digitized polygon is only a partial fragment
frag_mask = (obs['rep_area'] > 0) & (obs['total_db_area'] < 0.2 * obs['rep_area'])
print(f"Incomplete fragment observations filtered out: {frag_mask.sum():,}")
obs = obs[~frag_mask].copy()

# If there are multiple submissions for the exact same date and glacier, take the most complete (largest db_area or latest anlys_time)
obs = obs.sort_values(['glac_id', '_date', 'total_db_area'], ascending=[True, True, False])
obs = obs.drop_duplicates(subset=['glac_id', '_date'], keep='first')

print(f"Unique (glacier, date) observations: {len(obs):,}")

# 4. Glacier multi-temporal pairing
# Glaciers with >= 2 dates
glac_counts = obs.groupby('glac_id')['_date'].count()
multi_glacs = glac_counts[glac_counts >= 2].index
obs_multi = obs[obs['glac_id'].isin(multi_glacs)].sort_values(['glac_id', '_date'])

print(f"Glaciers with >= 2 observations: {len(multi_glacs):,}")

# Pair earliest and latest with time gap >= 3 years
pairs = []
for gid, group in obs_multi.groupby('glac_id'):
    group = group.sort_values('_date')
    earliest = group.iloc[0]
    latest = group.iloc[-1]
    
    t1 = earliest['_date']
    t2 = latest['_date']
    gap_years = (t2 - t1).days / 365.2425
    if gap_years < 3.0 or gap_years > 50.0:
        continue
        
    a1 = earliest['total_db_area']
    a2 = latest['total_db_area']
    
    if a1 < 0.01 or a2 < 0.01:
        continue
        
    # Check area ratio plausibility:
    # A glacier does not shrink by > 80% or grow by > 100% in a few years unless it's a completely different spatial delineation
    ratio = a2 / a1
    if ratio < 0.15 or ratio > 3.0:
        continue
        
    loss_km2 = a1 - a2
    pct_change = (loss_km2 / a1) * 100.0
    annual_loss = loss_km2 / gap_years
    annual_pct = pct_change / gap_years
    
    pairs.append({
        'glac_id': gid,
        'earlier_date': t1.strftime('%Y-%m-%d'),
        'later_date': t2.strftime('%Y-%m-%d'),
        'earlier_year': int(t1.year),
        'later_year': int(t2.year),
        'time_gap_years': round(gap_years, 2),
        'earlier_area_km2': round(a1, 4),
        'later_area_km2': round(a2, 4),
        'area_loss_km2': round(loss_km2, 4),
        'percentage_change': round(pct_change, 2),
        'annual_loss_km2_yr': round(annual_loss, 5),
        'annual_pct_change_yr': round(annual_pct, 3)
    })

df_pairs = pd.DataFrame(pairs)
print(f"\nValid paired glaciers: {len(df_pairs):,}")
total_loss = df_pairs['area_loss_km2'].sum()
mean_loss_pct = df_pairs['percentage_change'].mean()
retreating = (df_pairs['area_loss_km2'] > 0).sum()
advancing = (df_pairs['area_loss_km2'] < 0).sum()
stable = (df_pairs['area_loss_km2'] == 0).sum()

print(f"Total net area change: {total_loss:,.2f} km^2")
print(f"Mean percentage change: {mean_loss_pct:.2f}%")
print(f"Retreating glaciers: {retreating:,} ({retreating/len(df_pairs)*100:.1f}%)")
print(f"Advancing glaciers: {advancing:,} ({advancing/len(df_pairs)*100:.1f}%)")
print(f"Top 5 retreat cases:")
print(df_pairs.sort_values('area_loss_km2', ascending=False).head(5)[['glac_id', 'earlier_year', 'later_year', 'time_gap_years', 'earlier_area_km2', 'later_area_km2', 'area_loss_km2', 'percentage_change']])
