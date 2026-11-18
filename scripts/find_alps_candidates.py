import pandas as pd
import pyogrio

csv_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/glims_verified_repeated_observations.csv"
df = pd.read_csv(csv_path, low_memory=False)

# Check glaciers with longitudes between 6 and 11, latitudes between 45 and 48
# In GLIMS IDs: G00[6-9]... or G01[0-1]... and E4[5-7]...
# Example: G008...E46...
alps = df[df['glac_id'].str.match(r'^G00[789]\d{3}E4[567]\d{3}[NS]$', na=False)].copy()
alps['_date'] = pd.to_datetime(alps['src_date'], errors='coerce', utc=True)
alps = alps[alps['line_type'] == 'glac_bound']
print("Alps glacier records:", len(alps))
alps_counts = alps.groupby('glac_id').agg(
    n_obs=('_date', 'nunique'),
    min_date=('_date', 'min'),
    max_date=('_date', 'max'),
    mean_db_area=('db_area', 'mean')
)
multi_alps = alps_counts[alps_counts['n_obs'] >= 2].sort_values('mean_db_area', ascending=False)
print("Top Alps multi-temporal glaciers:")
print(multi_alps.head(15))
