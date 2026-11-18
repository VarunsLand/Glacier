import pandas as pd

csv_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/glims_verified_repeated_observations.csv"
df = pd.read_csv(csv_path, low_memory=False)

for gid, name in [("G008379E46648N", "Rhone Glacier"), ("G008032E46504N", "Great Aletsch Glacier")]:
    sub = df[(df['glac_id'] == gid) & (df['line_type'] == 'glac_bound')].copy()
    sub['_date'] = pd.to_datetime(sub['src_date'], errors='coerce', utc=True)
    sub = sub.sort_values('_date')
    print(f"\n=== {name} ({gid}) ===")
    print(sub[['glac_id', 'anlys_id', 'src_date', 'anlys_time', 'db_area', 'area']])
