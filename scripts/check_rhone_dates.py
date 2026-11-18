import pandas as pd

csv_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/glims_verified_repeated_observations.csv"
df = pd.read_csv(csv_path, low_memory=False)

sub = df[(df['glac_id'] == 'G008379E46648N') & (df['line_type'] == 'glac_bound')].copy()
sub['_date'] = pd.to_datetime(sub['src_date'], errors='coerce', utc=True)
sub = sub.sort_values('_date')
print("Rhone Glacier Observations:")
for idx, r in sub.iterrows():
    print(f"Date: {r['src_date'][:10]}, anlys_id: {r['anlys_id']}, db_area: {r['db_area']:.4f}, area: {r['area']}")
