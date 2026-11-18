import pandas as pd

csv_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/glims_verified_repeated_observations.csv"
df = pd.read_csv(csv_path, low_memory=False)

for gid, name in [("G008187E46569N", "Unteraar Glacier"), ("G008398E46623N", "Oberaar Glacier")]:
    sub = df[(df['glac_id'] == gid) & (df['line_type'] == 'glac_bound')].copy()
    sub['_date'] = pd.to_datetime(sub['src_date'], errors='coerce', utc=True)
    sub = sub.sort_values('_date')
    print(f"\n=== {name} ({gid}) ===")
    obs = sub.groupby(['src_date'], as_index=False).agg(
        total_db_area=('db_area', 'sum'),
        anlys_id=('anlys_id', 'first')
    )
    for idx, r in obs.iterrows():
        print(f"Date: {r['src_date'][:10]}, anlys_id: {r['anlys_id']}, total_db_area: {r['total_db_area']:.4f}")
