import pandas as pd

p = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/glims_verified_repeated_observations.csv"
df = pd.read_csv(p, low_memory=False)

check_ids = ["G279683E81662N", "G216453E60499N", "G281287E82135N", "G217991E60521N", "G227688E56917N", "G215764E60620N"]
for gid in check_ids:
    sub = df[df['glac_id'] == gid]
    print(f"\n=== Glacier {gid} ===")
    print(sub[['glac_id', 'anlys_id', 'src_date', 'line_type', 'db_area', 'area']])
