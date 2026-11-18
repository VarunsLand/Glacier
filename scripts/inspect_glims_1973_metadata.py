import pyogrio, json
from pathlib import Path

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
shp = ROOT / "datasets" / "glims" / "extracted" / "glims_download_52211" / "glims_polygons.shp"

where_clause = "glac_id IN ('G008379E46648N', 'G008032E46504N', 'G008187E46569N')"
df = pyogrio.read_dataframe(shp, where=where_clause)
print(f"Loaded {len(df)} rows total.", flush=True)

out = []
for idx, row in df.iterrows():
    d = {k: str(v) for k, v in row.items() if k != "geometry"}
    out.append(d)

out_file = ROOT / "scratch" / "benchmark_glims_metadata.json"
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"Wrote metadata to {out_file}", flush=True)
