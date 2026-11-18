import json
from pathlib import Path
import pyogrio
import pandas as pd

shp_path = Path("datasets/glims/extracted/glims_download_52211/glims_polygons.shp")
results_dir = Path("outputs/results")
results_dir.mkdir(parents=True, exist_ok=True)

print("Reading attributes from glims_polygons.shp...")
df = pyogrio.read_dataframe(
    shp_path,
    read_geometry=False,
    columns=["glac_id", "anlys_id", "src_date", "glac_name", "db_area", "line_type"]
)
print(f"Total rows: {len(df)}")

# Filter to boundary lines
bounds_df = df[df["line_type"] == "glac_bound"].copy()

# Build unique name mapping
df_named = bounds_df[
    bounds_df["glac_name"].notna() &
    (bounds_df["glac_name"].str.strip() != "") &
    (bounds_df["glac_name"].str.strip() != "None")
]
name_map = df_named.drop_duplicates(subset=["glac_id"]).set_index("glac_id")["glac_name"].to_dict()

# Add benchmark names cleanly
benchmark_canonical_names = {
    "G008379E46648N": "Rhone Glacier (Rhonegletscher)",
    "G008032E46504N": "Great Aletsch Glacier (Grosser Aletschgletscher)",
    "G008187E46569N": "Unteraar Glacier (Unteraargletscher)",
    "G007800E45965N": "Gorner Glacier (Gornergletscher)",
    "G007880E45990N": "Findel Glacier (Findelgletscher)",
}
name_map.update(benchmark_canonical_names)

out_names_path = results_dir / "glims_glacier_names.json"
with open(out_names_path, "w", encoding="utf-8") as f:
    json.dump(name_map, f, indent=2)
print(f"Saved {len(name_map)} glacier names to {out_names_path}")

# Check specific glaciers
target_ids = [
    "G008379E46648N",
    "G008032E46504N",
    "G008187E46569N",
    "G007800E45965N",
    "G010725E46789N",
    "G007880E45990N",
]

glacier_meta = {}
for gid in target_ids:
    sub = bounds_df[bounds_df["glac_id"] == gid]
    glacier_meta[gid] = sub.to_dict(orient="records")

with open(results_dir / "target_glaciers_metadata.json", "w", encoding="utf-8") as f:
    json.dump(glacier_meta, f, indent=2, default=str)
print("Saved target_glaciers_metadata.json")
