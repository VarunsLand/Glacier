"""Analyze sample IDs and design split."""
import csv
from pathlib import Path
from collections import defaultdict, Counter

INVENTORY_CSV = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\results\objective4_landsat_dataset_inventory.csv")
with open(INVENTORY_CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total samples: {len(rows)}")
by_region = defaultdict(list)
for r in rows:
    by_region[r["region_prefix"]].append(r)

for reg, items in sorted(by_region.items()):
    active = sum(1 for x in items if x["is_active_change"] == "True")
    no_ch = len(items) - active
    print(f"Region {reg}: {len(items)} samples (Active: {active}, NoChange: {no_ch})")

# Check row/col distributions
for reg, items in sorted(by_region.items())[:2]:
    parsed = []
    for x in items:
        parts = x["sample_id"].split("_")
        parsed.append((int(parts[1]), int(parts[2])))
    rows_idx = sorted(set(p[0] for p in parsed))
    cols_idx = sorted(set(p[1] for p in parsed))
    print(f"Region {reg} rows: {min(rows_idx)}..{max(rows_idx)} ({len(rows_idx)} unique), cols: {min(cols_idx)}..{max(cols_idx)} ({len(cols_idx)} unique)")
