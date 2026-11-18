"""GlacierVision: Comprehensive Landsat Dataset Inventory & Integrity Audit.

Inspects all files in datasets/landsat/A, B, labels.
Generates outputs/results/objective4_landsat_dataset_inventory.csv.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

DATA_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\datasets\landsat")
OUT_CSV = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\results\objective4_landsat_dataset_inventory.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

dir_a = DATA_DIR / "A"
dir_b = DATA_DIR / "B"
dir_lbl = DATA_DIR / "labels"

files_a = {p.stem: p for p in dir_a.glob("*.png")}
files_b = {p.stem: p for p in dir_b.glob("*.png")}
files_lbl = {p.stem: p for p in dir_lbl.glob("*.png")}

all_ids = sorted(set(files_a.keys()) | set(files_b.keys()) | set(files_lbl.keys()))
print(f"Total unique sample identifiers found: {len(all_ids)}")
print(f"Files in A: {len(files_a)}, Files in B: {len(files_b)}, Files in labels: {len(files_lbl)}")

rows = []
active_change_count = 0
no_change_count = 0
corrupted_count = 0

for sid in all_ids:
    has_a = sid in files_a
    has_b = sid in files_b
    has_lbl = sid in files_lbl
    is_paired = has_a and has_b and has_lbl

    path_a = str(files_a[sid]) if has_a else ""
    path_b = str(files_b[sid]) if has_b else ""
    path_lbl = str(files_lbl[sid]) if has_lbl else ""

    dim_a, dim_b, dim_lbl = "", "", ""
    ch_a, ch_b, ch_lbl = "", "", ""
    dtype_a, dtype_b, dtype_lbl = "", "", ""
    lbl_unique = ""
    lbl_pos_px = 0
    lbl_change_pct = 0.0
    valid_status = "VALID"

    try:
        if has_a:
            with Image.open(path_a) as im_a:
                dim_a = f"{im_a.width}x{im_a.height}"
                ch_a = len(im_a.getbands())
                arr_a = np.array(im_a)
                dtype_a = str(arr_a.dtype)

        if has_b:
            with Image.open(path_b) as im_b:
                dim_b = f"{im_b.width}x{im_b.height}"
                ch_b = len(im_b.getbands())
                arr_b = np.array(im_b)
                dtype_b = str(arr_b.dtype)

        if has_lbl:
            with Image.open(path_lbl) as im_l:
                dim_lbl = f"{im_l.width}x{im_l.height}"
                ch_lbl = len(im_l.getbands())
                arr_l = np.array(im_l)
                dtype_lbl = str(arr_l.dtype)
                uniques = np.unique(arr_l).tolist()
                lbl_unique = str(uniques)
                pos = int(np.sum(arr_l > 0))
                lbl_pos_px = pos
                lbl_change_pct = (pos / (im_l.width * im_l.height)) * 100.0
                if pos > 0:
                    active_change_count += 1
                else:
                    no_change_count += 1
    except Exception as e:
        valid_status = f"ERROR: {e}"
        corrupted_count += 1

    # Check dimension match
    if is_paired:
        if not (dim_a == "256x256" and dim_b == "256x256" and dim_lbl == "256x256"):
            valid_status = f"DIM_MISMATCH: A={dim_a}, B={dim_b}, L={dim_lbl}"
        elif ch_a != 3 or ch_b != 3:
            valid_status = f"CHANNEL_MISMATCH: A={ch_a}, B={ch_b}"

    # Determine region from prefix
    parts = sid.split("_")
    reg_prefix = parts[0] if parts else ""
    region_name = "Greenland" if reg_prefix == "1" else ("Svalbard" if reg_prefix in ("2", "3") else "Unknown")

    rows.append({
        "sample_id": sid,
        "region_prefix": reg_prefix,
        "region_name": region_name,
        "path_a": path_a,
        "path_b": path_b,
        "path_label": path_lbl,
        "dimensions_a": dim_a,
        "dimensions_b": dim_b,
        "dimensions_label": dim_lbl,
        "channels_a": ch_a,
        "channels_b": ch_b,
        "channels_label": ch_lbl,
        "dtype_a": dtype_a,
        "dtype_b": dtype_b,
        "dtype_label": dtype_lbl,
        "label_unique_values": lbl_unique,
        "label_positive_pixels": lbl_pos_px,
        "label_change_percentage": f"{lbl_change_pct:.4f}",
        "is_active_change": lbl_pos_px > 0,
        "validation_status": valid_status
    })

# Write CSV
fieldnames = list(rows[0].keys())
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Inventory saved to {OUT_CSV}")
print(f"Active Change Triplets: {active_change_count}")
print(f"No-Change Triplets:     {no_change_count}")
print(f"Total Valid Triplets:   {active_change_count + no_change_count}")
print(f"Corrupted Triplets:     {corrupted_count}")

# Distribution of region prefixes
from collections import Counter
pfx_counts = Counter(r["region_prefix"] for r in rows)
print("Region prefix distribution:", dict(pfx_counts))
