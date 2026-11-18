from pathlib import Path, PurePosixPath
from collections import Counter
from openpyxl import load_workbook
import csv

ROOT = Path.home() / "Desktop" / "GlacierResearch"
EIS = ROOT / "datasets" / "eis"

csv_path = EIS / "eis_inventory.csv"

with csv_path.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

paths = [r["remote_path"] for r in rows]

print("========== INVENTORY CHECK ==========")
print("Records:", len(rows))
print("Unique paths:", len(set(paths)))

print("Extensions:",
      Counter(PurePosixPath(p).suffix.lower() for p in paths))

print("\nUnusual .1 file examples:")
for p in [p for p in paths if p.endswith(".1")][:15]:
    print(p)

print("\nRegions:")
regions = Counter(
    PurePosixPath(p).parts[4]
    for p in paths
    if len(PurePosixPath(p).parts) > 4
)
print(regions)

err = EIS / "inventory_errors.txt"
if err.exists():
    errors = err.read_text(errors="replace").splitlines()
    print("\nInventory errors:", len(errors))
    for e in errors[:10]:
        print(e)
else:
    print("\nNo inventory error file.")

print("\n========== OFFICIAL MANIFEST ==========")

manifest = EIS / "nsidc-0783_manifest.xlsx"

if not manifest.exists():
    print("Official XLSX manifest not found.")
else:
    wb = load_workbook(manifest, read_only=True, data_only=True)
    for ws in wb.worksheets:
        print("\nSheet:", ws.title)
        print("Dimensions:", ws.max_row, "rows x", ws.max_column, "columns")
        for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 6), values_only=True):
            print(row)
