from pathlib import Path
from collections import Counter
import csv
import json
import xarray as xr

ROOT = Path.home() / "Desktop" / "GlacierResearch"
RESULTS = ROOT / "outputs"
RESULTS.mkdir(parents=True, exist_ok=True)

report = {}

print("\n========== LANDSAT AUDIT ==========")

landsat = ROOT / "datasets" / "landsat"

for folder in ["A", "B", "labels", "Samples"]:
    p = landsat / folder
    if p.exists():
        files = [x for x in p.rglob("*") if x.is_file()]
        print(f"{folder}: {len(files)} files")
        print("Extensions:", Counter(x.suffix.lower() for x in files))
        report[f"landsat_{folder}"] = len(files)
    else:
        print("MISSING:", folder)

print("\n========== DL4GAM AUDIT ==========")

patch_folder = ROOT / "datasets" / "dl4gam" / "extracted"
nc_files = list(patch_folder.rglob("*.nc"))

print("NetCDF patches:", len(nc_files))
report["dl4gam_patches"] = len(nc_files)

csv_files = list((ROOT / "datasets" / "dl4gam").rglob("*.csv"))

for p in csv_files:
    print("\nCSV:", p.name)
    with p.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        print("Columns:", next(reader, []))
        for _ in range(3):
            row = next(reader, None)
            if row:
                print(row)

if nc_files:
    sample = nc_files[0]

    print("\n========== NETCDF SAMPLE ==========")
    print("File:", sample.name)

    with xr.open_dataset(sample) as ds:
        print("Dimensions:", dict(ds.sizes))
        print("Variables:", list(ds.data_vars))
        print("Coordinates:", list(ds.coords))
        print("Attributes:", dict(ds.attrs))

        for name, var in ds.data_vars.items():
            print(name, var.dims, var.shape, var.dtype)

report["netcdf_sample"] = str(nc_files[0]) if nc_files else None

with (RESULTS / "dataset_audit.json").open("w") as f:
    json.dump(report, f, indent=4)

print("\n========== AUDIT COMPLETED ==========")
print("Report:", RESULTS / "dataset_audit.json")
