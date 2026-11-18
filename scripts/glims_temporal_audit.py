from pathlib import Path
import pandas as pd
import pyogrio

ROOT = Path.home() / "Desktop" / "GlacierResearch"

shp = (
    ROOT / "datasets" / "glims" / "extracted" /
    "glims_download_52211" / "glims_polygons.shp"
)

print("========== GLIMS HISTORICAL AUDIT ==========")

info = pyogrio.read_info(shp)
fields = list(info["fields"])

print("Total polygon features:", info["features"])
print("CRS:", info["crs"])
print("\nAll fields:")
print(fields)

df = pyogrio.read_dataframe(
    shp,
    columns=fields,
    read_geometry=False
)

print("\nAttribute table:", df.shape)

# Identify glacier ID field
id_col = next(
    (c for c in fields if c.lower() in ["glac_id", "id", "glacier_id"]),
    None
)

# Identify available date fields
date_cols = [
    c for c in fields
    if any(x in c.lower() for x in ["date", "time"])
]

print("\nGlacier ID field:", id_col)
print("Date fields:", date_cols)

for col in date_cols:
    dates = pd.to_datetime(df[col], errors="coerce", utc=True)
    valid = dates.dropna()

    print("\nDATE FIELD:", col)
    print("Valid dates:", len(valid))
    print("Unique dates:", valid.nunique())

    if len(valid):
        print("Earliest:", valid.min())
        print("Latest:", valid.max())
        print("Years:")
        print(valid.dt.year.value_counts().sort_index().to_string())

# Check inventory sources
source_cols = [
    c for c in fields
    if any(x in c.lower() for x in ["src", "inv", "source"])
]

for col in source_cols:
    print("\nSOURCE FIELD:", col)
    print(df[col].value_counts(dropna=False).head(12).to_string())

# Find repeated glacier observations
if id_col and date_cols:
    date_col = next(
        (c for c in date_cols if "anlys" in c.lower()),
        date_cols[0]
    )

    temp = df[[id_col, date_col]].copy()
    temp[date_col] = pd.to_datetime(
        temp[date_col], errors="coerce", utc=True
    )
    temp = temp.dropna()

    counts = temp.groupby(id_col)[date_col].nunique()

    print("\n========== REPEATED OBSERVATIONS ==========")
    print("Unique glacier IDs:", temp[id_col].nunique())
    print("IDs with multiple observation dates:", (counts > 1).sum())
    print("Maximum observations per glacier:", counts.max())

    repeated = counts[counts > 1].sort_values(ascending=False)

    print("\nExample repeated glacier IDs:")
    print(repeated.head(15).to_string())

    repeated_ids = repeated.head(20).index
    sample = temp[temp[id_col].isin(repeated_ids)]

    sample.to_csv(
        ROOT / "outputs" / "glims_repeated_observations.csv",
        index=False
    )

    print("\nSaved repeated observation sample.")

print("\n========== AUDIT COMPLETE ==========")
