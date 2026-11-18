import sys
from pathlib import Path
import pandas as pd
import pyogrio

shp = Path(sys.argv[1])

print("Reading GLIMS polygon attributes...")
df = pyogrio.read_dataframe(shp, read_geometry=False)

print("\n========== AVAILABLE FIELDS ==========")
for c in df.columns:
    print(c)

lookup = {str(c).lower(): c for c in df.columns}

def get_col(*names):
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None

glac = get_col("glac_id", "glacier_id")
analysis = get_col("anlys_id", "analysis_id")
analysis_date = get_col("anlys_time", "analysis_timestamp")
source_date = get_col("src_date")
gone_date = get_col("gone_date")
line_type = get_col("line_type")
area = get_col("area")
db_area = get_col("db_area")

print("\n========== IMPORTANT FIELD MAPPING ==========")
for name, col in [
    ("Glacier ID", glac),
    ("Analysis ID", analysis),
    ("Analysis timestamp", analysis_date),
    ("Source date", source_date),
    ("Gone date", gone_date),
    ("Line type", line_type),
    ("Provider area", area),
    ("NSIDC area", db_area),
]:
    print(f"{name}: {col}")

for col in [analysis_date, source_date, gone_date]:
    if col is None:
        continue

    parsed = pd.to_datetime(
        df[col].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
        utc=True
    )

    plausible = parsed[
        parsed.notna()
        & parsed.dt.year.between(1950, 2026)
    ]

    print(f"\n========== DATE AUDIT: {col} ==========")
    print("Valid parsed dates:", int(parsed.notna().sum()))
    print("Plausible dates (1950-2026):", int(len(plausible)))
    print("Unique plausible dates:", int(plausible.nunique()))

    if len(plausible):
        print("Earliest:", plausible.min())
        print("Latest:", plausible.max())
        print("\nYear distribution:")
        print(plausible.dt.year.value_counts().sort_index().to_string())

if line_type:
    print("\n========== LINE TYPE COUNTS ==========")
    print(df[line_type].value_counts(dropna=False).head(20).to_string())

if glac and analysis:
    print("\n========== ANALYSIS ID AUDIT ==========")

    observations = df.dropna(subset=[glac, analysis]).copy()
    observations = observations.drop_duplicates([glac, analysis])

    counts = observations.groupby(glac)[analysis].nunique()

    print("Unique glacier IDs:", int(counts.size))
    print("Unique glacier-analysis pairs:", len(observations))
    print("Glaciers with multiple analysis IDs:", int((counts > 1).sum()))
    print("Maximum analyses per glacier:", int(counts.max()))

    repeated_ids = counts[counts > 1].sort_values(ascending=False)
    print("\nExample glaciers with repeated analyses:")
    print(repeated_ids.head(20).to_string())

    keep = [glac, analysis]
    for col in [analysis_date, source_date, gone_date, line_type, area, db_area]:
        if col and col not in keep:
            keep.append(col)

    repeated = observations[
        observations[glac].isin(repeated_ids.index)
    ][keep].copy()

    sort_cols = [glac]
    if analysis_date:
        repeated[analysis_date] = pd.to_datetime(
            repeated[analysis_date].astype(str),
            format="mixed",
            errors="coerce",
            utc=True
        )
        sort_cols.append(analysis_date)

    repeated = repeated.sort_values(sort_cols)

    out = Path.home() / "Desktop" / "GlacierResearch" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    output_file = out / "glims_verified_repeated_observations.csv"
    repeated.to_csv(output_file, index=False)

    print("\nSaved repeated observations to:")
    print(output_file)

    print("\n========== SAMPLE REPEATED RECORDS ==========")
    print(repeated.head(25).to_string(index=False))

else:
    print("\nCould not find both glacier ID and analysis ID fields.")

print("\n========== AUDIT FINISHED ==========")
