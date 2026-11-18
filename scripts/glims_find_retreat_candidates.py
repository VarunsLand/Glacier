import sys
from pathlib import Path
import pandas as pd
import pyogrio

shp = Path(sys.argv[1])

print("Reading GLIMS data...")
df = pyogrio.read_dataframe(shp, read_geometry=False)

lookup = {str(c).lower(): c for c in df.columns}

def col(*names):
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None

glac = col("glac_id", "glacier_id")
analysis = col("anlys_id", "analysis_id")
src = col("src_date")
anlys = col("anlys_time", "analysis_timestamp")
area = col("db_area")
line = col("line_type")
status = col("rec_status", "record_status")
name = col("glac_name", "glac_name")

required = {
    "glac_id": glac,
    "anlys_id": analysis,
    "src_date": src,
    "db_area": area,
    "line_type": line
}

missing = [k for k, v in required.items() if v is None]
if missing:
    raise ValueError("Required fields missing: " + ", ".join(missing))

print("\n========== INPUT SUMMARY ==========")
print("Total source records:", len(df))
print("Line types:")
print(df[line].value_counts(dropna=False).head(15).to_string())

# Keep only actual glacier boundary records.
work = df[df[line].astype(str).str.lower().eq("glac_bound")].copy()
print("\nGlacier-boundary records:", len(work))

work["_src"] = pd.to_datetime(
    work[src].astype(str),
    format="mixed",
    errors="coerce",
    utc=True
)

if anlys:
    work["_anlys"] = pd.to_datetime(
        work[anlys].astype(str),
        format="mixed",
        errors="coerce",
        utc=True
    )
else:
    work["_anlys"] = pd.NaT

work["_area"] = pd.to_numeric(work[area], errors="coerce")

# Count date quality before applying the modern-observation filter.
print("\n========== SOURCE DATE QUALITY ==========")
print("Missing/unreadable dates:", int(work["_src"].isna().sum()))
print("Dates before 1980:", int((work["_src"].dt.year < 1980).sum()))
print("Dates after 2025:", int((work["_src"].dt.year > 2025).sum()))
print("Zero or missing db_area:", int((work["_area"].isna() | (work["_area"] <= 0)).sum()))

# Make one observation per glacier ID and analysis ID.
observations = []
ambiguous = 0

for (gid, aid), group in work.groupby([glac, analysis], dropna=True):

    dates = group["_src"].dropna().drop_duplicates()
    areas = group["_area"].dropna().round(8).drop_duplicates()

    if len(dates) != 1 or len(areas) != 1:
        ambiguous += 1
        continue

    dt = dates.iloc[0]
    a = float(areas.iloc[0])

    # Restrict this first analysis to modern, plausible dated outlines.
    if dt.year < 1980 or dt.year > 2025:
        continue

    # January 1 dates can indicate year-only or placeholder dates.
    if dt.month == 1 and dt.day == 1:
        continue

    if a <= 0:
        continue

    record = {
        "glac_id": str(gid),
        "anlys_id": aid,
        "src_date": dt,
        "anlys_time": group["_anlys"].max(),
        "db_area_km2": a
    }

    if status:
        record["record_status"] = str(group[status].iloc[0])

    if name:
        record["glac_name"] = str(group[name].iloc[0])

    observations.append(record)

obs = pd.DataFrame(observations)

print("\n========== VALID OBSERVATIONS ==========")
print("Valid glacier-analysis observations:", len(obs))
print("Ambiguous analysis groups skipped:", ambiguous)

if obs.empty:
    raise RuntimeError("No valid observations found. Review date and area fields.")

# If multiple analyses exist for the same glacier and source date,
# keep the most recently performed analysis for that acquisition date.
obs = obs.sort_values(
    ["glac_id", "src_date", "anlys_time"],
    na_position="first"
)

obs = obs.drop_duplicates(
    subset=["glac_id", "src_date"],
    keep="last"
)

candidates = []

for gid, group in obs.groupby("glac_id"):
    group = group.sort_values("src_date")

    if len(group) < 2:
        continue

    first = group.iloc[0]
    last = group.iloc[-1]

    gap_days = (last["src_date"] - first["src_date"]).days
    gap_years = gap_days / 365.2425

    if gap_years < 3:
        continue

    old_area = float(first["db_area_km2"])
    new_area = float(last["db_area_km2"])

    loss = old_area - new_area
    loss_pct = (loss / old_area) * 100 if old_area > 0 else None

    candidates.append({
        "glac_id": gid,
        "old_anlys_id": first["anlys_id"],
        "new_anlys_id": last["anlys_id"],
        "old_src_date": first["src_date"].date().isoformat(),
        "new_src_date": last["src_date"].date().isoformat(),
        "old_anlys_time": first["anlys_time"],
        "new_anlys_time": last["anlys_time"],
        "old_area_km2": old_area,
        "new_area_km2": new_area,
        "area_loss_km2": loss,
        "area_change_percent": loss_pct,
        "time_gap_years": round(gap_years, 2),
        "change_class": "Area reduction" if loss > 0 else (
            "Area increase" if loss < 0 else "No area change"
        )
    })

result = pd.DataFrame(candidates)

if result.empty:
    raise RuntimeError("No glacier candidates found with a 3-year interval.")

result = result.sort_values(
    "time_gap_years",
    ascending=False
)

out = Path.home() / "Desktop" / "GlacierResearch" / "outputs"
out.mkdir(parents=True, exist_ok=True)

csv_path = out / "glims_retreat_candidates.csv"
result.to_csv(csv_path, index=False)

print("\n========== RETREAT CANDIDATE SUMMARY ==========")
print("Glaciers with 2+ valid observations and 3+ year gap:", len(result))
print("Area reductions:", int((result["area_loss_km2"] > 0).sum()))
print("Area increases:", int((result["area_loss_km2"] < 0).sum()))
print("No area change:", int((result["area_loss_km2"] == 0).sum()))

print("\n========== TOP 25 CANDIDATES ==========")
print(
    result.head(25)[[
        "glac_id",
        "old_src_date",
        "new_src_date",
        "old_area_km2",
        "new_area_km2",
        "area_loss_km2",
        "area_change_percent",
        "time_gap_years",
        "change_class"
    ]].to_string(index=False)
)

print("\nSaved results to:")
print(csv_path)

print("\n========== AUDIT COMPLETE ==========")
