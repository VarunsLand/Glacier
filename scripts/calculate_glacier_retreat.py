"""GlacierVision: Quantitative Glacier Retreat & Area Change Assessment.

Analyzes multi-temporal GLIMS polygon observations:
- Evaluates real multi-temporal glacier boundaries across multi-year baselines (Delta t >= 3 years).
- Uses validated NSIDC geodesic db_area and equal-area geometry.
- Quantifies initial area, later area, area loss (km²), percentage change (%), and annualized retreat rates.
- Saves clean, publication-grade CSV and summary report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "outputs" / "glims_verified_repeated_observations.csv"
OUT_DIR = ROOT / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_retreat_assessment(min_gap_years: float = 3.0, max_gap_years: float = 50.0):
    print("=" * 65)
    print("OBJECTIVES 2 & 3: GLACIER RETREAT & QUANTITATIVE AREA ASSESSMENT")
    print("=" * 65)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    print(f"Reading verified observations from: {INPUT_CSV.name}...")
    # Load required columns
    cols = ["glac_id", "anlys_id", "anlys_time", "src_date", "line_type", "db_area", "area"]
    df = pd.read_csv(INPUT_CSV, usecols=lambda c: c in cols, low_memory=False)
    print(f"Loaded {len(df):,} total records.")

    # 1. Filter for boundary outlines only
    if "line_type" in df.columns:
        df = df[df["line_type"].astype(str).str.lower() == "glac_bound"].copy()
        print(f"Glacier boundary polygons: {len(df):,}")

    # 2. Parse dates and valid areas
    df["_date"] = pd.to_datetime(df["src_date"], errors="coerce", utc=True)
    df["_area"] = pd.to_numeric(df["db_area"], errors="coerce")

    # Fallback to provider area if db_area is missing
    if "area" in df.columns:
        df["_area"] = df["_area"].fillna(pd.to_numeric(df["area"], errors="coerce"))

    # Quality filters: plausible satellite-era dates and positive area
    valid = df[
        df["_date"].notna()
        & df["_date"].dt.year.between(1970, 2024)
        & df["_area"].notna()
        & (df["_area"] > 0.01)  # At least 0.01 km²
    ].copy()

    # Exclude placeholder Jan 1 dates if unverified
    valid = valid[~((valid["_date"].dt.month == 1) & (valid["_date"].dt.day == 1))]
    print(f"High-confidence dated observations: {len(valid):,}")

    # Deduplicate: 1 observation per glacier ID and source date
    valid = valid.sort_values(["glac_id", "_date"])
    valid = valid.drop_duplicates(subset=["glac_id", "_date"], keep="last")

    # Group by glacier ID to find multi-temporal pairs
    candidates = []
    glacier_groups = valid.groupby("glac_id")
    print(f"Unique glaciers with valid observations: {len(glacier_groups):,}")

    for gid, group in glacier_groups:
        if len(group) < 2:
            continue

        group = group.sort_values("_date")
        earliest = group.iloc[0]
        latest = group.iloc[-1]

        t1 = earliest["_date"]
        t2 = latest["_date"]

        gap_days = (t2 - t1).days
        gap_years = gap_days / 365.2425

        if gap_years < min_gap_years or gap_years > max_gap_years:
            continue

        a1 = float(earliest["_area"])
        a2 = float(latest["_area"])

        if a1 <= 0 or a2 <= 0:
            continue

        loss_km2 = a1 - a2
        pct_change = (loss_km2 / a1) * 100.0
        annual_loss_km2 = loss_km2 / gap_years
        annual_pct = pct_change / gap_years

        # Classify retreat dynamics
        if pct_change >= 10.0:
            dynamics = "Significant Retreat"
        elif pct_change >= 2.0:
            dynamics = "Moderate Retreat"
        elif pct_change <= -2.0:
            dynamics = "Advance / Growth"
        else:
            dynamics = "Stable / Equilibrium"

        candidates.append({
            "glacier_id": str(gid),
            "earlier_analysis_id": earliest.get("anlys_id", ""),
            "later_analysis_id": latest.get("anlys_id", ""),
            "earlier_date": t1.strftime("%Y-%m-%d"),
            "later_date": t2.strftime("%Y-%m-%d"),
            "earlier_year": int(t1.year),
            "later_year": int(t2.year),
            "time_gap_years": round(gap_years, 2),
            "earlier_area_km2": round(a1, 4),
            "later_area_km2": round(a2, 4),
            "area_loss_km2": round(loss_km2, 4),
            "percentage_change": round(pct_change, 2),
            "annual_loss_km2_yr": round(annual_loss_km2, 5),
            "annual_pct_change_yr": round(annual_pct, 3),
            "dynamics_class": dynamics,
        })

    df_res = pd.DataFrame(candidates)
    if df_res.empty:
        raise RuntimeError("No candidate glaciers matched the multi-year filter criteria.")

    # Sort by absolute loss
    df_res = df_res.sort_values("area_loss_km2", ascending=False)

    out_csv = OUT_DIR / "glims_retreat_assessment.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved quantitative retreat assessment ({len(df_res)} glaciers) to:\n  {out_csv}")

    # Summary Statistics
    total_glaciers = len(df_res)
    retreating = int((df_res["area_loss_km2"] > 0).sum())
    retreat_pct = (retreating / total_glaciers) * 100.0
    advancing = int((df_res["area_loss_km2"] < 0).sum())
    stable = int((df_res["dynamics_class"] == "Stable / Equilibrium").sum())

    total_loss = float(df_res["area_loss_km2"].sum())
    avg_loss_pct = float(df_res["percentage_change"].mean())
    avg_gap = float(df_res["time_gap_years"].mean())

    summary = {
        "dataset": "GLIMS Global Land Ice Measurements from Space",
        "total_multi_temporal_glaciers_analyzed": total_glaciers,
        "retreating_glaciers_count": retreating,
        "retreating_glaciers_ratio": round(retreat_pct, 2),
        "advancing_glaciers_count": advancing,
        "stable_glaciers_count": stable,
        "total_net_area_loss_km2": round(total_loss, 2),
        "mean_percentage_area_change": round(avg_loss_pct, 2),
        "mean_observation_gap_years": round(avg_gap, 2),
        "dynamics_breakdown": df_res["dynamics_class"].value_counts().to_dict(),
        "top_10_retreat_glaciers": df_res.head(10)[[
            "glacier_id", "earlier_date", "later_date", "earlier_area_km2", "later_area_km2", "area_loss_km2", "percentage_change"
        ]].to_dict(orient="records"),
    }

    out_summary = OUT_DIR / "glims_retreat_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY OF RETREAT ASSESSMENT ===")
    print(f"Total Glaciers with Valid Multi-Year Pairs: {total_glaciers:,}")
    print(f"Glaciers Showing Area Loss:                {retreating:,} ({retreat_pct:.1f}%)")
    print(f"Glaciers Showing Area Gain / Advance:      {advancing:,}")
    print(f"Total Cumulative Area Loss:                {total_loss:,.2f} km²")
    print(f"Mean Percentage Change:                    {avg_loss_pct:.2f}%")
    print(f"Mean Observation Gap:                      {avg_gap:.1f} years")
    print(f"Saved summary JSON: {out_summary}")


if __name__ == "__main__":
    compute_retreat_assessment()
