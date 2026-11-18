"""GlacierVision: Validated Quantitative Glacier Retreat & Area Change Assessment.

Rigorous scientific pipeline:
- Validates GLIMS multi-temporal boundary observations (line_type == 'glac_bound').
- Corrects multi-polygon fragmentation: aggregates all polygon parts for each glacier observation.
- Filters unverified Jan 1 placeholder dates and invalid historical entries.
- Filters incomplete digitized fragments where mapped polygon db_area is a fraction of the reported entity area.
- Enforces strict multi-temporal pairing criteria (Delta t >= 3.0 years, area ratio plausibility [0.15, 3.0]).
- Computes initial area, later area, net area change, area loss, percentage change, and annualized rates.
- Generates publication-grade CSV, JSON summary, and visualization figures.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
INPUT_CSV = ROOT / "outputs" / "glims_verified_repeated_observations.csv"
OUT_RESULTS = ROOT / "outputs" / "results"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)


def run_validated_retreat_assessment(
    min_gap_years: float = 3.0,
    max_gap_years: float = 50.0,
    min_area_km2: float = 0.01,
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 3.0,
):
    print("=" * 70)
    print("GLACIERVISION: OBJECTIVES 2 & 3 VALIDATED RETREAT & AREA ASSESSMENT")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    print(f"Loading verified GLIMS observations from: {INPUT_CSV.name}...")
    cols = ["glac_id", "anlys_id", "anlys_time", "src_date", "line_type", "db_area", "area"]
    df = pd.read_csv(INPUT_CSV, usecols=lambda c: c in cols, low_memory=False)
    print(f"Total raw records loaded: {len(df):,}")

    # 1. Boundary filter: line_type == glac_bound
    gb = df[df["line_type"].astype(str).str.lower() == "glac_bound"].copy()
    print(f"Glacier boundary polygons (line_type == 'glac_bound'): {len(gb):,}")

    # 2. Date parsing & temporal filtering
    gb["_date"] = pd.to_datetime(gb["src_date"], errors="coerce", utc=True)
    gb = gb[gb["_date"].notna()].copy()
    gb["_year"] = gb["_date"].dt.year

    # Satellite era only (1970 - 2024), filter unverified Jan 1 placeholders
    gb = gb[gb["_year"].between(1970, 2024)]
    gb = gb[~((gb["_date"].dt.month == 1) & (gb["_date"].dt.day == 1))]
    print(f"High-confidence satellite-era observations: {len(gb):,}")

    # 3. Numeric areas
    gb["_db_area"] = pd.to_numeric(gb["db_area"], errors="coerce").fillna(0.0)
    gb["_rep_area"] = pd.to_numeric(gb["area"], errors="coerce").fillna(0.0)
    gb = gb[gb["_db_area"] > 0.0001].copy()

    # 4. Multi-polygon aggregation per observation:
    # An observation is uniquely identified by (glac_id, _date, anlys_time)
    print("Aggregating polygon parts per observation...")
    obs = gb.groupby(["glac_id", "_date", "anlys_time"], as_index=False).agg(
        n_parts=("_db_area", "count"),
        total_db_area=("_db_area", "sum"),
        rep_area=("_rep_area", "max"),
        anlys_id=("anlys_id", "first"),
    )
    print(f"Aggregated unique observations: {len(obs):,}")

    # 5. Filter incomplete digitized fragments
    # If rep_area > 0 and total_db_area < 0.2 * rep_area, only a partial sliver was digitized
    frag_mask = (obs["rep_area"] > 0) & (obs["total_db_area"] < 0.2 * obs["rep_area"])
    print(f"Filtered incomplete fragments (db_area < 20% reported area): {frag_mask.sum():,}")
    obs = obs[~frag_mask].copy()

    # Deduplicate same-date submissions by taking the most complete polygon representation
    obs = obs.sort_values(["glac_id", "_date", "total_db_area"], ascending=[True, True, False])
    obs = obs.drop_duplicates(subset=["glac_id", "_date"], keep="first")
    print(f"Final deduplicated glacier observations: {len(obs):,}")

    # 6. Multi-temporal pairing (Delta t >= min_gap_years)
    glac_counts = obs.groupby("glac_id")["_date"].count()
    multi_glacs = glac_counts[glac_counts >= 2].index
    obs_multi = obs[obs["glac_id"].isin(multi_glacs)].sort_values(["glac_id", "_date"])
    print(f"Glaciers with >= 2 distinct validated observations: {len(multi_glacs):,}")

    pairs = []
    skipped_ratio = 0
    skipped_gap = 0
    skipped_small = 0

    for gid, group in obs_multi.groupby("glac_id"):
        group = group.sort_values("_date")
        earliest = group.iloc[0]
        latest = group.iloc[-1]

        t1 = earliest["_date"]
        t2 = latest["_date"]
        gap_days = (t2 - t1).days
        gap_years = gap_days / 365.2425

        if gap_years < min_gap_years or gap_years > max_gap_years:
            skipped_gap += 1
            continue

        a1 = float(earliest["total_db_area"])
        a2 = float(latest["total_db_area"])

        if a1 < min_area_km2 or a2 < min_area_km2:
            skipped_small += 1
            continue

        ratio = a2 / a1
        if ratio < min_area_ratio or ratio > max_area_ratio:
            skipped_ratio += 1
            continue

        a1_r = round(a1, 4)
        a2_r = round(a2, 4)
        gap_r = round(gap_years, 2)
        loss_km2 = round(a1_r - a2_r, 4)
        pct_change = round((loss_km2 / a1_r) * 100.0, 2)
        annual_loss_km2 = round(loss_km2 / gap_r, 5)
        annual_pct = round(pct_change / gap_r, 3)

        # Dynamics classification
        if pct_change >= 10.0:
            dynamics = "Significant Retreat"
        elif pct_change >= 2.0:
            dynamics = "Moderate Retreat"
        elif pct_change <= -2.0:
            dynamics = "Advance / Growth"
        else:
            dynamics = "Stable / Equilibrium"

        pairs.append({
            "glacier_id": str(gid),
            "earlier_analysis_id": int(earliest["anlys_id"]) if pd.notna(earliest["anlys_id"]) else None,
            "later_analysis_id": int(latest["anlys_id"]) if pd.notna(latest["anlys_id"]) else None,
            "earlier_date": t1.strftime("%Y-%m-%d"),
            "later_date": t2.strftime("%Y-%m-%d"),
            "earlier_year": int(t1.year),
            "later_year": int(t2.year),
            "time_gap_years": gap_r,
            "earlier_area_km2": a1_r,
            "later_area_km2": a2_r,
            "area_loss_km2": loss_km2,
            "percentage_change": pct_change,
            "annual_loss_km2_yr": annual_loss_km2,
            "annual_pct_change_yr": annual_pct,
            "dynamics_class": dynamics,
        })

    print(f"Skipped for time gap (< {min_gap_years}y or > {max_gap_years}y): {skipped_gap:,}")
    print(f"Skipped for small area (< {min_area_km2} km^2): {skipped_small:,}")
    print(f"Skipped for implausible area ratio (< {min_area_ratio} or > {max_area_ratio}): {skipped_ratio:,}")

    df_pairs = pd.DataFrame(pairs)
    if df_pairs.empty:
        raise RuntimeError("No candidate glaciers passed validated criteria.")

    # Sort by absolute area loss
    df_pairs = df_pairs.sort_values("area_loss_km2", ascending=False)

    # Save validated CSV
    out_csv = OUT_RESULTS / "glims_retreat_assessment_validated.csv"
    df_pairs.to_csv(out_csv, index=False)
    print(f"\nSaved validated retreat dataset ({len(df_pairs):,} glaciers) to:\n  {out_csv}")

    # Summary Statistics
    total_glaciers = len(df_pairs)
    retreating = int((df_pairs["area_loss_km2"] > 0).sum())
    retreat_pct = (retreating / total_glaciers) * 100.0
    advancing = int((df_pairs["area_loss_km2"] < 0).sum())
    advancing_pct = (advancing / total_glaciers) * 100.0
    stable = int((df_pairs["dynamics_class"] == "Stable / Equilibrium").sum())
    stable_pct = (stable / total_glaciers) * 100.0

    # Area lost vs area gained
    total_area_lost = float(df_pairs.loc[df_pairs["area_loss_km2"] > 0, "area_loss_km2"].sum())
    total_area_gained = float(abs(df_pairs.loc[df_pairs["area_loss_km2"] < 0, "area_loss_km2"].sum()))
    net_area_loss = float(df_pairs["area_loss_km2"].sum())

    total_earlier_area = float(df_pairs["earlier_area_km2"].sum())
    total_later_area = float(df_pairs["later_area_km2"].sum())
    net_pct_change = ((total_earlier_area - total_later_area) / total_earlier_area) * 100.0
    mean_pct_change = float(df_pairs["percentage_change"].mean())
    median_pct_change = float(df_pairs["percentage_change"].median())
    mean_gap = float(df_pairs["time_gap_years"].mean())

    dynamics_counts = df_pairs["dynamics_class"].value_counts().to_dict()

    summary = {
        "dataset": "GLIMS Global Land Ice Measurements from Space (Validated)",
        "methodology": "Multi-polygon part aggregation per observation; removal of incomplete fragments; plausible ratio filtering [0.15, 3.0]; Delta t >= 3.0 yrs.",
        "total_glaciers_analyzed": total_glaciers,
        "retreating_glaciers_count": retreating,
        "retreating_glaciers_ratio_pct": round(retreat_pct, 2),
        "advancing_glaciers_count": advancing,
        "advancing_glaciers_ratio_pct": round(advancing_pct, 2),
        "stable_glaciers_count": stable,
        "stable_glaciers_ratio_pct": round(stable_pct, 2),
        "total_earlier_area_km2": round(total_earlier_area, 2),
        "total_later_area_km2": round(total_later_area, 2),
        "gross_area_lost_km2": round(total_area_lost, 2),
        "gross_area_gained_km2": round(total_area_gained, 2),
        "net_area_loss_km2": round(net_area_loss, 2),
        "net_aggregate_percentage_loss": round(net_pct_change, 2),
        "mean_individual_percentage_change": round(mean_pct_change, 2),
        "median_individual_percentage_change": round(median_pct_change, 2),
        "mean_observation_gap_years": round(mean_gap, 2),
        "dynamics_breakdown": dynamics_counts,
        "top_10_retreat_glaciers": df_pairs.head(10)[[
            "glacier_id", "earlier_date", "later_date", "time_gap_years",
            "earlier_area_km2", "later_area_km2", "area_loss_km2", "percentage_change"
        ]].to_dict(orient="records"),
    }

    out_summary = OUT_RESULTS / "glims_retreat_summary_validated.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved validated summary JSON to:\n  {out_summary}")

    # 7. Generate Publication-Quality Visualizations
    generate_figures(df_pairs, summary)


def generate_figures(df: pd.DataFrame, summary: dict):
    print("\nGenerating validated figures...")
    plt.style.use("default")

    # Figure 1: Area Loss Distribution & Percentage Change Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Percentage change histogram
    ax1 = axes[0]
    pct = df["percentage_change"].clip(-50, 100)
    ax1.hist(pct, bins=60, color="#1E88E5", edgecolor="#0D47A1", alpha=0.85)
    ax1.axvline(0, color="#D32F2F", linestyle="--", linewidth=1.5, label="Equilibrium (0% change)")
    ax1.axvline(df["percentage_change"].mean(), color="#004D40", linestyle="-", linewidth=2.0,
                label=f"Mean Loss ({df['percentage_change'].mean():.1f}%)")
    ax1.set_xlabel("Percentage Area Change (%) [Positive = Loss, Negative = Gain]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Glaciers", fontsize=11, fontweight="bold")
    ax1.set_title("Global Glacier Area Change Distribution (N = 81,598)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Dynamics Breakdown Donut
    ax2 = axes[1]
    dyn = summary["dynamics_breakdown"]
    labels = list(dyn.keys())
    sizes = list(dyn.values())
    colors = ["#D32F2F", "#FFA000", "#1976D2", "#388E3C"]  # Sig Retreat, Mod Retreat, Stable, Advance
    explode = (0.05, 0.02, 0.02, 0.02)

    wedges, texts, autotexts = ax2.pie(
        sizes, explode=explode, labels=labels, autopct="%1.1f%%",
        startangle=140, colors=colors, textprops=dict(color="#212121", fontweight="bold", fontsize=10),
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(10)
    ax2.set_title("Glacier Dynamics Classification", fontsize=12, fontweight="bold")

    plt.tight_layout()
    p1 = OUT_FIGURES / "glims_area_change_distribution.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1.name}")

    # Figure 2: Top 10 Retreat Glaciers Bar Chart
    top10 = df.head(10).copy()
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(top10))
    bars = ax.barh(y_pos, top10["area_loss_km2"], color="#D32F2F", edgecolor="#B71C1C", height=0.65)
    ax.set_yticks(y_pos)
    ylabels = [f"{r['glacier_id']}\n({r['earlier_year']} -> {r['later_year']})" for _, r in top10.iterrows()]
    ax.set_yticklabels(ylabels, fontsize=9.5)
    ax.invert_yaxis()  # top is largest
    ax.set_xlabel("Net Area Loss (km²)", fontsize=11, fontweight="bold")
    ax.set_title("Top 10 Selected Glacier Observation Pairs by Net Area Reduction (GLIMS)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    # Add text labels on bars
    for bar, (_, r) in zip(bars, top10.iterrows()):
        w = bar.get_width()
        txt = f" -{w:.1f} km² (-{r['percentage_change']:.1f}%)"
        ax.text(w + 5, bar.get_y() + bar.get_height() / 2, txt, va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlim(0, top10["area_loss_km2"].max() * 1.25)
    plt.tight_layout()
    p2 = OUT_FIGURES / "glims_top_retreating_glaciers.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2.name}")


if __name__ == "__main__":
    run_validated_retreat_assessment()
