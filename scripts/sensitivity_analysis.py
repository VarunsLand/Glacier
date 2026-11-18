"""GlacierVision: Comprehensive Sensitivity Analysis of Filtering Thresholds.

Systematically evaluates:
1. Fragment removal threshold (db_area / rep_area < tau_frag):
   tau_frag in [0.0 (none), 0.05, 0.10, 0.20 (baseline), 0.30, 0.50]
2. Area ratio bounds (A_later / A_earlier in [r_min, r_max]):
   bounds in [None (raw), [0.05, 10.0], [0.10, 5.0], [0.15, 3.0] (baseline), [0.25, 2.0], [0.50, 1.5]]

Outputs:
- outputs/results/filter_sensitivity_analysis.csv
- outputs/results/filter_sensitivity_analysis.json
- outputs/figures/filter_sensitivity_curves.png
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
INPUT_CSV = ROOT / "outputs" / "glims_verified_repeated_observations.csv"
OUT_RESULTS = ROOT / "outputs" / "results"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)


def load_base_observations():
    print("Loading raw GLIMS boundary observations...")
    cols = ["glac_id", "anlys_id", "anlys_time", "src_date", "line_type", "db_area", "area"]
    df = pd.read_csv(INPUT_CSV, usecols=lambda c: c in cols, low_memory=False)
    
    # Line type filter
    gb = df[df["line_type"].astype(str).str.lower() == "glac_bound"].copy()
    
    # Dates
    gb["_date"] = pd.to_datetime(gb["src_date"], errors="coerce", utc=True)
    gb = gb[gb["_date"].notna()].copy()
    gb["_year"] = gb["_date"].dt.year
    gb = gb[gb["_year"].between(1970, 2024)]
    gb = gb[~((gb["_date"].dt.month == 1) & (gb["_date"].dt.day == 1))]
    
    # Numeric areas
    gb["_db_area"] = pd.to_numeric(gb["db_area"], errors="coerce").fillna(0.0)
    gb["_rep_area"] = pd.to_numeric(gb["area"], errors="coerce").fillna(0.0)
    gb = gb[gb["_db_area"] > 0.0001].copy()
    
    # Aggregate polygon parts per observation
    obs = gb.groupby(["glac_id", "_date", "anlys_time"], as_index=False).agg(
        n_parts=("_db_area", "count"),
        total_db_area=("_db_area", "sum"),
        rep_area=("_rep_area", "max"),
        anlys_id=("anlys_id", "first"),
    )
    return obs


def evaluate_pipeline(obs_base: pd.DataFrame, frag_threshold: float, ratio_bounds: tuple[float, float] | None):
    obs = obs_base.copy()
    
    # 1. Fragment filter
    if frag_threshold > 0:
        frag_mask = (obs["rep_area"] > 0) & (obs["total_db_area"] < frag_threshold * obs["rep_area"])
        n_frag_removed = int(frag_mask.sum())
        obs = obs[~frag_mask].copy()
    else:
        n_frag_removed = 0
        
    # Deduplicate same-date submissions
    obs = obs.sort_values(["glac_id", "_date", "total_db_area"], ascending=[True, True, False])
    obs = obs.drop_duplicates(subset=["glac_id", "_date"], keep="first")
    
    # Multi-temporal pairing
    glac_counts = obs.groupby("glac_id")["_date"].count()
    multi_glacs = glac_counts[glac_counts >= 2].index
    obs_multi = obs[obs["glac_id"].isin(multi_glacs)].sort_values(["glac_id", "_date"])
    
    pairs = []
    n_ratio_removed = 0
    
    for gid, group in obs_multi.groupby("glac_id"):
        group = group.sort_values("_date")
        earliest = group.iloc[0]
        latest = group.iloc[-1]
        
        t1 = earliest["_date"]
        t2 = latest["_date"]
        gap_years = (t2 - t1).days / 365.2425
        if gap_years < 3.0 or gap_years > 50.0:
            continue
            
        a1 = float(earliest["total_db_area"])
        a2 = float(latest["total_db_area"])
        if a1 < 0.01 or a2 < 0.01:
            continue
            
        ratio = a2 / a1
        if ratio_bounds is not None:
            if ratio < ratio_bounds[0] or ratio > ratio_bounds[1]:
                n_ratio_removed += 1
                continue
                
        loss = a1 - a2
        pct = (loss / a1) * 100.0
        
        pairs.append({
            "glac_id": gid,
            "earlier_area": a1,
            "later_area": a2,
            "loss": loss,
            "pct": pct,
            "ratio": ratio
        })
        
    df_p = pd.DataFrame(pairs)
    if df_p.empty:
        return {}
        
    total_glac = len(df_p)
    gross_loss = float(df_p.loc[df_p["loss"] > 0, "loss"].sum())
    gross_gain = float(abs(df_p.loc[df_p["loss"] < 0, "loss"].sum()))
    net_loss = float(df_p["loss"].sum())
    
    # Mutually exclusive partitions:
    # High-level 3-class partition (2% deadband)
    ret_count = int((df_p["pct"] >= 2.0).sum())
    adv_count = int((df_p["pct"] <= -2.0).sum())
    sta_count = int(((df_p["pct"] > -2.0) & (df_p["pct"] < 2.0)).sum())
    
    return {
        "frag_threshold": frag_threshold,
        "ratio_bounds": str(ratio_bounds) if ratio_bounds else "None",
        "fragments_removed": n_frag_removed,
        "ratio_outliers_removed": n_ratio_removed,
        "total_glaciers": total_glac,
        "retreating_glaciers": ret_count,
        "retreating_pct": round(ret_count / total_glac * 100.0, 2),
        "advancing_glaciers": adv_count,
        "advancing_pct": round(adv_count / total_glac * 100.0, 2),
        "stable_glaciers": sta_count,
        "stable_pct": round(sta_count / total_glac * 100.0, 2),
        "sum_pcts": round((ret_count + adv_count + sta_count) / total_glac * 100.0, 2),
        "gross_area_lost_km2": round(gross_loss, 2),
        "gross_area_gained_km2": round(gross_gain, 2),
        "net_area_loss_km2": round(net_loss, 2),
        "mean_pct_change": round(float(df_p["pct"].mean()), 2),
        "median_pct_change": round(float(df_p["pct"].median()), 2),
    }


def run_sensitivity_suite():
    print("=" * 75)
    print("OBJECTIVE 3: FILTER SENSITIVITY & ABLATION ANALYSIS")
    print("=" * 75)
    
    obs_base = load_base_observations()
    print(f"Base unique observations: {len(obs_base):,}")
    
    # 1. Experiment Suite A: Varying Fragment Removal Threshold (Fixed ratio bounds [0.15, 3.0])
    frag_thresholds = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    results_frag = []
    print("\nRunning Fragment Threshold Sensitivity Suite...")
    for ft in frag_thresholds:
        res = evaluate_pipeline(obs_base, frag_threshold=ft, ratio_bounds=(0.15, 3.0))
        results_frag.append(res)
        print(f"  tau_frag = {ft:4.2f} -> Basins: {res['total_glaciers']:,} | Frags Removed: {res['fragments_removed']:,} | Net Loss: {res['net_area_loss_km2']:,.1f} km²")
        
    # 2. Experiment Suite B: Varying Area Ratio Bounds (Fixed frag threshold 0.20)
    ratio_options = [
        None,             # No bounds
        (0.05, 10.0),     # Extremely loose
        (0.10, 5.0),      # Loose
        (0.15, 3.0),      # Baseline
        (0.25, 2.0),      # Strict
        (0.50, 1.5),      # Very strict
    ]
    results_ratio = []
    print("\nRunning Area Ratio Bounds Sensitivity Suite...")
    for rb in ratio_options:
        res = evaluate_pipeline(obs_base, frag_threshold=0.20, ratio_bounds=rb)
        results_ratio.append(res)
        lbl = str(rb) if rb else "None (Raw)"
        print(f"  Ratio = {lbl:16s} -> Basins: {res['total_glaciers']:,} | Outliers Removed: {res['ratio_outliers_removed']:,} | Net Loss: {res['net_area_loss_km2']:,.1f} km²")

    # Combine into full report
    all_experiments = {
        "fragment_threshold_sensitivity": results_frag,
        "ratio_bounds_sensitivity": results_ratio,
    }
    
    out_json = OUT_RESULTS / "filter_sensitivity_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, indent=2)
    print(f"\nSaved sensitivity JSON to: {out_json}")
    
    df_all = pd.DataFrame(results_frag + results_ratio).drop_duplicates()
    out_csv = OUT_RESULTS / "filter_sensitivity_analysis.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"Saved sensitivity CSV to: {out_csv}")
    
    # 3. Publication Sensitivity Curves Figure
    generate_sensitivity_figures(results_frag, results_ratio)


def generate_sensitivity_figures(results_frag, results_ratio):
    print("Generating sensitivity figures...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    
    # Panel 1: Fragment Threshold Sensitivity
    ax1 = axes[0]
    df_f = pd.DataFrame(results_frag)
    x_f = df_f["frag_threshold"] * 100.0
    
    ax1.plot(x_f, df_f["net_area_loss_km2"], marker="o", color="#D32F2F", linewidth=2.5, label="Net Cryospheric Loss (km²)")
    ax1.set_xlabel("Fragment Removal Threshold (%) [db_area < tau * area]", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Net Area Loss (km²)", fontsize=10.5, fontweight="bold", color="#D32F2F")
    ax1.tick_params(axis="y", labelcolor="#D32F2F")
    ax1.axvline(20.0, color="#1976D2", linestyle="--", linewidth=1.8, label="Selected Baseline (20%)")
    ax1.set_title("A. Sensitivity to Incomplete Fragment Threshold", fontsize=11.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Second y-axis for glacier count
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_f, df_f["total_glaciers"], marker="s", color="#388E3C", linewidth=2, linestyle="--", label="Total Basins Analyzed")
    ax1_twin.set_ylabel("Glacier Basin Count", fontsize=10.5, fontweight="bold", color="#388E3C")
    ax1_twin.tick_params(axis="y", labelcolor="#388E3C")
    
    # Panel 2: Ratio Bounds Sensitivity
    ax2 = axes[1]
    df_r = pd.DataFrame(results_ratio)
    # x labels
    x_labels = [r["ratio_bounds"] for r in results_ratio]
    x_pos = np.arange(len(x_labels))
    
    bars = ax2.bar(x_pos - 0.15, df_r["net_area_loss_km2"], width=0.3, color="#D32F2F", label="Net Loss (km²)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Net Area Loss (km²)", fontsize=10.5, fontweight="bold", color="#D32F2F")
    ax2.set_title("B. Sensitivity to Area Ratio Bounds [r_min, r_max]", fontsize=11.5, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_pos, df_r["total_glaciers"], marker="s", color="#1976D2", linewidth=2, label="Total Glaciers")
    ax2_twin.set_ylabel("Glacier Basin Count", fontsize=10.5, fontweight="bold", color="#1976D2")
    ax2_twin.tick_params(axis="y", labelcolor="#1976D2")
    
    plt.suptitle("GlacierVision Objective 3: Quality Filter Sensitivity & Ablation Analysis",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    out_fig = OUT_FIGURES / "filter_sensitivity_curves.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sensitivity figure to: {out_fig}")


if __name__ == "__main__":
    run_sensitivity_suite()
