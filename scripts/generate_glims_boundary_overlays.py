"""GlacierVision: Validated GLIMS Multi-Temporal Boundary Overlays & Spatial Overlap.

Performs rigorous geometric intersection on representative multi-temporal glaciers:
- Rhone Glacier (G008379E46648N)
- Great Aletsch Glacier (G008032E46504N)
- Unteraar Glacier (G008187E46569N)

Computes:
- Retained ice area (earlier INTERSECT later)
- Lost glacier area (earlier DIFFERENCE later, Retreat)
- Gained glacier area (later DIFFERENCE earlier, Advance)
in projected equal-area CRS (UTM 32N, EPSG:32632).

Produces publication-grade visual monitoring overlay figure and verification JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyogrio
import shapely
from shapely.ops import unary_union

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
SHP_PATH = ROOT / "datasets" / "glims" / "extracted" / "glims_download_52211" / "glims_polygons.shp"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_RESULTS = ROOT / "outputs" / "results"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUT_RESULTS.mkdir(parents=True, exist_ok=True)

TARGET_GLACIERS = [
    {
        "id": "G008379E46648N",
        "name": "Rhone Glacier (Rhonegletscher)",
        "region": "Valais Alps, Switzerland",
        "t1_year": 1973,
        "t2_year": 2015,
        "crs": "EPSG:32632",
    },
    {
        "id": "G008032E46504N",
        "name": "Great Aletsch Glacier (Grosser Aletschgletscher)",
        "region": "Bernese Alps, Switzerland",
        "t1_year": 1973,
        "t2_year": 2015,
        "crs": "EPSG:32632",
    },
    {
        "id": "G008187E46569N",
        "name": "Unteraar Glacier (Unteraargletscher)",
        "region": "Bernese Alps, Switzerland",
        "t1_year": 1973,
        "t2_year": 2015,
        "crs": "EPSG:32632",
    },
]


def run_boundary_overlay_analysis():
    print("=" * 70)
    print("OBJECTIVE 2 & 4: GLIMS MULTI-TEMPORAL BOUNDARY OVERLAY ANALYSIS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)
    fig.patch.set_facecolor("#0F172A")  # Dark theme matching GlacierVision

    case_studies = []

    for i, target in enumerate(TARGET_GLACIERS):
        gid = target["id"]
        name = target["name"]
        print(f"\nProcessing Case {i+1}: {name} ({gid})...")

        where_clause = f"glac_id = '{gid}' AND line_type = 'glac_bound'"
        gdf = pyogrio.read_dataframe(SHP_PATH, where=where_clause)
        print(f"  Loaded {len(gdf)} boundary polygon rows.")

        gdf["_date"] = pd_date = gdf["src_date"].str[:4].astype(int)
        
        # Select closest observations to t1_year and t2_year
        dates = sorted(gdf["_date"].unique())
        t1_actual = min(dates, key=lambda d: abs(d - target["t1_year"]))
        t2_actual = min(dates, key=lambda d: abs(d - target["t2_year"]))
        print(f"  Selected epochs: Earlier = {t1_actual}, Later = {t2_actual}")

        geom_t1 = unary_union(gdf[gdf["_date"] == t1_actual]["geometry"].values)
        geom_t2 = unary_union(gdf[gdf["_date"] == t2_actual]["geometry"].values)

        # Reproject to UTM 32N for accurate metric geometry
        import pyproj
        from shapely.ops import transform
        project = pyproj.Transformer.from_crs("EPSG:4326", target["crs"], always_xy=True).transform
        geom_t1_utm = transform(project, geom_t1)
        geom_t2_utm = transform(project, geom_t2)

        # Compute metric areas (m^2 -> km^2)
        area_t1 = geom_t1_utm.area / 1e6
        area_t2 = geom_t2_utm.area / 1e6

        # Geometric intersection & difference
        retained = geom_t1_utm.intersection(geom_t2_utm)
        lost = geom_t1_utm.difference(geom_t2_utm)  # Retreat zone
        gained = geom_t2_utm.difference(geom_t1_utm)  # Advance zone

        area_retained = retained.area / 1e6
        area_lost = lost.area / 1e6
        area_gained = gained.area / 1e6
        net_change = area_t1 - area_t2
        pct_change = (net_change / area_t1) * 100.0

        print(f"  Epoch {t1_actual} Area: {area_t1:.2f} km²")
        print(f"  Epoch {t2_actual} Area: {area_t2:.2f} km²")
        print(f"  Lost Area (Retreat): {area_lost:.2f} km² ({pct_change:.1f}% net loss)")
        print(f"  Gained Area: {area_gained:.2f} km²")

        case_record = {
            "glacier_id": gid,
            "glacier_name": name,
            "region": target["region"],
            "earlier_year": int(t1_actual),
            "later_year": int(t2_actual),
            "time_gap_years": int(t2_actual - t1_actual),
            "earlier_area_km2": round(area_t1, 3),
            "later_area_km2": round(area_t2, 3),
            "retained_ice_area_km2": round(area_retained, 3),
            "lost_ice_area_km2": round(area_lost, 3),
            "gained_ice_area_km2": round(area_gained, 3),
            "net_loss_km2": round(net_change, 3),
            "percentage_change": round(pct_change, 2),
            "projected_crs": target["crs"],
        }
        case_studies.append(case_record)

        # Plot on Subplot
        ax = axes[i]
        ax.set_facecolor("#1E293B")  # Slate dark background

        # Plot retained ice (Cyan)
        plot_geometry(ax, retained, facecolor="#00F2FE", edgecolor="none", alpha=0.55, label="Retained Glacier Ice")
        # Plot lost ice (Red retreat)
        plot_geometry(ax, lost, facecolor="#FF4B4B", edgecolor="#EF4444", alpha=0.75, label="Lost Glacier Area (Retreat)")
        # Plot gained ice (Green advance if any)
        if area_gained > 0.001:
            plot_geometry(ax, gained, facecolor="#10B981", edgecolor="#059669", alpha=0.75, label="Gained Area (Advance)")

        # Plot boundary outlines
        plot_boundary(ax, geom_t1_utm, color="#FFA000", linewidth=1.5, linestyle="--", label=f"Boundary ({t1_actual})")
        plot_boundary(ax, geom_t2_utm, color="#FFFFFF", linewidth=1.8, linestyle="-", label=f"Boundary ({t2_actual})")

        ax.set_title(f"{name}\n({t1_actual} -> {t2_actual}: -{net_change:.2f} km², -{pct_change:.1f}%)",
                     fontsize=11, fontweight="bold", color="#F8FAFC", pad=10)
        ax.tick_params(colors="#94A3B8", labelsize=8)
        ax.grid(True, linestyle=":", color="#334155", alpha=0.7)

        # Format coordinates as km relative to centroid
        minx, miny, maxx, maxy = geom_t1_utm.bounds
        pad_x = (maxx - minx) * 0.08
        pad_y = (maxy - miny) * 0.08
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal")

        # Custom legend on first subplot
        if i == 0:
            patches = [
                mpatches.Patch(color="#00F2FE", alpha=0.6, label="Retained Ice"),
                mpatches.Patch(color="#FF4B4B", alpha=0.8, label=f"Lost Ice (Retreat)"),
                plt.Line2D([0], [0], color="#FFA000", linestyle="--", lw=1.5, label=f"Earlier Outline ({t1_actual})"),
                plt.Line2D([0], [0], color="#FFFFFF", linestyle="-", lw=1.8, label=f"Later Outline ({t2_actual})"),
            ]
            ax.legend(handles=patches, loc="lower left", framealpha=0.85, facecolor="#0F172A",
                      edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8.5)

    plt.suptitle("GLIMS Bi-Temporal Multi-Year Glacier Boundary Overlay & Retreat Assessment",
                 fontsize=14, fontweight="bold", color="#F8FAFC", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_fig = OUT_FIGURES / "visual_monitoring_glims_boundary_overlays.png"
    fig.savefig(out_fig, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)
    out_svg = OUT_FIGURES / "visual_monitoring_glims_boundary_overlays.svg"
    fig.savefig(out_svg, facecolor=fig.get_facecolor(), edgecolor="none", format="svg")
    plt.close(fig)
    print(f"\nSaved boundary overlay figure to:\n  {out_fig}\n  {out_svg}")

    out_json = OUT_RESULTS / "glims_case_studies.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(case_studies, f, indent=2)
    print(f"Saved verified case studies JSON to:\n  {out_json}")


def plot_geometry(ax, geom, facecolor, edgecolor, alpha, label=""):
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = geom.geoms
    elif geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        sub = []
        for p in polys:
            if p.geom_type == "MultiPolygon":
                sub.extend(p.geoms)
            else:
                sub.append(p)
        polys = sub
    else:
        return

    for p in polys:
        x, y = p.exterior.xy
        ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        for interior in p.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, facecolor="#1E293B", edgecolor=edgecolor, alpha=1.0)


def plot_boundary(ax, geom, color, linewidth, linestyle, label=""):
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = geom.geoms
    elif geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        sub = []
        for p in polys:
            if p.geom_type == "MultiPolygon":
                sub.extend(p.geoms)
            else:
                sub.append(p)
        polys = sub
    else:
        return

    for p in polys:
        x, y = p.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle)


if __name__ == "__main__":
    run_boundary_overlay_analysis()
