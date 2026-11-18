"""Generate publication-grade individual GLIMS boundary overlay figures.
Produces both:
- High-resolution raster PNGs (300 DPI)
- Scalable vector graphics (SVG)
for:
- Rhone Glacier (G008379E46648N)
- Great Aletsch Glacier (G008032E46504N)
- Unteraar Glacier (G008187E46569N)
- Gornergletscher (G007800E45965N)
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import pyproj
import shapely
from shapely.ops import transform, unary_union

ROOT = Path(os.environ["USERPROFILE"]) / "Desktop" / "GlacierResearch"
SHP_PATH = ROOT / "datasets" / "glims" / "extracted" / "glims_download_52211" / "glims_polygons.shp"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {
        "id": "G008379E46648N",
        "key": "rhone",
        "name": "Rhone Glacier (Rhonegletscher)",
        "region": "Valais Alps, Switzerland",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
        "crs": "EPSG:32632",
        "out_base": "glims_boundary_rhone",
    },
    {
        "id": "G008032E46504N",
        "key": "great_aletsch",
        "name": "Great Aletsch Glacier (Grosser Aletschgletscher)",
        "region": "Bernese Alps, Switzerland",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
        "crs": "EPSG:32632",
        "out_base": "glims_boundary_great_aletsch",
    },
    {
        "id": "G008187E46569N",
        "key": "unteraar",
        "name": "Unteraar Glacier (Unteraargletscher)",
        "region": "Bernese Alps, Switzerland",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
        "crs": "EPSG:32632",
        "out_base": "glims_boundary_unteraar",
    },
    {
        "id": "G007800E45965N",
        "key": "gorner",
        "name": "Gorner Glacier (Gornergletscher)",
        "region": "Pennine Alps, Switzerland",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
        "crs": "EPSG:32632",
        "out_base": "glims_boundary_gorner",
    },
]


def plot_geometry(ax, geom, facecolor, edgecolor, alpha=1.0):
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
            ax.fill(ix, iy, facecolor="#0B132B", edgecolor=edgecolor, alpha=1.0)


def plot_boundary(ax, geom, color, linewidth, linestyle):
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


def generate_maps():
    print("Starting individual boundary map generation (Raster PNG & Vector SVG)...")
    for target in TARGETS:
        gid = target["id"]
        name = target["name"]
        print(f"Processing {name} ({gid})...")

        where_clause = f"glac_id = '{gid}' AND line_type = 'glac_bound'"
        gdf = pyogrio.read_dataframe(SHP_PATH, where=where_clause)
        gdf["_date"] = gdf["src_date"].str[:10]

        e_df = gdf[gdf["_date"] == target["t1_date"]]
        l_df = gdf[gdf["_date"] == target["t2_date"]]

        geom_e = unary_union(e_df.geometry.values)
        geom_l = unary_union(l_df.geometry.values)

        project = pyproj.Transformer.from_crs("EPSG:4326", target["crs"], always_xy=True).transform
        geom_e_utm = transform(project, geom_e)
        geom_l_utm = transform(project, geom_l)

        area_e = geom_e_utm.area / 1e6
        area_l = geom_l_utm.area / 1e6
        retained = geom_e_utm.intersection(geom_l_utm)
        lost = geom_e_utm.difference(geom_l_utm)
        gained = geom_l_utm.difference(geom_e_utm)

        area_ret = retained.area / 1e6
        area_lost = lost.area / 1e6
        area_gain = gained.area / 1e6
        net_loss = area_e - area_l
        pct_loss = (net_loss / area_e) * 100.0

        fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
        fig.patch.set_facecolor("#0B132B")
        ax.set_facecolor("#111E38")

        # Plot geometry zones
        plot_geometry(ax, retained, facecolor="#00F2FE", edgecolor="none", alpha=0.55)
        plot_geometry(ax, lost, facecolor="#FF4B4B", edgecolor="#EF4444", alpha=0.80)
        if area_gain > 0.005:
            plot_geometry(ax, gained, facecolor="#10B981", edgecolor="#059669", alpha=0.80)

        plot_boundary(ax, geom_e_utm, color="#F59E0B", linewidth=1.8, linestyle="--")
        plot_boundary(ax, geom_l_utm, color="#FFFFFF", linewidth=2.0, linestyle="-")

        # Bounding box & aspect
        minx, miny, maxx, maxy = geom_e_utm.bounds
        l_minx, l_miny, l_maxx, l_maxy = geom_l_utm.bounds
        minx = min(minx, l_minx)
        miny = min(miny, l_miny)
        maxx = max(maxx, l_maxx)
        maxy = max(maxy, l_maxy)

        padx = (maxx - minx) * 0.08
        pady = (maxy - miny) * 0.08
        ax.set_xlim(minx - padx, maxx + padx)
        ax.set_ylim(miny - pady, maxy + pady)
        ax.set_aspect("equal")

        # Format ticks to km
        ax.tick_params(colors="#94A3B8", labelsize=8)
        ax.grid(True, linestyle=":", color="#2A3D66", alpha=0.6)

        # Format axes as km offset
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{(val - cx)/1e3:+.1f} km"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{(val - cy)/1e3:+.1f} km"))
        ax.set_xlabel("Relative Easting (km)", color="#94A3B8", fontsize=9)
        ax.set_ylabel("Relative Northing (km)", color="#94A3B8", fontsize=9)

        # Title and Subtitle - accurately describing measured change without unsupported p-value
        t1_yr = target["t1_date"][:4]
        t2_yr = target["t2_date"][:4]
        ax.set_title(
            f"{name} ({gid})\n"
            f"Multi-Temporal Boundary Overlay ({t1_yr} vs {t2_yr}) | Net Loss: {net_loss:.3f} km² (-{pct_loss:.2f}%)",
            fontsize=11.5,
            fontweight="bold",
            color="#F8FAFC",
            pad=12,
        )

        # Legend
        handles = [
            mpatches.Patch(color="#00F2FE", alpha=0.60, label=f"Retained Ice: {area_ret:.3f} km²"),
            mpatches.Patch(color="#FF4B4B", alpha=0.80, label=f"Lost Ice (Retreat): {area_lost:.3f} km²"),
        ]
        if area_gain > 0.005:
            handles.append(mpatches.Patch(color="#10B981", alpha=0.80, label=f"Gained Ice: {area_gain:.3f} km²"))
        handles.extend([
            plt.Line2D([0], [0], color="#F59E0B", linestyle="--", lw=1.8, label=f"Baseline Boundary ({t1_yr}): {area_e:.3f} km²"),
            plt.Line2D([0], [0], color="#FFFFFF", linestyle="-", lw=2.0, label=f"Modern Boundary ({t2_yr}): {area_l:.3f} km²"),
        ])
        ax.legend(
            handles=handles,
            loc="upper right",
            framealpha=0.90,
            facecolor="#0B132B",
            edgecolor="#334155",
            labelcolor="#F8FAFC",
            fontsize=8.5,
        )

        plt.tight_layout()

        # 1. Save High-Resolution Raster PNG (300 DPI)
        out_png = OUT_FIGURES / f"{target['out_base']}.png"
        fig.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)
        print(f"  Saved raster PNG (300 DPI): {out_png.name}")

        # 2. Save True Scalable Vector Graphics (SVG)
        out_svg = OUT_FIGURES / f"{target['out_base']}.svg"
        fig.savefig(out_svg, facecolor=fig.get_facecolor(), edgecolor="none", format="svg")
        print(f"  Saved vector SVG: {out_svg.name}")

        plt.close(fig)


if __name__ == "__main__":
    generate_maps()
