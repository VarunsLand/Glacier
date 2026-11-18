import os
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import pyproj
from shapely.ops import transform, unary_union

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
SHP_PATH = ROOT / "datasets" / "glims" / "extracted" / "glims_download_52211" / "glims_polygons.shp"
OUT_FIGURES = ROOT / "outputs" / "figures"

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


def generate_findel_map():
    gid = "G007880E45990N"
    name = "Findel Glacier (Findelgletscher)"
    t1_date = "1973-09-01"
    t2_date = "2015-08-29"
    crs_str = "EPSG:32632"

    where_clause = f"glac_id = '{gid}' AND line_type = 'glac_bound'"
    gdf = pyogrio.read_dataframe(SHP_PATH, where=where_clause)
    gdf["_date"] = gdf["src_date"].str[:10]

    e_df = gdf[gdf["_date"] == t1_date]
    l_df = gdf[gdf["_date"] == t2_date]

    geom_e = unary_union(e_df.geometry.values)
    geom_l = unary_union(l_df.geometry.values)

    project = pyproj.Transformer.from_crs("EPSG:4326", crs_str, always_xy=True).transform
    geom_e_proj = transform(project, geom_e)
    geom_l_proj = transform(project, geom_l)

    area_e = geom_e_proj.area / 1e6
    area_l = geom_l_proj.area / 1e6
    retained = geom_e_proj.intersection(geom_l_proj)
    lost = geom_e_proj.difference(geom_l_proj)
    gained = geom_l_proj.difference(geom_e_proj)

    area_ret = retained.area / 1e6
    area_lost = lost.area / 1e6
    area_gain = gained.area / 1e6
    net_loss = area_e - area_l
    pct_loss = (net_loss / area_e) * 100.0

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    fig.patch.set_facecolor("#0B132B")
    ax.set_facecolor("#111E38")

    plot_geometry(ax, retained, facecolor="#00F2FE", edgecolor="none", alpha=0.55)
    plot_geometry(ax, lost, facecolor="#FF4B4B", edgecolor="#EF4444", alpha=0.80)
    if area_gain > 0.005:
        plot_geometry(ax, gained, facecolor="#10B981", edgecolor="#059669", alpha=0.80)

    plot_boundary(ax, geom_e_proj, color="#F59E0B", linewidth=1.8, linestyle="--")
    plot_boundary(ax, geom_l_proj, color="#FFFFFF", linewidth=2.0, linestyle="-")

    minx, miny, maxx, maxy = geom_e_proj.bounds
    l_minx, l_miny, l_maxx, l_maxy = geom_l_proj.bounds
    minx = min(minx, l_minx)
    miny = min(miny, l_miny)
    maxx = max(maxx, l_maxx)
    maxy = max(maxy, l_maxy)

    padx = (maxx - minx) * 0.10
    pady = (maxy - miny) * 0.10
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)
    ax.set_aspect("equal")

    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.grid(True, linestyle=":", color="#2A3D66", alpha=0.6)

    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{(val - cx)/1e3:+.1f} km"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{(val - cy)/1e3:+.1f} km"))
    ax.set_xlabel("Relative Easting (km)", color="#94A3B8", fontsize=9)
    ax.set_ylabel("Relative Northing (km)", color="#94A3B8", fontsize=9)

    ax.set_title(
        f"{name} ({gid})\n"
        f"Multi-Temporal Boundary Overlay (1973 vs 2015) | Net Loss: {net_loss:.3f} km² (-{pct_loss:.2f}%)",
        fontsize=11.5,
        fontweight="bold",
        color="#F8FAFC",
        pad=12,
    )

    handles = [
        mpatches.Patch(color="#00F2FE", alpha=0.60, label=f"Retained Ice: {area_ret:.3f} km²"),
        mpatches.Patch(color="#FF4B4B", alpha=0.80, label=f"Lost Ice (Retreat): {area_lost:.3f} km²"),
        mpatches.Patch(color="#F59E0B", label=f"Earlier Outline (1973-09-01): {area_e:.3f} km²"),
        mpatches.Patch(color="#FFFFFF", label=f"Modern Outline (2015-08-29): {area_l:.3f} km²"),
    ]
    if area_gain > 0.005:
        handles.insert(2, mpatches.Patch(color="#10B981", alpha=0.80, label=f"Gained Ice: {area_gain:.3f} km²"))

    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=8.5,
        framealpha=0.85,
        facecolor="#0B132B",
        edgecolor="#2A3D66",
        labelcolor="#F8FAFC",
    )

    txt = (
        f"CRS: {crs_str} • Source: GLIMS ID {gid}\n"
        f"Analysis IDs: 342320 (1973) & 1096562/3/4 (2015)\n"
        f"Δt = 41.99 yrs | Annualized Rate: -{net_loss/41.99:.3f} km²/yr"
    )
    ax.text(
        0.02, 0.02, txt,
        transform=ax.transAxes,
        fontsize=8,
        color="#94A3B8",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0B132B", edgecolor="#2A3D66", alpha=0.85)
    )

    plt.tight_layout()

    bases = [f"glims_boundary_{gid}", "glims_boundary_findel", "glims_boundary_findelgletscher"]
    for base in bases:
        png_path = OUT_FIGURES / f"{base}.png"
        svg_path = OUT_FIGURES / f"{base}.svg"
        fig.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        fig.savefig(svg_path, format="svg", facecolor=fig.get_facecolor(), edgecolor="none")
        print(f"Saved: {png_path} and {svg_path}")

    plt.close(fig)
    print("Findel Done!")

if __name__ == "__main__":
    generate_findel_map()
