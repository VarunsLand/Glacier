from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import rasterio
import shapely.wkt

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
BENCHMARK_JSON = ROOT / "datasets" / "glims" / "benchmark_glims_geometries.json"
CASE_STUDIES_JSON = ROOT / "outputs" / "results" / "glims_case_studies.json"


def load_cached_glims_benchmarks() -> Dict[str, Any]:
    """Load benchmark GLIMS geometries and metadata.
    
    Returns authoritative data for:
    - Rhone Glacier (G008379E46648N)
    - Great Aletsch Glacier (G008032E46504N)
    - Unteraar Glacier (G008187E46569N)
    - Gorner Glacier (G007800E45965N)
    - Findel Glacier (G007880E45990N)
    """
    if BENCHMARK_JSON.exists():
        with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    if CASE_STUDIES_JSON.exists():
        with open(CASE_STUDIES_JSON, "r", encoding="utf-8") as f:
            cases = json.load(f)
        return {c["glacier_id"]: c for c in cases}
    return {}


def _plot_shapely_geom(ax, geom, facecolor, edgecolor, alpha=1.0):
    if geom is None or geom.is_empty:
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
            ax.fill(ix, iy, facecolor="#111E38", edgecolor=edgecolor, alpha=1.0)


def _plot_shapely_boundary(ax, geom, color, linewidth, linestyle="-", alpha=1.0):
    if geom is None or geom.is_empty:
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
        ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def render_benchmark_glims_overlay(
    bdata: Dict[str, Any],
    color_hex: str = "#12DFF3",
    line_thickness: int = 2,
    opacity: float = 0.9,
    show_secondary: bool = True,
) -> np.ndarray:
    """Render authoritative GLIMS benchmark vector outlines dynamically with custom styling."""
    geom_e = shapely.wkt.loads(bdata["wkt_earlier"])
    geom_l = shapely.wkt.loads(bdata["wkt_later"])
    geom_ret = shapely.wkt.loads(bdata["wkt_retained"])
    geom_lost = shapely.wkt.loads(bdata["wkt_lost"])
    geom_gain = shapely.wkt.loads(bdata["wkt_gained"]) if bdata.get("wkt_gained") else None

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
    fig.patch.set_facecolor("#0B132B")
    ax.set_facecolor("#111E38")

    if show_secondary:
        _plot_shapely_geom(ax, geom_ret, facecolor="#00F2FE", edgecolor="none", alpha=opacity * 0.45)
        _plot_shapely_geom(ax, geom_lost, facecolor="#FF4B4B", edgecolor="#EF4444", alpha=opacity * 0.70)
        if geom_gain and not geom_gain.is_empty:
            _plot_shapely_geom(ax, geom_gain, facecolor="#10B981", edgecolor="#059669", alpha=opacity * 0.70)
        _plot_shapely_boundary(ax, geom_e, color="#F59E0B", linewidth=max(1.0, float(line_thickness) - 0.5), linestyle="--", alpha=opacity * 0.8)

    _plot_shapely_boundary(ax, geom_l, color=color_hex, linewidth=float(line_thickness), linestyle="-", alpha=float(opacity))

    minx, miny, maxx, maxy = geom_e.bounds
    l_minx, l_miny, l_maxx, l_maxy = geom_l.bounds
    minx, miny = min(minx, l_minx), min(miny, l_miny)
    maxx, maxy = max(maxx, l_maxx), max(maxy, l_maxy)

    pad_x = (maxx - minx) * 0.08
    pad_y = (maxy - miny) * 0.08
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")

    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{(val - cx)/1e3:+.1f} km"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{(val - cy)/1e3:+.1f} km"))
    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.grid(True, linestyle=":", color="#2A3D66", alpha=0.6)
    ax.set_xlabel("Relative Easting (km)", color="#94A3B8", fontsize=9)
    ax.set_ylabel("Relative Northing (km)", color="#94A3B8", fontsize=9)

    t1_yr = str(bdata.get("earlier_date", "1973"))[:4]
    t2_yr = str(bdata.get("later_date", "2015"))[:4]
    ax.set_title(
        f"{bdata['name']} ({bdata['id']})\n"
        f"GLIMS Outline ({t2_yr}) vs ({t1_yr}) | Net Loss: {bdata['net_loss_km2']:.3f} km² (-{bdata['percentage_change']}%)",
        color="#F8FAFC", fontsize=10, fontweight="bold", pad=8
    )

    handles = [
        plt.Line2D([0], [0], color=color_hex, lw=float(line_thickness), label=f"Active Margin ({t2_yr})"),
    ]
    if show_secondary:
        handles.extend([
            plt.Line2D([0], [0], color="#F59E0B", linestyle="--", lw=max(1.0, float(line_thickness) - 0.5), label=f"Baseline ({t1_yr})"),
            mpatches.Patch(color="#00F2FE", alpha=opacity * 0.45, label="Retained Core Ice"),
            mpatches.Patch(color="#FF4B4B", alpha=opacity * 0.70, label="Lost Glacier Area (Retreat)"),
        ])
    ax.legend(handles=handles, loc="lower left", framealpha=0.85, facecolor="#0B132B", edgecolor="#2A3D66", labelcolor="#F8FAFC", fontsize=7.5)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))


def parse_and_validate_vector_upload(
    file_bytes: bytes,
    filename: str,
    target_crs: str = "EPSG:32632",
    color_hex: str = "#12DFF3",
    line_thickness: int = 2,
    opacity: float = 0.9,
) -> Dict[str, Any]:
    """Parse, validate, and render uploaded vector boundaries (GeoJSON or Shapefile ZIP)."""
    fname = filename.lower()
    if fname.endswith(".zip"):
        # Validate ZIP contents for required ESRI shapefile components
        with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as z:
            names = [n.lower() for n in z.namelist()]
            has_shp = any(n.endswith(".shp") for n in names)
            has_shx = any(n.endswith(".shx") for n in names)
            has_dbf = any(n.endswith(".dbf") for n in names)
            has_prj = any(n.endswith(".prj") for n in names)

            if not has_prj:
                raise ValueError("Incomplete Shapefile: Missing .prj projection file. Coordinate Reference System (CRS) cannot be verified.")
            if not (has_shp and has_shx and has_dbf):
                raise ValueError("Incomplete Shapefile: A valid ESRI shapefile requires .shp, .shx, and .dbf files.")

        gdf = gpd.read_file(io.BytesIO(file_bytes))
    elif fname.endswith(".geojson") or fname.endswith(".json"):
        try:
            # Check valid JSON first
            data = json.loads(file_bytes.decode("utf-8"))
            if not isinstance(data, dict) or "features" not in data:
                raise ValueError("Invalid GeoJSON: File does not contain a FeatureCollection or 'features' array.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt GeoJSON file: {e}")
        gdf = gpd.read_file(io.BytesIO(file_bytes))
    else:
        raise ValueError(f"Unsupported vector format: '{filename}'. Expected .geojson, .json, or .zip shapefile archive.")

    if gdf is None or len(gdf) == 0:
        raise ValueError("Uploaded vector file contains zero geometry features.")

    if gdf.crs is None:
        raise ValueError("Uploaded vector geometry lacks Coordinate Reference System (CRS) metadata. Cannot project reliably.")

    orig_crs = str(gdf.crs)
    # Reproject to target projected CRS (default UTM 32N)
    gdf_utm = gdf.to_crs(target_crs)
    area_km2 = float(gdf_utm.geometry.area.sum() / 1e6)
    bounds = [float(b) for b in gdf_utm.total_bounds]

    # Render vector outline
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
    fig.patch.set_facecolor("#0B132B")
    ax.set_facecolor("#111E38")

    for geom in gdf_utm.geometry:
        _plot_shapely_geom(ax, geom, facecolor=color_hex, edgecolor=color_hex, alpha=opacity * 0.35)
        _plot_shapely_boundary(ax, geom, color=color_hex, linewidth=float(line_thickness), linestyle="-", alpha=float(opacity))

    minx, miny, maxx, maxy = bounds
    pad_x = max((maxx - minx) * 0.08, 100.0)
    pad_y = max((maxy - miny) * 0.08, 100.0)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")

    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{(val - cx)/1e3:+.1f} km"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{(val - cy)/1e3:+.1f} km"))
    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.grid(True, linestyle=":", color="#2A3D66", alpha=0.6)
    ax.set_xlabel("Relative Easting (km)", color="#94A3B8", fontsize=9)
    ax.set_ylabel("Relative Northing (km)", color="#94A3B8", fontsize=9)
    ax.set_title(f"Uploaded Vector Geometry: {filename}\nCRS: {orig_crs} -> {target_crs} | Total Area: {area_km2:.3f} km²",
                 color="#F8FAFC", fontsize=10, fontweight="bold", pad=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none", dpi=150)
    plt.close(fig)
    buf.seek(0)
    rendered_img = np.array(Image.open(buf))

    return {
        "valid": True,
        "feature_count": len(gdf),
        "original_crs": orig_crs,
        "projected_crs": target_crs,
        "area_km2": area_km2,
        "bounds": bounds,
        "rendered_array": rendered_img,
    }


def inspect_uploaded_geotiff(file_bytes: bytes) -> Dict[str, Any]:
    """Inspect and validate an uploaded satellite GeoTIFF file."""
    with rasterio.open(io.BytesIO(file_bytes)) as src:
        if src.crs is None:
            raise ValueError("Uploaded TIFF raster lacks georeferencing metadata (CRS is undefined). Please upload a valid GeoTIFF.")

        bounds = [float(b) for b in (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)]
        res = [float(r) for r in src.res]
        crs_str = str(src.crs)
        width, height = src.width, src.height
        band_count = src.count

        # Read sample array for quick preview
        if band_count >= 3:
            rgb_sample = src.read([1, 2, 3])
            # Normalize to 0-255 uint8
            rgb_disp = np.zeros((height, width, 3), dtype=np.uint8)
            for c in range(3):
                b = rgb_sample[c].astype(np.float32)
                p2, p98 = np.nanpercentile(b, [2, 98])
                if p98 > p2:
                    norm = np.clip((b - p2) / (p98 - p2) * 255.0, 0, 255)
                else:
                    norm = np.zeros_like(b)
                rgb_disp[:, :, c] = norm.astype(np.uint8)
        else:
            b1 = src.read(1).astype(np.float32)
            p2, p98 = np.nanpercentile(b1, [2, 98])
            if p98 > p2:
                norm = np.clip((b1 - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
            else:
                norm = np.zeros_like(b1, dtype=np.uint8)
            rgb_disp = np.stack([norm, norm, norm], axis=-1)

        return {
            "valid": True,
            "crs": crs_str,
            "bounds": bounds,
            "resolution": res,
            "width": width,
            "height": height,
            "band_count": band_count,
            "preview_rgb": rgb_disp,
        }


# Quick test
benchmarks = load_cached_glims_benchmarks()
print("Loaded benchmarks:", list(benchmarks.keys()))
arr = render_benchmark_glims_overlay(benchmarks["rhone"])
print("Rendered Rhone array:", arr.shape)
