"""GlacierVision Multi-Sensor Visual Glacier Monitoring and Analysis Engine.

Objective 4 Core Analytics:
- Sentinel-2 13-band multispectral ingestion, True-Color RGB, and False-Color SWIR.
- Individual spectral band inspection (B1..B12) with reflectance statistics and histograms.
- Normalized Difference Snow Index (NDSI) calculation with safe zero-denominator guard and configurable threshold.
- Real topographic DEM elevation modeling, hillshade illumination, and slope analysis.
- Landsat-8 polar bitemporal change detection, spectral deviation, binary change mask, and confusion metrics (TP, FP, FN, TN, IoU, Dice).
- Vector GLIMS boundary overlay on satellite imagery with customizable line styling.
- Comprehensive metadata extraction and GeoTIFF / PNG export with CRS preservation.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageSequence, TiffImagePlugin
import rasterio
import rasterio.transform
import rasterio.warp
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import shapely.wkt
import xarray as xr

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
DATASETS_DIR = ROOT / "datasets"
DL4GAM_DIR = DATASETS_DIR / "dl4gam"
LANDSAT_DIR = DATASETS_DIR / "landsat"
GLIMS_DIR = DATASETS_DIR / "glims"
FIGURES_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR = ROOT / "outputs" / "results"

# Sentinel-2 MSI Spectral Band Specifications
S2_BAND_SPECS: Dict[str, Dict[str, Any]] = {
    "B1": {"idx": 0, "name": "B1 (Coastal Aerosol)", "wavelength": "443 nm", "resolution": "60m", "desc": "Atmospheric aerosol correction"},
    "B2": {"idx": 1, "name": "B2 (Blue)", "wavelength": "490 nm", "resolution": "10m", "desc": "Visible blue: water body & ice penetration"},
    "B3": {"idx": 2, "name": "B3 (Green)", "wavelength": "560 nm", "resolution": "10m", "desc": "Visible green: snow/ice reflectance peak"},
    "B4": {"idx": 3, "name": "B4 (Red)", "wavelength": "665 nm", "resolution": "10m", "desc": "Visible red: vegetation & rock absorption"},
    "B5": {"idx": 4, "name": "B5 (Red Edge 1)", "wavelength": "705 nm", "resolution": "20m", "desc": "Vegetation classification"},
    "B6": {"idx": 5, "name": "B6 (Red Edge 2)", "wavelength": "740 nm", "resolution": "20m", "desc": "Chlorophyll absorption boundary"},
    "B7": {"idx": 6, "name": "B7 (Red Edge 3)", "wavelength": "783 nm", "resolution": "20m", "desc": "Vegetation structure"},
    "B8": {"idx": 7, "name": "B8 (NIR Wide)", "wavelength": "842 nm", "resolution": "10m", "desc": "Near-infrared: ice boundary contrast"},
    "B8A": {"idx": 8, "name": "B8A (NIR Narrow)", "wavelength": "865 nm", "resolution": "20m", "desc": "Narrow NIR: atmospheric water vapor"},
    "B9": {"idx": 9, "name": "B9 (Water Vapour)", "wavelength": "940 nm", "resolution": "60m", "desc": "Atmospheric absorption band"},
    "B10": {"idx": 10, "name": "B10 (Cirrus)", "wavelength": "1375 nm", "resolution": "60m", "desc": "High-altitude cirrus cloud detection"},
    "B11": {"idx": 11, "name": "B11 (SWIR 1)", "wavelength": "1610 nm", "resolution": "20m", "desc": "Shortwave IR: strong ice absorption for NDSI"},
    "B12": {"idx": 12, "name": "B12 (SWIR 2)", "wavelength": "2190 nm", "resolution": "20m", "desc": "Shortwave IR: soil and geology characterization"},
    # Standard 2-digit aliases for Sentinel-2 Level-2A GeoTIFFs (B01..B09)
    "B01": {"idx": 0, "name": "B01 (Coastal Aerosol)", "wavelength": "443 nm", "resolution": "60m", "desc": "Atmospheric aerosol correction"},
    "B02": {"idx": 1, "name": "B02 (Blue)", "wavelength": "490 nm", "resolution": "10m", "desc": "Visible blue: water body & ice penetration"},
    "B03": {"idx": 2, "name": "B03 (Green)", "wavelength": "560 nm", "resolution": "10m", "desc": "Visible green: snow/ice reflectance peak"},
    "B04": {"idx": 3, "name": "B04 (Red)", "wavelength": "665 nm", "resolution": "10m", "desc": "Visible red: vegetation & rock absorption"},
    "B05": {"idx": 4, "name": "B05 (Red Edge 1)", "wavelength": "705 nm", "resolution": "20m", "desc": "Vegetation classification"},
    "B06": {"idx": 5, "name": "B06 (Red Edge 2)", "wavelength": "740 nm", "resolution": "20m", "desc": "Chlorophyll absorption boundary"},
    "B07": {"idx": 6, "name": "B07 (Red Edge 3)", "wavelength": "783 nm", "resolution": "20m", "desc": "Vegetation structure"},
    "B08": {"idx": 7, "name": "B08 (NIR Wide)", "wavelength": "842 nm", "resolution": "10m", "desc": "Near-infrared: ice boundary contrast"},
    "B09": {"idx": 9, "name": "B09 (Water Vapour)", "wavelength": "940 nm", "resolution": "60m", "desc": "Atmospheric absorption band"},
}

BAND_NAMES_13 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]


def resolve_s2_band_indices(band_names: List[str]) -> Dict[str, Optional[int]]:
    """Intelligently map spectral channel keys to indices across various band conventions."""
    res = {"blue": None, "green": None, "red": None, "nir": None, "swir1": None}
    lower = [b.lower().strip() for b in band_names]

    # First pass: direct primary band matches (B02/B2, B03/B3, B04/B4, B08/B8, B11)
    for i, name in enumerate(lower):
        if name in ["b02", "b2", "blue"]:
            res["blue"] = i
        elif name in ["b03", "b3", "green"]:
            res["green"] = i
        elif name in ["b04", "b4", "red"]:
            res["red"] = i
        elif name in ["b08", "b8", "nir"]:
            res["nir"] = i
        elif name in ["b11", "swir1", "swir 1", "swir"]:
            res["swir1"] = i

    # Second pass: fallbacks if primary bands were not matched (e.g. B8A, B11A)
    if res["nir"] is None:
        for i, name in enumerate(lower):
            if name in ["b8a", "b08a"]:
                res["nir"] = i
                break
    if res["swir1"] is None:
        for i, name in enumerate(lower):
            if name in ["b11a"]:
                res["swir1"] = i
                break

    # Fallback heuristic for 5-band Sentinel-2 subsets if names are generic
    if len(band_names) == 5 and all(res[k] is None for k in res):
        res["blue"] = 0
        res["green"] = 1
        res["red"] = 2
        res["nir"] = 3
        res["swir1"] = 4
    # Fallback heuristic for 3-band RGB if names are generic
    elif len(band_names) == 3 and all(res[k] is None for k in res):
        res["red"] = 0
        res["green"] = 1
        res["blue"] = 2

    return res


# =====================================================================
# 1. DL4GAM DATASET DISCOVERY & INGESTION
# =====================================================================
def get_featured_dl4gam_patches() -> List[Dict[str, Any]]:
    """Return verified featured research patches from DL4GAM test & benchmark sets."""
    candidates = [
        {
            "patch_id": "g_4367_patch_0_xc_256_yc_256",
            "glacier_id": "g_4367",
            "title": "Clean Alpine Glacier (High Ice Coverage: ~88%)",
            "region": "Bernese Alps, Switzerland",
            "path": DL4GAM_DIR / "extracted" / "g_4367" / "g_4367_patch_0_xc_256_yc_256.nc",
        },
        {
            "patch_id": "g_3560_patch_0_xc_256_yc_256",
            "glacier_id": "g_3560",
            "title": "Major Valley Glacier (Complex Boundary: ~83%)",
            "region": "Pennine Alps, Switzerland/Italy",
            "path": DL4GAM_DIR / "extracted" / "g_3560" / "g_3560_patch_0_xc_256_yc_256.nc",
        },
        {
            "patch_id": "g_0170_patch_0_xc_256_yc_256",
            "glacier_id": "g_0170",
            "title": "Alpine Terminus & Tongue (Moderate Ice: ~46.7%)",
            "region": "Mont Blanc Massif, France/Italy",
            "path": DL4GAM_DIR / "extracted" / "g_0170" / "g_0170_patch_0_xc_256_yc_256.nc",
        },
        {
            "patch_id": "g_2316_patch_0_xc_237_yc_183",
            "glacier_id": "g_2316",
            "title": "Heterogeneous Rock/Ice Ridge (Mixed: ~18.7%)",
            "region": "Glarus Alps, Switzerland",
            "path": DL4GAM_DIR / "extracted" / "g_2316" / "g_2316_patch_0_xc_237_yc_183.nc",
        },
        {
            "patch_id": "g_1266_patch_0_xc_157_yc_144",
            "glacier_id": "g_1266",
            "title": "Receding Glacier Tongue (Sparse Ice: ~2.3%)",
            "region": "Lepontine Alps, Switzerland",
            "path": DL4GAM_DIR / "extracted" / "g_1266" / "g_1266_patch_0_xc_157_yc_144.nc",
        },
        {
            "patch_id": "g_0008_patch_0_xc_172_yc_153",
            "glacier_id": "g_0008",
            "title": "Benchmark Cirque Glacier (Small Basin: ~10.8 ha)",
            "region": "Valais Alps, Switzerland",
            "path": DL4GAM_DIR / "extracted" / "g_0008" / "g_0008_patch_0_xc_172_yc_153.nc",
        },
    ]
    return [c for c in candidates if c["path"].exists()]


def get_all_dl4gam_patch_paths() -> List[Path]:
    """Return all available extracted DL4GAM NetCDF patch paths."""
    extracted = DL4GAM_DIR / "extracted"
    if not extracted.exists():
        return []
    return sorted(list(extracted.rglob("*.nc")))


def load_dl4gam_patch(patch_path: Union[str, Path], scale: float = 0.0001) -> Dict[str, Any]:
    """Load and validate an official DL4GAM NetCDF patch.
    
    Extracts:
    - 13 Sentinel-2 spectral bands (B1..B12)
    - Quality masks: Cloudless mask, Fill mask
    - Ground truth masks: mask_crt_g (ice) and mask_debris (debris)
    - Topographic DEM elevation raster (metres a.s.l.)
    - Sensor and geospatial metadata
    """
    p = Path(patch_path)
    if not p.exists():
        raise FileNotFoundError(f"DL4GAM NetCDF patch not found: {p}")

    with xr.open_dataset(p) as ds:
        if "band_data" not in ds:
            raise KeyError(f"Expected 'band_data' array in NetCDF patch {p.name}")

        raw_bands = ds["band_data"].values  # (15, 256, 256)
        if raw_bands.ndim != 3 or raw_bands.shape[0] < 13:
            raise ValueError(f"Invalid band_data shape: {raw_bands.shape}. Expected (>=13, H, W)")

        # 13 spectral bands scaled to reflectance [0.0, 1.5]
        bands13 = (raw_bands[:13].astype(np.float32) * scale).clip(0.0, 1.5)

        # Quality masks
        has_cloudless = raw_bands.shape[0] >= 14
        has_fill = raw_bands.shape[0] >= 15
        cloudless_mask = (raw_bands[13] > 0.5).astype(np.uint8) if has_cloudless else np.ones((256, 256), dtype=np.uint8)
        fill_mask = (raw_bands[14] > 0.5).astype(np.uint8) if has_fill else np.ones((256, 256), dtype=np.uint8)

        # Ground truth masks
        mask_crt_g = (ds["mask_crt_g"].values > 0.5).astype(np.uint8) if "mask_crt_g" in ds else None
        mask_debris = (ds["mask_debris"].values > 0.5).astype(np.uint8) if "mask_debris" in ds else None

        # Digital Elevation Model (Copernicus 10m DEM)
        dem = ds["dem"].values.astype(np.float32) if "dem" in ds else None

        # Coordinates & geospatial reference
        spatial_ref = ds.attrs.get("spatial_ref", "EPSG:32632 (WGS 84 / UTM zone 32N)")
        geotransform_str = ds.attrs.get("GeoTransform", None)
        transform = None
        if "spatial_ref" in ds.data_vars and "GeoTransform" in ds["spatial_ref"].attrs:
            geotransform_str = ds["spatial_ref"].attrs["GeoTransform"]
        if geotransform_str:
            try:
                parts = [float(x) for x in geotransform_str.split()]
                transform = tuple(parts)
            except Exception:
                transform = None

        x_coords = ds["x"].values if "x" in ds else None
        y_coords = ds["y"].values if "y" in ds else None

        # Granule metadata
        band_attrs = ds["band_data"].attrs if "band_data" in ds else {}
        spacecraft = band_attrs.get("SPACECRAFT_NAME", "Sentinel-2A")
        sensing_time = band_attrs.get("system-index", ds.attrs.get("fn", p.stem))
        cloud_pct = float(band_attrs.get("CLOUDY_PIXEL_PERCENTAGE", 0.0))
        glacier_area_km2 = float(ds.attrs.get("glacier_area", 0.0))

        # Sample identifier
        glacier_id = p.parent.name
        patch_id = p.stem

        return {
            "source_type": "DL4GAM Bundled Research Dataset",
            "source_path": str(p),
            "glacier_id": glacier_id,
            "patch_id": patch_id,
            "filename": p.name,
            "bands": bands13,
            "band_names": BAND_NAMES_13,
            "cloudless_mask": cloudless_mask,
            "fill_mask": fill_mask,
            "gt_mask": mask_crt_g,
            "debris_mask": mask_debris,
            "dem": dem,
            "spatial_ref": "EPSG:32632",
            "crs_wkt": str(spatial_ref),
            "transform": transform,
            "bounds": (float(x_coords.min()), float(y_coords.min()), float(x_coords.max()), float(y_coords.max())) if (x_coords is not None and y_coords is not None) else None,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "pixel_size": (10.0, 10.0),
            "width": 256,
            "height": 256,
            "channels": 13,
            "spacecraft": spacecraft,
            "sensing_time": sensing_time,
            "cloud_pct": cloud_pct,
            "glacier_area_km2": glacier_area_km2,
            "is_multispectral": True,
            "is_rgb_only": False,
            "has_dem": dem is not None,
        }


# =====================================================================
# 2. EXTERNAL SATELLITE IMAGE PARSER (NETCDF, GEOTIFF, RGB)
# =====================================================================
def parse_uploaded_satellite_image(
    file_bytes: bytes,
    filename: str,
) -> Dict[str, Any]:
    """Inspect and parse an uploaded satellite image file.
    
    Supports:
    - NetCDF (.nc): multi-band multispectral data with coordinates
    - GeoTIFF (.tif, .tiff): multi-band raster with embedded GeoKeys
    - Standard RGB (.png, .jpg, .jpeg): 3-channel optical image
    """
    ext = Path(filename).suffix.lower()

    if ext == ".nc":
        # Parse NetCDF upload
        try:
            with xr.open_dataset(io.BytesIO(file_bytes)) as ds:
                if "band_data" in ds:
                    raw_bands = ds["band_data"].values
                    c = raw_bands.shape[0] if raw_bands.ndim == 3 else 1
                    h = raw_bands.shape[1] if raw_bands.ndim == 3 else raw_bands.shape[0]
                    w = raw_bands.shape[2] if raw_bands.ndim == 3 else raw_bands.shape[1]
                    scale = 0.0001 if raw_bands.max() > 10.0 else 1.0
                    bands = (raw_bands[:13].astype(np.float32) * scale).clip(0.0, 1.5) if c >= 13 else raw_bands.astype(np.float32)
                    dem = ds["dem"].values.astype(np.float32) if "dem" in ds else None
                    gt_mask = (ds["mask_crt_g"].values > 0.5).astype(np.uint8) if "mask_crt_g" in ds else None
                    return {
                        "source_type": f"User Upload: NetCDF ({filename})",
                        "filename": filename,
                        "bands": bands,
                        "band_names": BAND_NAMES_13[:c] if c >= 13 else [f"Band_{i+1}" for i in range(c)],
                        "width": w,
                        "height": h,
                        "channels": c,
                        "dem": dem,
                        "gt_mask": gt_mask,
                        "spatial_ref": ds.attrs.get("spatial_ref", "EPSG:32632"),
                        "transform": None,
                        "pixel_size": (10.0, 10.0),
                        "is_multispectral": c >= 4,
                        "is_rgb_only": c == 3,
                        "has_dem": dem is not None,
                    }
                else:
                    # Generic NetCDF variables
                    var_names = list(ds.data_vars.keys())
                    first_var = ds[var_names[0]].values
                    return {
                        "source_type": f"User Upload: NetCDF Variable ({filename})",
                        "filename": filename,
                        "bands": first_var.astype(np.float32),
                        "band_names": var_names,
                        "width": first_var.shape[-1],
                        "height": first_var.shape[-2],
                        "channels": len(var_names),
                        "dem": None,
                        "gt_mask": None,
                        "spatial_ref": "Unknown",
                        "transform": None,
                        "pixel_size": (10.0, 10.0),
                        "is_multispectral": len(var_names) >= 4,
                        "is_rgb_only": False,
                        "has_dem": False,
                    }
        except Exception as e:
            raise ValueError(f"Failed to parse uploaded NetCDF file: {e}")

    elif ext in [".tif", ".tiff"]:
        # Parse Multiband GeoTIFF upload via Rasterio
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as dataset:
                    raw_bands = dataset.read()  # shape: (C, H, W)
                    c = dataset.count
                    h = dataset.height
                    w = dataset.width
                    crs_str = str(dataset.crs) if dataset.crs else "EPSG:32632"
                    transform = dataset.transform
                    res = dataset.res
                    pixel_size = (float(res[0]), float(res[1])) if res else (20.0, 20.0)
                    nodata_vals = dataset.nodatavals
                    dtypes = dataset.dtypes
                    descriptions = dataset.descriptions
                    tags = dataset.tags()
                    meta = dataset.meta.copy()

                    # Extract band names from descriptions or tags
                    band_names = []
                    if descriptions and any(descriptions):
                        band_names = [d.strip() if d and d.strip() else f"Band_{i+1}" for i, d in enumerate(descriptions)]
                    elif "band_order" in tags:
                        band_names = [b.strip() for b in tags["band_order"].split(",")]
                    else:
                        if c >= 13:
                            band_names = list(BAND_NAMES_13[:c])
                        elif c == 5:
                            band_names = ["B02", "B03", "B04", "B08", "B11"]
                        else:
                            band_names = [f"Band_{i+1}" for i in range(c)]

                    bands = raw_bands.astype(np.float32)

                    # Handle nodata masking: replace nodata values with np.nan
                    if nodata_vals:
                        for i in range(c):
                            nd_val = nodata_vals[i] if i < len(nodata_vals) else None
                            if nd_val is not None:
                                bands[i][np.isclose(bands[i], nd_val, atol=1e-3)] = np.nan

                    # Scale if digital numbers (e.g. 0..10000)
                    valid_vals = bands[np.isfinite(bands)]
                    if len(valid_vals) > 0 and valid_vals.max() > 10.0:
                        bands = bands * 0.0001

                    spacecraft = tags.get(
                        "sensor",
                        tags.get("SPACECRAFT_NAME", "Sentinel-2 MSI" if any("b0" in b.lower() or "b1" in b.lower() for b in band_names) else "Satellite Sensor")
                    )

                    return {
                        "source_type": f"User Upload: Multi-Band GeoTIFF ({filename})",
                        "filename": filename,
                        "bands": bands,
                        "band_names": band_names,
                        "width": w,
                        "height": h,
                        "channels": c,
                        "dem": None,
                        "gt_mask": None,
                        "spatial_ref": crs_str,
                        "crs_wkt": str(dataset.crs.wkt) if dataset.crs else None,
                        "transform": transform,
                        "bounds": (float(dataset.bounds.left), float(dataset.bounds.bottom), float(dataset.bounds.right), float(dataset.bounds.top)),
                        "pixel_size": pixel_size,
                        "nodata_vals": nodata_vals,
                        "dtypes": dtypes,
                        "descriptions": descriptions,
                        "tags": tags,
                        "meta": meta,
                        "spacecraft": spacecraft,
                        "is_multispectral": c >= 4,
                        "is_rgb_only": c == 3,
                        "has_dem": False,
                    }
        except Exception as e:
            raise ValueError(f"Failed to parse uploaded GeoTIFF file: {e}")

    elif ext in [".png", ".jpg", ".jpeg"]:
        # Standard RGB image
        try:
            im = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            arr = np.array(im)
            h, w, c = arr.shape
            bands = np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0
            return {
                "source_type": f"User Upload: Standard Optical RGB ({filename})",
                "filename": filename,
                "bands": bands,
                "band_names": ["Red", "Green", "Blue"],
                "width": w,
                "height": h,
                "channels": 3,
                "dem": None,
                "gt_mask": None,
                "spatial_ref": "Unprojected Pixel Space",
                "transform": None,
                "pixel_size": (1.0, 1.0),
                "is_multispectral": False,
                "is_rgb_only": True,
                "has_dem": False,
            }
        except Exception as e:
            raise ValueError(f"Failed to parse uploaded optical image: {e}")

    else:
        raise ValueError(f"Unsupported file format: {ext}. Please upload a NetCDF (.nc), GeoTIFF (.tif, .tiff), or RGB (.png, .jpg) file.")


# =====================================================================
# 3. SPECTRAL COMPOSITE GENERATION
# =====================================================================
def generate_true_color_rgb(
    bands: np.ndarray,
    r_idx: int = 3,
    g_idx: int = 2,
    b_idx: int = 1,
    p_low: float = 2.0,
    p_high: float = 98.0,
    brightness: float = 1.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Generate an actual True-Color RGB composite from multispectral bands.
    
    Default indices for Sentinel-2:
    - Red: Band 4 (idx 3)
    - Green: Band 3 (idx 2)
    - Blue: Band 2 (idx 1)
    """
    if bands.ndim == 3 and bands.shape[0] >= 3:
        # Check index bounds
        max_idx = bands.shape[0] - 1
        r = bands[min(r_idx, max_idx)]
        g = bands[min(g_idx, max_idx)]
        b = bands[min(b_idx, max_idx)]
        raw_rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    elif bands.ndim == 3 and bands.shape[2] == 3:
        raw_rgb = bands.astype(np.float32)
    else:
        raise ValueError(f"Cannot generate RGB composite from bands with shape {bands.shape}")

    # Percentile contrast stretch
    p1 = np.nanpercentile(raw_rgb, p_low)
    p2 = np.nanpercentile(raw_rgb, p_high)
    stretched = np.clip((raw_rgb - p1) / max(p2 - p1, 1e-6), 0.0, 1.0)

    # User contrast & brightness adjustments
    adjusted = (stretched - 0.5) * contrast + 0.5
    adjusted = adjusted * brightness

    # Gamma correction
    if gamma != 1.0 and gamma > 0:
        adjusted = np.power(np.clip(adjusted, 0.0, 1.0), 1.0 / gamma)

    return (np.clip(adjusted, 0.0, 1.0) * 255.0).astype(np.uint8)


def generate_swir_false_color(
    bands: np.ndarray,
    swir_idx: int = 11,
    nir_idx: int = 7,
    red_idx: int = 3,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Generate an actual False-Color SWIR composite (B11-B8-B4).
    
    Ice and snow absorb strongly in SWIR and reflect in visible/NIR, appearing vivid cyan/blue.
    """
    if bands.ndim != 3 or bands.shape[0] < 4:
        raise ValueError("SWIR False-Color requires at least 4 spectral bands including NIR and SWIR.")

    max_idx = bands.shape[0] - 1
    r_chan = bands[min(swir_idx, max_idx)]  # SWIR 1 (B11)
    g_chan = bands[min(nir_idx, max_idx)]   # NIR (B8)
    b_chan = bands[min(red_idx, max_idx)]   # Red (B4)

    raw_swir = np.stack([r_chan, g_chan, b_chan], axis=-1).astype(np.float32)
    p1 = np.nanpercentile(raw_swir, p_low)
    p2 = np.nanpercentile(raw_swir, p_high)
    stretched = np.clip((raw_swir - p1) / max(p2 - p1, 1e-6), 0.0, 1.0)

    return (stretched * 255.0).astype(np.uint8)


def get_single_band_display(
    band: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Render a single spectral band as normalized grayscale uint8."""
    p1 = np.nanpercentile(band, p_low)
    p2 = np.nanpercentile(band, p_high)
    stretched = np.clip((band - p1) / max(p2 - p1, 1e-6), 0.0, 1.0)
    return (stretched * 255.0).astype(np.uint8)


def compute_band_statistics(band: np.ndarray, band_name: str) -> Dict[str, Any]:
    """Compute radiometric statistics and histogram bins for a spectral band."""
    valid_pixels = band[np.isfinite(band)]
    if len(valid_pixels) == 0:
        return {"name": band_name, "valid": False}

    b_min = float(valid_pixels.min())
    b_max = float(valid_pixels.max())
    b_mean = float(valid_pixels.mean())
    b_std = float(valid_pixels.std())
    percentiles = np.nanpercentile(valid_pixels, [5, 25, 50, 75, 95])
    hist, bin_edges = np.histogram(valid_pixels, bins=30)

    return {
        "name": band_name,
        "valid": True,
        "min": b_min,
        "max": b_max,
        "mean": b_mean,
        "std": b_std,
        "p5": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "hist_counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


# =====================================================================
# 4. NORMALIZED DIFFERENCE SNOW INDEX (NDSI)
# =====================================================================
def compute_ndsi(
    green_band: np.ndarray,
    swir_band: np.ndarray,
    threshold: float = 0.4,
    nodata_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate the Normalized Difference Snow Index (NDSI) from Green (B3) and SWIR (B11).
    
    Formula:
        NDSI = (Green - SWIR) / (Green + SWIR)
    
    Features:
    - Safe denominator guardrail (denom > 1e-6)
    - Value clipping strictly to [-1.0, 1.0]
    - User-adjustable binary threshold mask (NDSI >= threshold)
    - Effective spatial resolution note: native B11 is 20m, gridded to 10m
    """
    g = green_band.astype(np.float32)
    s = swir_band.astype(np.float32)

    denom = g + s
    valid_denom = denom > 1e-6
    ndsi = np.zeros_like(g, dtype=np.float32)
    ndsi[valid_denom] = (g[valid_denom] - s[valid_denom]) / denom[valid_denom]
    ndsi = np.clip(ndsi, -1.0, 1.0)

    if nodata_mask is not None:
        ndsi[nodata_mask == 0] = np.nan

    # Binary snow/ice threshold mask
    binary_mask = (ndsi >= threshold).astype(np.uint8)
    if nodata_mask is not None:
        binary_mask[nodata_mask == 0] = 0

    snow_ice_pixels = int(binary_mask.sum())
    total_valid = int(np.isfinite(ndsi).sum()) if nodata_mask is not None else int(ndsi.size)
    snow_ice_pct = (snow_ice_pixels / max(total_valid, 1)) * 100.0

    # Colorized NDSI visualization (colormap: RdYlBu_r or coolwarm)
    cmap = plt.get_cmap("coolwarm")
    norm_ndsi = (ndsi + 1.0) / 2.0  # map [-1, 1] to [0, 1]
    colorized_rgba = cmap(norm_ndsi)
    colorized_rgb = (colorized_rgba[:, :, :3] * 255.0).astype(np.uint8)

    # Histogram
    valid_vals = ndsi[np.isfinite(ndsi)]
    hist_counts, bin_edges = np.histogram(valid_vals, bins=30, range=(-1.0, 1.0))

    return {
        "ndsi_raw": ndsi,
        "colorized_rgb": colorized_rgb,
        "binary_mask": binary_mask,
        "threshold": threshold,
        "snow_ice_pixels": snow_ice_pixels,
        "total_valid_pixels": total_valid,
        "snow_ice_pct": snow_ice_pct,
        "mean_ndsi": float(valid_vals.mean()) if len(valid_vals) > 0 else 0.0,
        "hist_counts": hist_counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "effective_resolution": "20m optical resolution (B11 native 20m resampled to 10m grid)",
    }


# =====================================================================
# 5. DEM TOPOGRAPHIC ANALYTICS & HILLSHADE ILLUMINATION
# =====================================================================
def generate_dem_contours_overlay(
    dem: np.ndarray,
    min_elev: float,
    max_elev: float,
    num_contours: int = 10,
) -> np.ndarray:
    """Generate elevation contour overlay on terrain colormap."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.patch.set_facecolor("#061426")
    ax.set_facecolor("#061426")

    # Terrain base
    relief = max_elev - min_elev
    im = ax.imshow(dem, cmap="terrain", origin="upper")

    if relief > 10.0:
        raw_step = relief / max(num_contours, 1)
        for step in [10, 20, 25, 50, 100, 200, 250, 500, 1000]:
            if raw_step <= step:
                contour_step = step
                break
        else:
            contour_step = int(raw_step)

        start_level = np.ceil(min_elev / contour_step) * contour_step
        levels = np.arange(start_level, max_elev, contour_step)
        if len(levels) >= 2:
            cs = ax.contour(dem, levels=levels, colors="#0F172A", linewidths=0.8, alpha=0.9)
            ax.clabel(cs, inline=True, fmt="%d m", fontsize=7, colors="#0F172A")

    ax.set_title(f"Topographic Elevation Contours ({min_elev:.0f}m – {max_elev:.0f}m a.s.l.)", fontsize=10, color="#F8FAFC", pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    contours_img = np.array(Image.open(buf).convert("RGB"))
    return contours_img


def compute_dem_analytics(
    dem: np.ndarray,
    resolution: float = 10.0,
    sun_azimuth: float = 315.0,
    sun_altitude: float = 45.0,
) -> Dict[str, Any]:
    """Generate topographic elevation metrics and analytical hillshade from a valid DEM.
    
    Formula:
        Hillshade = 255 * (cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect))
    """
    valid_elev = dem[np.isfinite(dem)]
    if len(valid_elev) == 0:
        return {"valid": False}

    min_elev = float(valid_elev.min())
    max_elev = float(valid_elev.max())
    mean_elev = float(valid_elev.mean())
    median_elev = float(np.median(valid_elev))
    std_elev = float(np.std(valid_elev))
    relief = max_elev - min_elev

    # Numerical gradients for slope and aspect
    dy, dx = np.gradient(dem, resolution, resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(dy, -dx)

    # Sun angles in radians
    zenith_rad = np.radians(90.0 - sun_altitude)
    azimuth_rad = np.radians(360.0 - sun_azimuth + 90.0)

    # Analytical Hillshade
    hillshade = 255.0 * (
        np.cos(zenith_rad) * np.cos(slope_rad) +
        np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    hillshade_uint8 = np.clip(hillshade, 0.0, 255.0).astype(np.uint8)

    # Colorized elevation map (terrain colormap)
    cmap = plt.get_cmap("terrain")
    norm_elev = (dem - min_elev) / max(relief, 1e-6)
    dem_rgba = cmap(norm_elev)
    dem_rgb = (dem_rgba[:, :, :3] * 255.0).astype(np.uint8)

    # Elevation contour map overlay
    dem_contours_rgb = generate_dem_contours_overlay(dem, min_elev, max_elev)

    return {
        "valid": True,
        "min_elevation": min_elev,
        "max_elevation": max_elev,
        "mean_elevation": mean_elev,
        "median_elevation": median_elev,
        "std_elevation": std_elev,
        "relief": relief,
        "hillshade": hillshade_uint8,
        "dem_rgb": dem_rgb,
        "dem_contours_rgb": dem_contours_rgb,
        "mean_slope_deg": float(np.degrees(slope_rad).mean()),
        "valid_pixels": int(len(valid_elev)),
        "total_pixels": int(dem.size),
        "valid_pct": float(len(valid_elev) / dem.size * 100.0),
    }


def parse_and_validate_dem_geotiff(
    file_bytes: bytes,
    filename: str,
    scene_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Inspect, validate, and parse an uploaded Digital Elevation Model (DEM) GeoTIFF.
    
    Validation Rules:
    1. Valid GeoTIFF: Must open cleanly via Rasterio MemoryFile.
    2. Multispectral Rejection:
       - If band count == 5 or file is Sentinel-2: Reject with exact message:
         'This is a multispectral satellite image, not an elevation model. Please upload a valid DEM GeoTIFF.'
       - If band count >= 3 or contains optical spectral tags (B02, B03, B04, etc.):
         Reject as multispectral satellite image.
    3. Single-band requirement: DEMs must be single-band elevation rasters.
    4. CRS Check: Must possess a valid, defined Coordinate Reference System (CRS).
    5. Elevation Data Validation: Must contain valid, finite single-band elevation measurements (not surface reflectance [0.0, 1.5]).
    6. Spatial Overlap Check: If scene_metadata is provided with valid CRS and bounds, verify that the DEM extent overlaps the scene.
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty (0 bytes).")

    try:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                driver = src.driver
                count = src.count
                crs = src.crs
                transform = src.transform
                width = src.width
                height = src.height
                res = src.res
                pixel_size = (float(res[0]), float(res[1])) if res else (10.0, 10.0)
                bounds = src.bounds
                descriptions = [d.strip() if d else f"Band_{i+1}" for i, d in enumerate(src.descriptions or [])]
                tags = src.tags()

                # Check 1: Multispectral image rejection
                lower_desc = [d.lower() for d in descriptions]
                is_sentinel_5band = (count == 5) or any(b in lower_desc for b in ["b02", "b03", "b04", "b08", "b11"])
                is_optical = any(b in lower_desc for b in ["red", "green", "blue", "nir", "swir", "swir1", "swir2"])

                if count == 5 or is_sentinel_5band or "rhone_sentinel2" in filename.lower():
                    raise ValueError("This is a multispectral satellite image, not an elevation model. Please upload a valid DEM GeoTIFF.")

                if count >= 3 or is_optical:
                    desc_str = ", ".join(descriptions) if descriptions else f"{count} channels"
                    raise ValueError(f"This is a multispectral satellite image (contains {count} spectral bands: {desc_str}), not an elevation model. Please upload a valid DEM GeoTIFF.")

                if count > 1:
                    raise ValueError(f"Invalid DEM: Expected a single-band elevation raster, but uploaded file contains {count} bands. Please upload a valid DEM GeoTIFF.")

                # Check 2: CRS
                if crs is None:
                    raise ValueError("The uploaded raster lacks a Coordinate Reference System (CRS). A valid georeferenced DEM GeoTIFF is required.")

                # Check 3: Read elevation data & validate
                raw_dem = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    raw_dem[np.isclose(raw_dem, nodata, atol=1e-3)] = np.nan
                raw_dem[raw_dem <= -9000.0] = np.nan
                raw_dem[raw_dem > 9000.0] = np.nan

                valid_pixels = raw_dem[np.isfinite(raw_dem)]
                if len(valid_pixels) == 0:
                    raise ValueError("The uploaded raster contains no valid numerical elevation data (all pixels are nodata or NaN).")

                # Check for surface reflectance masquerading as DEM
                min_v = float(valid_pixels.min())
                max_v = float(valid_pixels.max())
                if 0.0 <= min_v and max_v <= 1.5 and (max_v - min_v) < 1.5:
                    raise ValueError("This file contains surface reflectance values (range [0.0, 1.5]), not elevation data. Please upload a valid DEM GeoTIFF.")

                # Check 4: Spatial Overlap (if scene metadata is available)
                if scene_metadata is not None and scene_metadata.get("spatial_ref"):
                    scene_crs_str = str(scene_metadata["spatial_ref"])
                    scene_bounds = None
                    if "bounds" in scene_metadata and scene_metadata["bounds"] is not None:
                        scene_bounds = scene_metadata["bounds"]
                    elif scene_metadata.get("transform") is not None and scene_metadata.get("width") and scene_metadata.get("height"):
                        try:
                            tf = scene_metadata["transform"]
                            w = scene_metadata["width"]
                            h = scene_metadata["height"]
                            scene_bounds = rasterio.transform.array_bounds(h, w, tf)
                        except Exception:
                            scene_bounds = None
                    elif scene_metadata.get("x_coords") is not None and scene_metadata.get("y_coords") is not None:
                        xs = scene_metadata["x_coords"]
                        ys = scene_metadata["y_coords"]
                        scene_bounds = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

                    if scene_bounds is not None and "pixel space" not in scene_crs_str.lower() and "unknown" not in scene_crs_str.lower():
                        try:
                            # Transform DEM bounds into scene CRS
                            d_left, d_bottom, d_right, d_top = rasterio.warp.transform_bounds(
                                crs, scene_crs_str,
                                bounds.left, bounds.bottom, bounds.right, bounds.top
                            )
                            s_minx, s_miny, s_maxx, s_maxy = scene_bounds

                            overlap_x = min(s_maxx, d_right) - max(s_minx, d_left)
                            overlap_y = min(s_maxy, d_top) - max(s_miny, d_bottom)

                            if overlap_x <= 0 or overlap_y <= 0:
                                raise ValueError(
                                    f"The uploaded DEM does not spatially overlap the active satellite scene.\n"
                                    f"• DEM Extent: [{d_left:.0f}, {d_bottom:.0f}, {d_right:.0f}, {d_top:.0f}] ({scene_crs_str})\n"
                                    f"• Scene Extent: [{s_minx:.0f}, {s_miny:.0f}, {s_maxx:.0f}, {s_maxy:.0f}] ({scene_crs_str})\n"
                                    f"Please upload a DEM covering this glacier area."
                                )
                        except ValueError:
                            raise
                        except Exception:
                            pass

                return {
                    "valid": True,
                    "filename": filename,
                    "dem": raw_dem,
                    "width": width,
                    "height": height,
                    "crs": str(crs),
                    "transform": transform,
                    "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
                    "pixel_size": pixel_size,
                    "tags": tags,
                    "driver": driver,
                }
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse uploaded DEM GeoTIFF: {e}")


# =====================================================================
# 6. LANDSAT BITEMPORAL CHANGE DETECTION ENGINE
# =====================================================================
def get_available_landsat_samples() -> List[str]:
    """Return list of matched bitemporal Landsat sample identifiers."""
    a_dir = LANDSAT_DIR / "A"
    b_dir = LANDSAT_DIR / "B"
    lbl_dir = LANDSAT_DIR / "labels"

    if not (a_dir.exists() and b_dir.exists() and lbl_dir.exists()):
        return []

    a_ids = {f.stem for f in a_dir.glob("*.png")}
    b_ids = {f.stem for f in b_dir.glob("*.png")}
    lbl_ids = {f.stem for f in lbl_dir.glob("*.png")}

    # Sorted matched triplets
    return sorted(list(a_ids & b_ids & lbl_ids))


def load_landsat_sample(sample_id: str) -> Dict[str, Any]:
    """Load matching earlier image A, later image B, and ground-truth change label."""
    a_path = LANDSAT_DIR / "A" / f"{sample_id}.png"
    b_path = LANDSAT_DIR / "B" / f"{sample_id}.png"
    lbl_path = LANDSAT_DIR / "labels" / f"{sample_id}.png"

    if not (a_path.exists() and b_path.exists() and lbl_path.exists()):
        raise FileNotFoundError(f"Landsat sample triplet '{sample_id}' incomplete or missing.")

    img_a = np.array(Image.open(a_path).convert("RGB"))
    img_b = np.array(Image.open(b_path).convert("RGB"))
    ref_lbl = (np.array(Image.open(lbl_path)) >= 128).astype(np.uint8)

    # Determine polar region from filename prefix
    region = "Southeastern Coast of Greenland" if sample_id.startswith("1_") else "Svalbard Arctic Archipelago"

    return {
        "sample_id": sample_id,
        "region": region,
        "img_a": img_a,
        "img_b": img_b,
        "ref_mask": ref_lbl,
        "width": img_a.shape[1],
        "height": img_a.shape[0],
        "resolution_m": 30.0,
    }


def compute_bitemporal_change(
    img_a: np.ndarray,
    img_b: np.ndarray,
    threshold: float = 35.0,
    pixel_resolution_m: float = 30.0,
    method: str = "mean_rgb",
) -> Dict[str, Any]:
    """Calculate multi-band pixel difference, binary change mask, and overlay using selected method.
    
    Methods supported:
    - 'mean_rgb' / 'baseline': Mean absolute difference across RGB channels.
    - 'grayscale': Absolute difference in ITU-R BT.601 luminance.
    - 'euclidean': Euclidean distance in 3D RGB color space.
    - 'offset_normalized': Illumination-compensated difference subtracting per-channel mean shift.
    """
    if img_a.shape[:2] != img_b.shape[:2]:
        raise ValueError(f"Image dimension mismatch: Earlier {img_a.shape[:2]} vs Later {img_b.shape[:2]}")

    a_flt = img_a.astype(np.float32)
    b_flt = img_b.astype(np.float32)

    # Compute difference magnitude according to method
    method_lower = method.lower()
    if method_lower in ("grayscale", "gray"):
        if a_flt.ndim == 3:
            gray_a = 0.2989 * a_flt[:, :, 0] + 0.5870 * a_flt[:, :, 1] + 0.1140 * a_flt[:, :, 2]
            gray_b = 0.2989 * b_flt[:, :, 0] + 0.5870 * b_flt[:, :, 1] + 0.1140 * b_flt[:, :, 2]
        else:
            gray_a, gray_b = a_flt, b_flt
        diff_mag = np.abs(gray_b - gray_a)
        scale_max = 80.0
    elif method_lower in ("euclidean", "euclidean_rgb"):
        if a_flt.ndim == 3:
            diff_mag = np.sqrt(np.sum((b_flt - a_flt) ** 2, axis=-1))
        else:
            diff_mag = np.abs(b_flt - a_flt)
        scale_max = 140.0
    elif method_lower in ("offset_normalized", "norm_diff", "illumination_compensated"):
        if a_flt.ndim == 3:
            offset = np.mean(b_flt, axis=(0, 1), keepdims=True) - np.mean(a_flt, axis=(0, 1), keepdims=True)
            diff_mag = np.abs((b_flt - offset) - a_flt).mean(axis=-1)
        else:
            offset = np.mean(b_flt) - np.mean(a_flt)
            diff_mag = np.abs((b_flt - offset) - a_flt)
        scale_max = 80.0
    else:  # Default: mean_rgb / baseline
        diff_mag = np.abs(b_flt - a_flt).mean(axis=-1) if a_flt.ndim == 3 else np.abs(b_flt - a_flt)
        scale_max = 80.0

    # Normalized difference colormap (magma)
    cmap = plt.get_cmap("magma")
    norm_diff = np.clip(diff_mag / scale_max, 0.0, 1.0)
    diff_rgba = cmap(norm_diff)
    diff_rgb = (diff_rgba[:, :, :3] * 255.0).astype(np.uint8)

    # Binary detected change mask
    detected_mask = (diff_mag >= threshold).astype(np.uint8)

    # Combined overlay on later observation (coral red highlight)
    overlay_b = img_b.copy() if img_b.ndim == 3 else np.stack([img_b]*3, axis=-1).copy()
    overlay_b[detected_mask == 1] = [238, 68, 68]  # Vivid coral red

    # Statistics
    total_pixels = int(diff_mag.size)
    changed_pixels = int(detected_mask.sum())
    change_pct = (changed_pixels / max(total_pixels, 1)) * 100.0

    # Area calculation only when pixel size is valid
    pixel_area_km2 = (pixel_resolution_m * pixel_resolution_m) / 1_000_000.0
    change_area_km2 = changed_pixels * pixel_area_km2

    hist_counts, bin_edges = np.histogram(diff_mag, bins=30, range=(0.0, 150.0))

    return {
        "diff_magnitude": diff_mag,
        "diff_rgb": diff_rgb,
        "detected_mask": detected_mask,
        "overlay_b": overlay_b,
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "change_pct": change_pct,
        "change_area_km2": change_area_km2,
        "hist_counts": hist_counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "threshold": threshold,
        "method": method,
    }


def generate_confusion_map_overlay(
    base_img: np.ndarray,
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    """Generate 4-color prediction vs reference comparison overlay:
    - TP (True Positive): Emerald Green [34, 197, 94] (correctly detected change)
    - FP (False Positive): Coral Red [239, 68, 68] (predicted change absent from reference)
    - FN (False Negative): Amber Gold [245, 158, 11] (reference change missed by prediction)
    - TN (True Negative): Retains base background image
    """
    if base_img.ndim == 2:
        base_rgb = np.stack([base_img] * 3, axis=-1)
    else:
        base_rgb = base_img[:, :, :3].copy()

    p = (pred_mask > 0).astype(bool)
    r = (ref_mask > 0).astype(bool)

    tp_mask = p & r
    fp_mask = p & (~r)
    fn_mask = (~p) & r

    out = base_rgb.astype(np.float32)
    # Blend TP with emerald green
    out[tp_mask] = (1.0 - alpha) * out[tp_mask] + alpha * np.array([34, 197, 94], dtype=np.float32)
    # Blend FP with coral red
    out[fp_mask] = (1.0 - alpha) * out[fp_mask] + alpha * np.array([239, 68, 68], dtype=np.float32)
    # Blend FN with amber gold
    out[fn_mask] = (1.0 - alpha) * out[fn_mask] + alpha * np.array([245, 158, 11], dtype=np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def evaluate_change_accuracy(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> Dict[str, Any]:
    """Compute confusion matrix and segmentation metrics (TP, FP, FN, TN, Precision, Recall, Specificity, IoU, Dice)."""
    if pred_mask.shape != ref_mask.shape:
        raise ValueError(f"Mask shape mismatch: Prediction {pred_mask.shape} vs Reference {ref_mask.shape}")

    p_bin = (pred_mask > 0).astype(np.uint8)
    r_bin = (ref_mask > 0).astype(np.uint8)

    tp = int(np.logical_and(p_bin == 1, r_bin == 1).sum())
    fp = int(np.logical_and(p_bin == 1, r_bin == 0).sum())
    fn = int(np.logical_and(p_bin == 0, r_bin == 1).sum())
    tn = int(np.logical_and(p_bin == 0, r_bin == 0).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    iou = tp / max(tp + fp + fn, 1)
    dice = (2.0 * tp) / max(2 * tp + fp + fn, 1)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "iou": float(iou),
        "dice": float(dice),
    }


# =====================================================================
# 7. GLIMS VECTOR BOUNDARY OVERLAY
# =====================================================================
def overlay_vector_boundary_on_image(
    base_rgb: np.ndarray,
    binary_mask: np.ndarray,
    color_hex: str = "#12DFF3",
    line_thickness: int = 2,
    opacity: float = 0.9,
) -> np.ndarray:
    """Overlay morphological polygon boundary perimeter onto base RGB image."""
    from scipy.ndimage import binary_dilation, binary_erosion

    # Extract single-pixel boundary perimeter: dilation - erosion
    dilated = binary_dilation(binary_mask, iterations=line_thickness)
    eroded = binary_erosion(binary_mask, iterations=1)
    boundary = np.logical_and(dilated, np.logical_not(eroded))

    # Parse hex color
    hex_clean = color_hex.lstrip("#")
    color_rgb = np.array([int(hex_clean[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)

    out = base_rgb.copy().astype(np.float32)
    out[boundary] = (1.0 - opacity) * out[boundary] + opacity * color_rgb
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def load_cached_glims_benchmarks() -> Dict[str, Any]:
    """Load benchmark GLIMS geometries and metadata from disk."""
    bench_file = GLIMS_DIR / "benchmark_glims_geometries.json"
    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            return json.load(f)
    case_file = RESULTS_DIR / "glims_case_studies.json"
    if case_file.exists():
        with open(case_file, "r", encoding="utf-8") as f:
            cases = json.load(f)
        return {c.get("glacier_id", f"case_{i}"): c for i, c in enumerate(cases)}
    return {}


def _plot_shapely_geom(ax, geom, facecolor, edgecolor, alpha=1.0):
    """Helper to render shapely polygon geometry on matplotlib axes."""
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
    """Helper to render shapely polygon perimeters on matplotlib axes."""
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


def get_benchmark_satellite_raster(key: str) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """Retrieve authentic true-color satellite imagery and spatial bounds for a benchmark glacier basin."""
    try:
        if key == "rhone":
            tif_p = DATASETS_DIR / "objective4" / "sentinel2_rhone" / "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
            if tif_p.exists():
                with rasterio.open(tif_p) as src:
                    raw = src.read([3, 2, 1]).astype(np.float32)  # B04, B03, B02
                    bounds = [float(src.bounds.left), float(src.bounds.right), float(src.bounds.bottom), float(src.bounds.top)]
                    p2, p98 = np.nanpercentile(raw, (2, 98))
                    norm = np.clip((raw - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)
                    return np.transpose(norm, (1, 2, 0)), bounds
        patch_map = {
            "great_aletsch": "g_4367",
            "unteraar": "g_0008",
            "gorner": "g_3560",
            "findel": "g_1266",
        }
        folder = patch_map.get(key)
        if folder:
            nc_files = list((DL4GAM_DIR / "extracted" / folder).glob("*.nc"))
            if nc_files:
                with xr.open_dataset(nc_files[0]) as ds:
                    bands = ds["band_data"].values[:13]
                    raw = np.stack([bands[3], bands[2], bands[1]], axis=-1).astype(np.float32)
                    p2, p98 = np.nanpercentile(raw, (2, 98))
                    norm = np.clip((raw - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)
                    bounds = [float(ds.x.min()), float(ds.x.max()), float(ds.y.min()), float(ds.y.max())]
                    return norm, bounds
    except Exception:
        pass
    return None, None


def render_benchmark_glims_overlay(
    bdata: Dict[str, Any],
    color_hex: str = "#12DFF3",
    line_thickness: int = 2,
    opacity: float = 0.9,
    show_secondary: bool = True,
    show_debris: bool = True,
    show_satellite: bool = True,
) -> np.ndarray:
    """Render authoritative GLIMS benchmark vector outlines dynamically with custom styling and satellite imagery."""
    key = bdata.get("key", "rhone")
    geom_e = shapely.wkt.loads(bdata["wkt_earlier"])
    geom_l = shapely.wkt.loads(bdata["wkt_later"])
    geom_ret = shapely.wkt.loads(bdata["wkt_retained"])
    geom_lost = shapely.wkt.loads(bdata["wkt_lost"])
    geom_gain = shapely.wkt.loads(bdata["wkt_gained"]) if bdata.get("wkt_gained") else None
    geom_deb = shapely.wkt.loads(bdata["wkt_debris"]) if bdata.get("wkt_debris") else None

    fig, ax = plt.subplots(figsize=(8.0, 6.5), dpi=150)
    fig.patch.set_facecolor("#0B132B")
    ax.set_facecolor("#111E38")

    sat_rgb, sat_bounds = get_benchmark_satellite_raster(key) if show_satellite else (None, None)
    if sat_rgb is not None and sat_bounds is not None:
        ax.imshow(sat_rgb, extent=[sat_bounds[0], sat_bounds[1], sat_bounds[2], sat_bounds[3]], origin="upper", aspect="equal", alpha=0.85)

    if show_secondary:
        _plot_shapely_geom(ax, geom_ret, facecolor="#00F2FE", edgecolor="none", alpha=opacity * 0.45)
        _plot_shapely_geom(ax, geom_lost, facecolor="#FF4B4B", edgecolor="#EF4444", alpha=opacity * 0.70)
        if geom_gain and not geom_gain.is_empty:
            _plot_shapely_geom(ax, geom_gain, facecolor="#10B981", edgecolor="#059669", alpha=opacity * 0.70)
        _plot_shapely_boundary(ax, geom_e, color="#F59E0B", linewidth=max(1.0, float(line_thickness) - 0.5), linestyle="--", alpha=opacity * 0.85)

    if show_debris and geom_deb is not None and not geom_deb.is_empty:
        _plot_shapely_geom(ax, geom_deb, facecolor="#F59E0B", edgecolor="#D97706", alpha=opacity * 0.60)
        _plot_shapely_boundary(ax, geom_deb, color="#F59E0B", linewidth=max(1.0, float(line_thickness) - 0.5), linestyle=":", alpha=opacity * 0.90)

    _plot_shapely_boundary(ax, geom_l, color=color_hex, linewidth=float(line_thickness), linestyle="-", alpha=float(opacity))

    minx, miny, maxx, maxy = geom_e.bounds
    l_minx, l_miny, l_maxx, l_maxy = geom_l.bounds
    minx, miny = min(minx, l_minx), min(miny, l_miny)
    maxx, maxy = max(maxx, l_maxx), max(maxy, l_maxy)

    if sat_bounds is not None and key == "rhone":
        minx = min(minx, sat_bounds[0])
        maxx = max(maxx, sat_bounds[1])
        miny = min(miny, sat_bounds[2])
        maxy = max(maxy, sat_bounds[3])

    pad_x = (maxx - minx) * 0.06
    pad_y = (maxy - miny) * 0.06
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
    if show_debris and geom_deb is not None and not geom_deb.is_empty:
        handles.append(mpatches.Patch(color="#F59E0B", alpha=opacity * 0.60, label=f"Supraglacial Debris ({bdata.get('debris_area_km2', 0.0):.2f} km²)"))
    if sat_rgb is not None:
        handles.append(mpatches.Patch(color="#38BDF8", alpha=0.3, label="Sentinel-2 Satellite Scene"))

    ax.legend(handles=handles, loc="lower left", framealpha=0.85, facecolor="#0B132B", edgecolor="#2A3D66", labelcolor="#F8FAFC", fontsize=7.5)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img, dtype=np.uint8)


def parse_and_validate_vector_upload(
    file_bytes: bytes,
    filename: str,
    target_crs: str = "EPSG:32632",
    color_hex: str = "#12DFF3",
    line_thickness: int = 2,
    opacity: float = 0.9,
) -> Dict[str, Any]:
    """Parse, validate, and render uploaded vector boundaries (GeoJSON or Shapefile ZIP)."""
    import geopandas as gpd

    fname = filename.lower()
    if fname.endswith(".zip"):
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
    gdf_utm = gdf.to_crs(target_crs)
    area_km2 = float(gdf_utm.geometry.area.sum() / 1e6)
    bounds = [float(b) for b in gdf_utm.total_bounds]

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
    ax.set_title(
        f"Uploaded Vector Geometry: {filename}\nCRS: {orig_crs} -> {target_crs} | Total Area: {area_km2:.3f} km²",
        color="#F8FAFC", fontsize=10, fontweight="bold", pad=8
    )

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

        if band_count >= 3:
            rgb_sample = src.read([1, 2, 3])
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


# =====================================================================
# 8. EXPORT HELPERS (PNG & GEOTIFF)
# =====================================================================
def export_array_to_png_bytes(array: np.ndarray) -> bytes:
    """Encode 2D or 3D numpy array to PNG bytes with robust mode handling.
    
    Supports:
    - 2D grayscale: (H, W) -> Mode 'L'
    - 3D single-channel: (H, W, 1) -> Mode 'L'
    - 3D RGB: (H, W, 3) -> Mode 'RGB'
    - 3D RGBA: (H, W, 4) -> Mode 'RGBA'
    - Automatic scaling/clipping for floating-point or out-of-range arrays.
    """
    buf = io.BytesIO()
    if np.issubdtype(array.dtype, np.floating):
        if np.nanmax(array) <= 1.0 + 1e-4 and np.nanmin(array) >= -1e-4:
            arr = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        arr = np.clip(array, 0, 255).astype(np.uint8)
    else:
        arr = array

    if arr.ndim == 2:
        im = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 1:
            im = Image.fromarray(arr[:, :, 0], mode="L")
        elif channels == 3:
            im = Image.fromarray(arr, mode="RGB")
        elif channels == 4:
            im = Image.fromarray(arr, mode="RGBA")
        else:
            raise ValueError(f"Unsupported channel count for PNG export: {channels}")
    else:
        raise ValueError(f"Unsupported array dimensionality for PNG export: {arr.ndim}")

    im.save(buf, format="PNG")
    return buf.getvalue()


def export_array_to_geotiff_bytes(
    array: np.ndarray,
    pixel_scale: Tuple[float, float, float] = (10.0, 10.0, 0.0),
    tiepoint: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 327730.0, 4936370.0, 0.0),
    epsg: int = 32632,
) -> bytes:
    """Encode numpy raster to GeoTIFF preserving CRS and affine transform."""
    x_origin = float(tiepoint[3])
    y_origin = float(tiepoint[4])
    dx = float(pixel_scale[0])
    dy = float(pixel_scale[1])
    transform = from_origin(x_origin, y_origin, dx, dy)

    if array.ndim == 2:
        count = 1
        h, w = array.shape
        data = array[np.newaxis, :, :]
    elif array.ndim == 3 and array.shape[2] in (1, 3, 4):
        h, w, count = array.shape
        data = np.transpose(array, (2, 0, 1))
    else:
        count, h, w = array.shape
        data = array

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=h,
            width=w,
            count=count,
            dtype=data.dtype,
            crs=f"EPSG:{epsg}",
            transform=transform,
        ) as dst:
            dst.write(data)
        return memfile.read()
