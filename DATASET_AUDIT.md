# Comprehensive Dataset Audit & Inventory

**Project:** Few-Shot Computer Vision for Automated Glacier Detection and Retreat Monitoring  
**Audit Date:** November 2026  
**Auditor:** Senior ML Engineer & Research Software Architect  
**Project Path:** `C:\Users\Varun's LOQ\Desktop\GlacierResearch`

---

## 1. Executive Summary

A comprehensive read-only audit was performed on all four candidate datasets stored in the local project environment. Dataset dimensions, spectral variables, binary targets, temporal metadata, quality flags, and official split partitions were verified directly from disk files.

---

## 2. Dataset Inventory

| Dataset | Storage Location | Files Verified | Primary Format | Spatial Extent | Verified Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DL4GAM Alps** | `datasets/dl4gam/extracted/` | **2,251** NetCDF patches in **1,593** glacier folders | NetCDF-4 (`.nc`), $256\times 256$, 10m GSD | European Alps | Objective 1 (Few-Shot Glacier Segmentation) & Objective 5 (CLIP/CoOp Prompts) |
| **Landsat Bitemporal** | `datasets/landsat/` | **683** image pairs in `A/` and `B/`, **683** binary masks in `labels/` | PNG, $256\times 256$, 30m GSD | Greenland & Svalbard Archipelago | Objective 2 (Retreat Detection), Objective 3 (Area Quantification), Objective 4 (Visual Overlays) |
| **GLIMS Database** | `datasets/glims/extracted/` | **616,629** outlines in `glims_polygons.shp` (1.63 GB) | ESRI Shapefile, EPSG:4326 | Global Ice Outlines | Historical multi-temporal retreat candidate validation |
| **NASA EIS** | `datasets/eis/` | **11,119** remote records indexed in `eis_inventory.csv` | Catalog CSV & XLSX Manifest | Extreme Ice Survey FTP Archive | Remote inventory cataloged; 0 local images downloaded |

---

## 3. DL4GAM NetCDF Detailed Specification

- **File Path Sample:** `datasets/dl4gam/extracted/g_0008/g_0008_patch_0_xc_172_yc_153.nc`
- **Tensor Dimensions:** `(band: 15, y: 256, x: 256)`
- **Native Spatial Resolution:** 10m per pixel ($2,560\text{ m} \times 2,560\text{ m}$ footprint per patch).
- **Coordinate Reference System:** UTM projected coordinate systems (e.g., WGS 84 / UTM Zone 32N, EPSG:32632).

### Variables and Channels Breakdown:

1. **`band_data` (15 Channels, float32, scale factor `0.0001`):**
   - Channels 0–12: 13 Sentinel-2 spectral bands:
     - $B_1$ (Coastal Aerosol, 443nm)
     - $B_2$ (Blue, 490nm), $B_3$ (Green, 560nm), $B_4$ (Red, 665nm)
     - $B_5, B_6, B_7$ (Red Edge 1/2/3: 705nm, 740nm, 783nm)
     - $B_8$ (NIR Broad, 842nm), $B_{8A}$ (NIR Narrow, 865nm)
     - $B_9$ (Water Vapour, 945nm), $B_{10}$ (SWIR Cirrus, 1375nm)
     - $B_{11}$ (SWIR 1, 1610nm), $B_{12}$ (SWIR 2, 2190nm)
   - Channels 13–14: Sentinel-2 QA flags:
     - $B_{13} = \text{CLOUDLESS\_MASK}$ (1.0 = cloud-free)
     - $B_{14} = \text{FILL\_MASK}$ (1.0 = valid instrument data)
2. **Segmentation Ground Truth Targets:**
   - `mask_crt_g`: Target glacier binary mask ($1.0$ = glacier outline, $0.0$ = background).
   - `mask_all_g_id`: Multi-glacier integer IDs occurring in patch footprint (NaN for non-glacier pixels).
   - `mask_debris`: Binary mask for debris-covered ice cover.
   - Buffer masks: `mask_crt_g_b-20`, `mask_crt_g_b-10`, `mask_crt_g_b10`, `mask_crt_g_b20`, `mask_crt_g_b50`.
3. **Geophysical Context Variables:**
   - `dem` (Copernicus DEM elevation in metres), `dhdt` (elevation change rate), `v` (ITS_LIVE ice flow velocity), `slope`, `aspect`, `planform_curvature`, `profile_curvature`, `terrain_ruggedness_index`.

---

## 4. Official Partitioning & Normalization Statistics

- **Split Metadata File:** `datasets/dl4gam/data/map_all_splits_all_folds.csv`
- **Partitioning Principle:** Glacier-level separation (zero spatial data leakage).
- **Split 1 Distribution:**
  - Training: 986 glacier IDs, 1,475 patches (65.5% of patches)
  - Validation: 181 glacier IDs, 242 patches (10.8% of patches)
  - Testing: 426 glacier IDs, 534 patches (23.7% of patches)
  - Total Patches: Exactly 2,251 patches (0 unmapped patches).

### Verified Training Fold (`split_1`) Normalization Statistics:
Calculated across all 1,475 training patches (saved in `outputs/normalization_split_1.json`):
- **Scale Factor:** $0.0001$
- **Per-Band Means ($B_1 \dots B_{12}$):**
  `[0.2608, 0.2561, 0.2519, 0.2590, 0.2598, 0.2640, 0.2620, 0.2442, 0.2526, 0.1028, 0.0017, 0.1192, 0.0967]`
- **Per-Band Standard Deviations ($B_1 \dots B_{12}$):**
  `[0.1833, 0.2040, 0.2049, 0.2183, 0.2129, 0.2047, 0.1946, 0.1825, 0.1816, 0.0745, 0.0037, 0.1148, 0.0940]`
- **Target Positive Pixel Ratio:** $23.8969\%$
- **Calculated Class Imbalance Weight (`pos_weight`):** $3.18$

---

## 5. Quality Mask and Guardrail Decision

- Quality masks $B_{13}$ (`CLOUDLESS_MASK`) and $B_{14}$ (`FILL_MASK`) verify that patches are substantially cloud-free (100% cloudless flag across tested Alps patches) and 100% valid instrument fill.
- The pipeline slices `band_data[:13]`, applies the standard $0.0001$ reflectance scale, clamps reflectance values to the physically valid range $[0.0, 1.5]$, and standardizes using the training statistics above.
- Target mask `mask_crt_g` is strictly thresholded at $0.5$ into binary float32 values $\{0.0, 1.0\}$.
