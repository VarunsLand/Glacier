# Findelgletscher Area Discrepancy Investigation & Methodology Unification Audit Report

**Audit Date:** 2026-11-15  
**Glacier:** Findel Glacier / Findelgletscher (`G007880E45990N`)  
**Component:** Objective 2 — Geographical Glacier Retreat Explorer (`src/ui/retreat_explorer.py`, `src/inference.py`, `outputs/results/glims_case_studies.json`)  
**Status:** ✅ RESOLVED & FULLY VERIFIED (53/53 Tests Passing)

---

## 1. Executive Summary & Source of Discrepancy

During inspection of Findelgletscher (`G007880E45990N`), a small area difference was identified:
- **KPI Cards & Waterfall Chart:** Earlier: `18.614 km²`, Modern: `16.082 km²`, Net Loss: `2.532 km²`
- **Individual Map Legend:** Earlier: `18.603 km²`, Modern: `16.072 km²`, Net Loss: `2.531 km²`

### Scientific Investigation & Root Cause:
Through independent extraction from the authoritative GLIMS vector database (`datasets/glims/extracted/glims_download_52211/glims_polygons.shp`), the source of each number was determined:

1. **KPI & Chart Origin (`18.614 km²`, `16.082 km²`, `2.532 km²`):**
   - Sourced from the **GLIMS database authalic attribute (`db_area`)**, representing ellipsoidal/equal-area surface extent.
   - SGI 1973 outline (Analysis ID: `342320`): `db_area = 18.6141 km²`
   - SGI 2015 outline (Analysis IDs: `1096562`, `1096563`, `1096564`): $\sum \text{db\_area} = 2.0656 + 13.4980 + 0.5183 = 16.0819\text{ km}^2$
   - Difference: $18.6141 - 16.0819 = 2.5322\text{ km}^2$
   - Direct projection to **EPSG:3035** (ETRS89-LAEA Europe, Equal-Area) yields virtually identical numbers:
     - Earlier: `18.614071 km²` (rounds to `18.614 km²`)
     - Modern: `16.081937 km²` (rounds to `16.082 km²`)
     - Net Loss: `2.532135 km²` (rounds to `2.532 km²`)

2. **Map Legend Origin (`18.603 km²`, `16.072 km²`, `2.531 km²`):**
   - Sourced from **planar polygon geometry projected into UTM Zone 32N (`EPSG:32632`)**.
   - UTM Zone 32N is a conformal Transverse Mercator projection with central scale factor $k_0 = 0.9996$.
   - Conformal planar projections introduce a slight scale contraction on area:
     $$k^2 \approx (0.9996)^2 = 0.99920016 \quad (-0.0612\% \text{ area scale offset})$$
   - Projected geometry calculations in `EPSG:32632`:
     - Earlier: `18.602687 km²` (rounds to `18.603 km²`)
     - Modern: `16.072076 km²` (rounds to `16.072 km²`)
     - Net Loss: `2.530611 km²` (rounds to `2.531 km²`)

3. **Why Benchmarks Were Unified While Findel Differed:**
   - The four core benchmark glaciers (**Rhone**, **Great Aletsch**, **Unteraar**, **Gorner**) had dedicated case-study records in `outputs/results/glims_case_studies.json` storing their planar projected `EPSG:32632` geometry values. Consequently, their KPI cards, Waterfall charts, and map legends were all 100% unified in `EPSG:32632`.
   - Findel was previously queried from `glims_retreat_assessment_validated.csv` (which stored `db_area`), creating the slight discrepancy between the KPI cards and the UTM-projected map graphic.

---

## 2. Unification Methodology

To establish a single consistent methodology across all visual components while preserving secondary metrics:

1. **Primary Operational System (EPSG:32632):**
   - Added Findel Glacier (`G007880E45990N`) to `outputs/results/glims_case_studies.json` using its authoritative `EPSG:32632` projected geometry values:
     - **Earlier Area ($A_{t_1}$):** `18.603 km²`
     - **Modern Area ($A_{t_2}$):** `16.072 km²`
     - **Retained Core Ice:** `15.849 km²`
     - **Lost Glacier Area (Retreat):** `2.753 km²`
     - **Gained Glacier Area (Advance):** `0.223 km²`
     - **Net Surface Shrinkage:** `2.531 km²`
     - **Decadal Retreat:** `-13.60%`
   - Now Findel's KPI metric cards, Waterfall budget chart, spatial boundary decomposition list, and map overlay legend are **100% unified**.

2. **Secondary Source Preservation:**
   - In the **Spatial Projection Sensitivity Analysis expander**, Findel is formally cataloged alongside Rhone, Great Aletsch, Unteraar, and Gorner in the Multi-CRS Empirical Comparison Table, presenting the `EPSG:3035` / authalic `db_area` values (`18.614 km²`, `16.082 km²`, `2.532 km²`) alongside `EPSG:32632` and `EPSG:2056`.

---

## 3. Geometric Conservation Proofs (Unrounded Values)

Using double-precision geometry calculations on Findel outlines, all spatial conservation identities hold exactly:

### In EPSG:32632 (Primary Conformal System):
- Earlier Area ($A_{t_1}$): `18.602687 km²`
- Modern Area ($A_{t_2}$): `16.072076 km²`
- Retained Core Ice: `15.849343 km²`
- Lost Ice (Retreat): `2.753344 km²`
- Gained Ice (Advance): `0.222733 km²`
- Net Loss ($\Delta A$): `2.530611 km²`

**Identity Proofs:**
- **Identity 1:** $\text{Retained} + \text{Lost} = 15.849343 + 2.753344 = 18.602687\text{ km}^2 \equiv A_{t_1}$ (Residual: $3.20 \times 10^{-14}\text{ km}^2$)
- **Identity 2:** $\text{Retained} + \text{Gained} = 15.849343 + 0.222733 = 16.072076\text{ km}^2 \equiv A_{t_2}$ (Residual: $8.88 \times 10^{-14}\text{ km}^2$)
- **Identity 3:** $\text{Lost} - \text{Gained} = 2.753344 - 0.222733 = 2.530611\text{ km}^2 \equiv \text{Net Loss}$ (Residual: $1.19 \times 10^{-13}\text{ km}^2$)
- **Identity 4:** $A_{t_1} - A_{t_2} = 18.602687 - 16.072076 = 2.530611\text{ km}^2 \equiv \text{Net Loss}$ (Residual: $0.00$)

### In EPSG:3035 (Secondary Equal-Area System):
- Earlier Area ($A_{t_1}$): `18.614071 km²`
- Modern Area ($A_{t_2}$): `16.081937 km²`
- Retained Core Ice: `15.859066 km²`
- Lost Ice (Retreat): `2.755006 km²`
- Gained Ice (Advance): `0.222871 km²`
- Net Loss ($\Delta A$): `2.532135 km²`

**Identity Proofs:**
- $\text{Retained} + \text{Lost} = 15.859066 + 2.755006 = 18.614071\text{ km}^2 \equiv A_{t_1}$ (Residual: $5.33 \times 10^{-14}\text{ km}^2$)
- $\text{Retained} + \text{Gained} = 15.859066 + 0.222871 = 16.081937\text{ km}^2 \equiv A_{t_2}$ (Residual: $6.75 \times 10^{-14}\text{ km}^2$)
- $\text{Lost} - \text{Gained} = 2.755006 - 0.222871 = 2.532135\text{ km}^2 \equiv \text{Net Loss}$ (Residual: $1.22 \times 10^{-13}\text{ km}^2$)

---

## 4. Multi-CRS Comparison Table (Findel & Benchmarks)

| Glacier | Metric | EPSG:3035 (LAEA Equal-Area) | EPSG:32632 (UTM Zone 32N) | EPSG:2056 (Swiss LV95) | GLIMS Authalic `db_area` | Quantified UTM Distortion |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Findel** | Earlier Area ($A_{t_1}$) | `18.6141 km²` | `18.6027 km²` | `18.6195 km²` | `18.6141 km²` | -0.0612% (-0.0114 km²) |
| **Findel** | Modern Area ($A_{t_2}$) | `16.0819 km²` | `16.0721 km²` | `16.0866 km²` | `16.0819 km²` | -0.0612% (-0.0099 km²) |
| **Findel** | Net Area Loss ($\Delta A$) | `2.5321 km²` | `2.5306 km²` | `2.5329 km²` | `2.5322 km²` | -0.0602% (-0.0015 km²) |
| **Findel** | **Decadal Area Change** | **-13.6033%** | **-13.6035%** | **-13.6033%** | **-13.6033%** | **Invariant ($\Delta_{\text{pct}} = 0.0002\%$)** |
| **Rhone** | Earlier Area / Modern Area | `17.1784 / 14.1895` | `17.1657 / 14.1789` | `17.1791 / 14.1900` | `17.1784 / 14.1895` | -0.0743% / -0.0747% |
| **Rhone** | Net Loss / Decadal Change | `2.9890 km² / -17.40%` | `2.9868 km² / -17.40%` | `2.9891 km² / -17.40%` | `2.9890 km² / -17.40%` | Invariant ($\Delta_{\text{pct}} = 0.0000\%$) |
| **Great Aletsch** | Earlier Area / Modern Area | `86.6225 / 78.1206` | `86.5652 / 78.0690` | `86.6292 / 78.1266` | `86.6225 / 78.1206` | -0.0661% / -0.0661% |
| **Great Aletsch** | Net Loss / Decadal Change | `8.5019 km² / -9.81%` | `8.4962 km² / -9.81%` | `8.5026 km² / -9.81%` | `8.5019 km² / -9.81%` | Invariant ($\Delta_{\text{pct}} = 0.0000\%$) |
| **Unteraar** | Earlier Area / Modern Area | `27.1505 / 20.8301` | `27.1315 / 20.8155` | `27.1521 / 20.8314` | `27.1505 / 20.8301` | -0.0700% / -0.0701% |
| **Unteraar** | Net Loss / Decadal Change | `6.3204 km² / -23.28%` | `6.3159 km² / -23.28%` | `6.3207 km² / -23.28%` | `6.3204 km² / -23.28%` | Invariant ($\Delta_{\text{pct}} = 0.0001\%$) |
| **Gorner** | Earlier Area / Modern Area | `57.7492 / 42.0219` | `57.7153 / 41.9969` | `57.7677 / 42.0354` | `57.7492 / 42.0219` | -0.0587% / -0.0595% |
| **Gorner** | Net Loss / Decadal Change | `15.7273 km² / -27.23%` | `15.7184 km² / -27.23%` | `15.7323 km² / -27.23%` | `15.7273 km² / -27.23%` | Invariant ($\Delta_{\text{pct}} = 0.0005\%$) |

---

## 5. Live Browser Verification & Regression Suite

Automated headless Chrome CDP test results (`findel_and_benchmarks_verification_results.json`):
- **Rhone:** Earlier: `17.166 km²`, Modern: `14.179 km²`, Net Loss: `-2.987 km²` (Screenshot: `verify_1_rhone.png`)
- **Great Aletsch:** Earlier: `86.565 km²`, Modern: `78.069 km²`, Net Loss: `-8.496 km²` (Screenshot: `verify_2_great_aletsch.png`)
- **Unteraar:** Earlier: `27.131 km²`, Modern: `20.816 km²`, Net Loss: `-6.316 km²` (Screenshot: `verify_3_unteraar.png`)
- **Gorner:** Earlier: `57.715 km²`, Modern: `41.997 km²`, Net Loss: `-15.718 km²` (Screenshot: `verify_4_gorner.png`)
- **Findel:** Earlier: `18.603 km²`, Modern: `16.072 km²`, Net Loss: `-2.531 km²`, Badge: `▼ -13.60%` (Screenshot: `verify_5_findel.png`)

Full repository test suite: **53 passed in 20.97s** (100% passing).
