# GLACIERVISION — OBJECTIVE 4 AUDIT REPORT
## Sentinel-2 SWIR False-Colour Composite Channel Mapping Fix & Verification

**Project:** Few-Shot Computer Vision for Automated Glacier Detection and Retreat Monitoring  
**Target:** Objective 4: Interactive Multi-Sensor Visual Glacier Monitoring Workspace  
**Test Asset:** `datasets/objective4/sentinel2_rhone/rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif` (Multiband GeoTIFF: B02, B03, B04, B08, B11)  
**Date:** November 16, 2026  
**Status:** **PASSED & VERIFIED (104/104 Regression Tests Passing, Live Browser CDP Verified)**  

---

## 1. Executive Summary & Problem Resolution

During user testing of Objective 4 with the live 5-band Sentinel-2 Rhone GeoTIFF (`B02, B03, B04, B08, B11`), the Glaciological False-Colour SWIR composite exhibited an incorrect channel assignment:
- **Reported Buggy State:** Red = `B11`, Green = `B11`, Blue = `B08`
- **Glaciological Intended State:** Red = `B11` (SWIR 1, ~1610 nm), Green = `B08` (NIR, ~842 nm), Blue = `B04` (Red, ~665 nm)

### Root Cause Analysis
1. **Fallback Index Clamp for 5-Band Rasters:**
   Prior to band resolution logic, default channel indices were calculated as `min(11, len-1)` for SWIR, `min(7, len-1)` for NIR, and `min(3, len-1)` for Red. For a 5-band raster (`len = 5`, maximum index 4):
   - Red index: $\min(11, 4) = 4 \rightarrow \text{Band } 4 = \text{B11}$
   - Green index: $\min(7, 4) = 4 \rightarrow \text{Band } 4 = \text{B11}$ (Duplicate assignment)
   - Blue index: $\min(3, 4) = 3 \rightarrow \text{Band } 3 = \text{B08}$
2. **Streamlit Session State Widget Persistence:**
   In Streamlit, `st.selectbox` widgets bound to `key="swir_g"` cache the selected option in `st.session_state["swir_g"]`. When a widget key already exists in session state, Streamlit's runtime completely ignores the `index=` parameter on reruns. Thus, even after adding band resolution, sessions where `swir_g` had been initialized to `"B11"` remained stuck on `"B11"`.
3. **Band Convention Variance:**
   13-band DL4GAM patches use single-digit aliases (`B8`, `B4`), whereas Sentinel-2 Level-2A GeoTIFFs use two-digit aliases (`B08`, `B04`). Furthermore, DL4GAM arrays contain both wide NIR (`B8`, 10m) and narrow NIR (`B8A`, 20m), requiring strict primary band precedence so `B8A` is never mistakenly picked over `B8`.

---

## 2. Engineering Corrections Implemented

### A. Intelligent Primary Band Resolver (`src/visual_analysis.py`)
Enhanced `resolve_s2_band_indices()` with a two-pass priority selector:
- **First pass (Primary):** Prioritizes exact primary glaciological bands (`B02/B2`, `B03/B3`, `B04/B4`, `B08/B8`, `B11`).
- **Second pass (Fallback):** Only evaluates narrow edge channels (`B8A`, `B11A`) if primary bands are missing.

```python
def resolve_s2_band_indices(band_names: List[str]) -> Dict[str, Optional[int]]:
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
    return res
```

### B. Dynamic Session State Synchronization (`src/ui/visual_monitoring.py`)
Auto-detects scene transitions, uninitialized keys, or corrupt state (such as `swir_r == swir_g`) and synchronizes `st.session_state` before widget instantiation:

```python
# Auto-synchronize session state when active scene changes or corrupted state is detected
current_scene_key = f"swir_{s2_data.get('filename', s2_data.get('patch_id', 'scene'))}"
needs_swir_init = (
    st.session_state.get("swir_dataset_tracker") != current_scene_key
    or "swir_r" not in st.session_state
    or "swir_g" not in st.session_state
    or "swir_b" not in st.session_state
    or st.session_state["swir_r"] not in s2_data["band_names"]
    or st.session_state["swir_g"] not in s2_data["band_names"]
    or st.session_state["swir_b"] not in s2_data["band_names"]
    or st.session_state["swir_r"] == st.session_state["swir_g"]  # Fix duplicate assignment bug
)

if needs_swir_init:
    st.session_state["swir_dataset_tracker"] = current_scene_key
    st.session_state["swir_r"] = target_r
    st.session_state["swir_g"] = target_g
    st.session_state["swir_b"] = target_b
```

### C. Dedicated Reset to Default Button
Added an interactive reset button allowing instant reversion to standard glaciological channel mapping:
```python
if st.button(f"🔄 Reset to Default ({target_r}-{target_g}-{target_b})", key="reset_swir_btn"):
    st.session_state["swir_r"] = target_r
    st.session_state["swir_g"] = target_g
    st.session_state["swir_b"] = target_b
    st.rerun()
```

### D. Physical Interpretation & Wavelength Text
Expanded scientific documentation to explain the physical color synthesis:
- **Red (R):** `B11` ($1610\text{ nm}$, SWIR 1) — Strong vibrational absorption by water molecules in glacier ice; renders clean ice dark in this channel.
- **Green (G):** `B08` ($842\text{ nm}$, NIR) — Moderate ice reflectance, strong vegetation canopy reflectance, total absorption by open water.
- **Blue (B):** `B04` ($665\text{ nm}$, Visible Red) — High optical scattering by snow and glacier ice ($>80\%$).
- **Composite Color Signature:** Low Red + Moderate Green + High Blue yields vibrant **electric cyan / aquamarine** for glacier ice; uniform reflectance across all three channels yields **opaque bright white** for clouds; moderate SWIR/visible with low NIR yields **brown/tan** for bedrock and moraines.

---

## 3. Physical Verification Results

| Dimension / Requirement | Verified Value / Behavior | Status |
| :--- | :--- | :--- |
| **Default Red Channel** | `B11` ($1610\text{ nm}$, SWIR 1) | **VERIFIED** |
| **Default Green Channel** | `B08` ($842\text{ nm}$, NIR) | **VERIFIED** |
| **Default Blue Channel** | `B04` ($665\text{ nm}$, Red) | **VERIFIED** |
| **Selector Disjointness** | $R \neq G \neq B$ (`B11`, `B08`, `B04` are distinct) | **VERIFIED** |
| **Image Caption** | `False-Color SWIR Composite (R: B11 / G: B08 / B: B04) — rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif` | **VERIFIED** |
| **Reset to Default Button** | Restores $R=\text{B11}, G=\text{B08}, B=\text{B04}$ upon click | **VERIFIED** |
| **Custom Selection Update** | Interactive channel changes update the output array immediately | **VERIFIED** |
| **Common Grid Preservation** | Preserved native $20.0\text{m} \times 20.0\text{m}$ grid in `EPSG:32632` | **VERIFIED** |

---

## 4. Automated Testing & Verification Evidence

### Objective 4 Verification Test Suite (`tests/test_objective_4_verification.py`)
- `test_swir_default_channel_mapping_rhone_and_dl4gam`: Verifies disjoint default mappings for 5-band GeoTIFF (`B11, B08, B04`) and 13-band NetCDF (`B11, B8, B4`).
- `test_swir_interactive_selection_and_reset_apptest`: Tests simulated Streamlit AppTest widget interaction, verifying custom channel selection (`swir_b = 'B2'`) and reset restoration (`swir_b = 'B4'`).
- `test_rhone_multiband_geotiff_upload_parsing`: Verifies in-memory GeoTIFF parsing with Rasterio.
- *Total Objective 4 Tests:* **24/24 PASSED** (100%).

### Full Repository Regression Test Suite
- **104 passed, 0 failed** in 32.36 seconds across all Objectives 1, 2, 3, 4, UI, and ML pipelines.

### Live Browser CDP Automation
- Script: `scripts/verify_swir_mapping_live.py`
- Executed via Chrome DevTools Protocol against headless Chrome on `http://localhost:8501/?page=monitoring`.
- Verified live upload of `rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif`, validated DOM text, verified button functionality, and captured high-resolution screenshot `rhone_swir_b11_b08_b04_verified.png`.
