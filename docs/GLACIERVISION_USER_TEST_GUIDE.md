# GlacierVision: End-to-End User Acceptance Testing (UAT) Guide
**Project:** Few-Shot Computer Vision for Automated Glacier Detection and Retreat Monitoring  
**Target User:** Non-Developer Evaluator, Scientific Reviewer, and Academic Examiner  
**Platform Version:** 3.0.0 (Premium UI Redesign Release)  
**Date:** November 15, 2026  

---

## 🚀 1. Quick Launch Instructions (Step-by-Step)

The GlacierVision web platform runs locally via Streamlit with zero external cloud dependencies. All models, checkpoints, datasets, and publication figures are stored locally on your machine.

### Prerequisites Check
- Operating System: Windows 10/11 (64-bit)
- Python: Version 3.10 to 3.13 installed (`C:\Users\Varun's LOQ\AppData\Local\Programs\Python\Python313\python.exe`)
- Compute: Dedicated NVIDIA GPU (CUDA) or host CPU

### How to Launch the Web Application
1. Open **PowerShell** or **Command Prompt** on Windows.
2. Navigate to the project root directory and start the Streamlit server:
   ```powershell
   cd "C:\Users\Varun's LOQ\Desktop\GlacierResearch"
   & "C:\Users\Varun's LOQ\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py --server.port 8501
   ```
3. Your default web browser will open automatically. If not, open your browser and navigate to:
   ```text
   http://localhost:8501
   ```
4. **Landing Screen:** The application opens on the **`🏠 Home`** cinematic landing page.

---

## 🧭 2. Global Application Navigation Map

The redesigned GlacierVision platform features a unified, dark navy (`#061426`), ice-blue (`#65CDEB`), and glowing cyan (`#12DFF3`) theme. The left sidebar hosts the persistent navigation menu across all 8 modules:

```text
❄️ GlacierVision Navigation Menu:
  ├── 🏠 Home                                 (Cinematic landing page & feature overview)
  ├── 🔬 Interactive Segmentation Lab         (4-stage 13-band multispectral AI inference)
  ├── 🗺️ Glacier Retreat Explorer             (Geographical workspace & vector outlines)
  ├── 📊 Area Assessment Workspace             (Quantitative 81,598-glacier area accounting)
  ├── 🛰️ Visual Monitoring Suite              (Sentinel-2, Landsat-8, GLIMS 300 DPI suites)
  ├── 🧠 CLIP vs CoOp Lab                     (Vision-Language Zero-Shot vs Learnable CoOp)
  ├── 📚 Research Results Center              (Organized benchmarks across all 5 objectives)
  └── 📖 Datasets & Methodology               (Data provenance, zero leakage & citations)
```

---

## 🧪 3. Detailed Walkthrough & Verification Steps

---

### Page 1: `🏠 Home` (Cinematic Landing Page)
* **Goal:** Verify the high-impact landing page, dynamic statistics, and navigation triggers.
* **Navigation:** Select `🏠 Home` in the sidebar.
* **What to Inspect:**
  1. **Hero Banner:** Confirms "GLACIERVISION", subtitle "AI-Powered Glacier Detection and Retreat Monitoring", and the glowing climate-tech styling.
  2. **Call-to-Action Buttons:**
     - Click **`🚀 Try GlacierVision Lab`**: Verifies it smoothly routes to `🔬 Interactive Segmentation Lab`.
     - Return to Home, then click **`📚 Explore Research Results`**: Verifies it routes to `📚 Research Results Center`.
  3. **Verified Project Statistics Tiles:**
     - **Analyzed Glaciers:** `81,598`
     - **Net Area Loss:** `16,321.76 km²`
     - **Retreating Basins:** `67.1%`
     - **U-Net Test Dice:** `0.6559`
     - **CoOp Prompt Gain:** `+9.0% Acc`
  4. **Five Feature Cards:** Each card displays a genuine project visual thumbnail and a direct launch button.

---

### Page 2: `🔬 Interactive Segmentation Lab` (13-Band Multispectral)
* **Goal:** Test the 4-stage interactive deep learning segmentation pipeline.
* **Navigation:** Select `🔬 Interactive Segmentation Lab` in the sidebar.
* **Testing Steps:**
  1. **Stage 1 (Input Data):**
     - Keep `🌟 Select Held-Out Test Glacier` selected.
     - Choose **`Clean Alpine Glacier (High Ice Coverage: ~88%) (g_4367)`** from the dropdown.
     - Inspect the three tabs: *True-Color RGB*, *False-Color SWIR*, and *NDSI Snow/Ice Map*. Notice how glacier ice appears bright cyan in SWIR.
  2. **Stage 2 (Model Architecture):**
     - Compare the two model cards: **Supervised 13-Band U-Net Baseline** vs **Prototypical Network (Few-Shot ProtoNet)**.
     - Select **`Supervised 13-Band U-Net Baseline`**.
  3. **Stage 3 (Configure & Run):**
     - Click the prominent blue **`⚡ Run Live AI Inference`** button.
  4. **Stage 4 (Results & Evaluation):**
     - Observe the execution latency badge (typically `< 50 ms` on GPU).
     - Inspect live metric cards: **Dice Score (~0.88)**, **IoU (~0.79)**, **Precision**, **Recall**, and **Pixel Accuracy**.
     - Inspect the 4-panel visual result:
       1. RGB satellite image with AI-predicted cyan boundary contour overlay.
       2. Clean binary glacier mask ($256 \times 256$).
       3. Prediction probability heatmap $[0.0, 1.0]$.
       4. 4-color diagnostic error classification map (Green=TP, Red=FP, Blue=FN, Dark Slate=TN).
     - Click **`💾 Download Predicted Mask as PNG`** and **`📄 Download Binary Mask Matrix as CSV`** to verify data export.
  5. **Test ProtoNet Few-Shot Mode:**
     - Select **`Prototypical Network (Few-Shot ProtoNet)`**.
     - Adjust the *Support Budget* slider to $K=5$ or $K=20$.
     - Click **`⚡ Run Live AI Inference`** again. Verify few-shot prototypes are dynamically extracted from `fold_train`.
  6. **(Optional) Test File Upload:**
     - Switch input source to `📤 Upload Custom Multispectral NetCDF (.nc)`.
     - Upload `datasets/dl4gam/extracted/g_3560/g_3560_patch_0_xc_256_yc_256.nc`.
     - Verify the file is parsed, channels validated, and preview rendered.

---

### Page 3: `🗺️ Glacier Retreat Explorer` (Geographical Workspace)
* **Goal:** Explore real multi-temporal GLIMS glacier outlines and simulate retreat trajectories.
* **Navigation:** Select `🗺️ Glacier Retreat Explorer` in the sidebar.
* **Testing Steps:**
  1. **Quick Benchmark Glaciers:**
     - Click **`Rhone Glacier`**. Verify the assessment automatically loads GLIMS basin `G008379E46648N`.
     - Historical Area: `17.63 km²` (1850).
     - Modern Area: `15.30 km²` (2010).
     - Net Change: `-2.33 km²` (`-13.22%`).
     - Dynamic Classification: **Significant Retreat**.
  2. **Multi-Temporal Vector Outlines:**
     - Check the European Alps UTM (EPSG:32632) multi-temporal vector boundary visualization on the left.
  3. **Area Budget Waterfall Chart:**
     - Check the interactive Plotly waterfall chart showing historical area, net loss, and modern area.
  4. **Report Export:**
     - Click **`💾 Download Retreat Accounting Report for Rhone (JSON)`** and verify the JSON export.

---

### Page 4: `📊 Area Assessment Workspace` (Quantitative Accounting)
* **Goal:** Verify macro-level glaciological accounting and full 81,598-row database search.
* **Navigation:** Select `📊 Area Assessment Workspace` in the sidebar.
* **Testing Steps:**
  1. **Accounting Verification:**
     - Confirm net area loss displays **16,321.76 km²** (-7.40% pooled surface loss) across 81,598 glaciers.
     - Expand **📐 Mathematical Accounting Formulas & Units** to review formal definitions.
  2. **Dynamic Parameter Simulation:**
     - Enter Initial Area = `50.0 km²`, Final Area = `42.5 km²`, Observation Interval = `15.0 Years`.
     - Verify instant recalculations: Net Change = `-7.50 km²` (`-15.00%`), Rate = `-0.500 km²/yr`, Class = **Significant Retreat**.
  3. **Dynamic Temporal Grouping Table:**
     - Verify all 5 intervals (3–5 yrs, 5–10 yrs, 10–15 yrs, 15–20 yrs, >20 yrs) display with correct basin counts and net losses.
  4. **Full 81,598 Database Search:**
     - In the search box, type `Great Aletsch` (or `G008032E46504N`).
     - Verify the matching row appears with earlier area `105.58 km²` and later area `82.79 km²`.
     - Test category dropdown filter (e.g., select `Significant Retreat` or `Advance / Growth`).

---

### Page 5: `🛰️ Visual Monitoring Suite` (Multi-Sensor 300 DPI)
* **Goal:** Inspect publication-quality visual suites and verify scientific sensor disclaimers.
* **Navigation:** Select `🛰️ Visual Monitoring Suite` in the sidebar.
* **Testing Steps:**
  1. **GLIMS Boundary Overlays Tab:** View multi-temporal outlines for Rhone, Great Aletsch, and Unteraar.
  2. **Sentinel-2 True Colour Tab:** Inspect 10m visible spectrum RGB composite.
  3. **Sentinel-2 False Colour SWIR Tab:** Inspect shortwave infrared composite where ice absorbs and appears cyan.
  4. **Cryogenic Indices & DEM Tab:** Inspect NDSI snow/ice map and formula.
  5. **Landsat-8 Bitemporal Change Tab:** Confirm the scientific disclaimer explaining that an undirected spectral surface change mask highlights surface alterations without asserting directional retreat or advance.

---

### Page 6: `🧠 CLIP vs CoOp Lab` (Vision-Language AI)
* **Goal:** Test live classification comparing frozen OpenAI CLIP ViT-B-32 against Learnable CoOp ($M=16$).
* **Navigation:** Select `🧠 CLIP vs CoOp Lab` in the sidebar.
* **Testing Steps:**
  1. **Live Classification on Clean Ice:**
     - Under *Select Sample*, choose **`Clean Alpine Glacier Ice (56.9% Ice)`**.
     - Observe dual-model inference: both Handcrafted CLIP and Learnable CoOp classify as **Glacier Ice** with >98% confidence.
  2. **Live Classification on Exposed Bedrock Ridge:**
     - Choose **`Exposed Alpine Bedrock Ridge (1.5% Ice)`**.
     - Observe that Learnable CoOp correctly classifies as **Mountain Bedrock**, reducing false alarms compared to zero-shot CLIP.
  3. **Dataset-Level Held-Out Benchmark ($N=100$ Patches):**
     - Review the comparative table: CoOp Accuracy = **81.00%** (+9.0% vs CLIP), Precision = **75.86%** (+13.97% vs CLIP), False Positives = **12 FP vs 27 FP (-55.6%)**.
     - Inspect the confusion matrices and performance bar chart.

---

### Page 7: `📚 Research Results Center` (Academic Benchmarks)
* **Goal:** Review peer-reviewed experimental benchmarks across all 5 objectives.
* **Navigation:** Select `📚 Research Results Center` in the sidebar.
* **Testing Steps:**
  1. **Objective 1 Tab:** Inspect validation Dice progression curves, 534 held-out test patch metrics table, and the 80 individual paired query episode expander.
  2. **Objectives 2 & 3 Tab:** Inspect 81,598-glacier donut chart and accounting breakdown.
  3. **Objective 4 Tab:** View 300 DPI publication visual suites.
  4. **Objective 5 Tab:** View CLIP vs CoOp prompt tuning benchmark.
  5. **Acceptance Reports & Downloads Tab:** Inspect markdown audit reports directly in the app or download them.

---

### Page 8: `📖 Datasets & Methodology`
* **Goal:** Verify dataset provenance, zero-leakage protocols, and hardware specifications.
* **Navigation:** Select `📖 Datasets & Methodology` in the sidebar.
* **What to Inspect:**
  - DL4GAM (13 Sentinel-2 channels, Alps split 1).
  - GLIMS (81,598 multi-temporal basins).
  - Landsat-8 (30m OLI).
  - Zero data-leakage protocols and hardware environment profile.

---

## 🛠️ 4. Automated Verification Suite Execution

To run all **35 automated unit and regression tests** from the terminal:

```powershell
cd "C:\Users\Varun's LOQ\Desktop\GlacierResearch"
& "C:\Users\Varun's LOQ\AppData\Local\Programs\Python\Python313\python.exe" -m pytest tests/
```

**Expected Passing Summary:**
```text
======================= 35 passed, 1 warning in 33.49s ========================
```
- `tests/test_final_website_improvements.py` (6 tests passed)
- `tests/test_interactive_platform.py` (7 tests passed)
- `tests/test_objectives_2_3_4.py` (13 tests passed)
- `tests/test_ui_redesign.py` (9 tests passed)
