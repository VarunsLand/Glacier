"""GlacierVision Interactive 4-Stage Segmentation Laboratory.

Implements:
- STEP 1: Upload 13-band Sentinel-2 NetCDF (.nc) or select from 4 verified held-out test scenes.
- STEP 2: Choose Model Architecture (ProtoNet Few-Shot vs Supervised 13-Band U-Net Baseline).
- STEP 3: Configure K-shot budget (K in {1, 5, 10, 20}) and execute live PyTorch forward pass.
- STEP 4: 4-panel diagnostic visualization (RGB contour, binary mask, probability heatmap, error map),
  real-time metrics (Dice, IoU, Precision, Recall, Accuracy), and PNG/CSV mask exports.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st

from src.inference import (
    get_device,
    read_and_validate_netcdf,
    run_unet_inference,
    run_protonet_inference,
    compute_segmentation_metrics,
    generate_contour_overlay,
    generate_error_classification_map,
    get_demo_netcdf_candidates,
)
from src.ui.theme import render_header
from src.ui.components import render_kpi_cards


def render_segmentation_lab():
    """Render the 4-stage interactive segmentation lab."""
    render_header(
        title="Interactive Glacier Segmentation Laboratory",
        subtitle="Pixel-Level Delineation on 13-Band Multispectral Sentinel-2 Earth Observation Data",
        badge="OBJECTIVE 1 • LIVE INFERENCE",
    )

    # Transparency & Demonstration Mode Guide
    st.info("ℹ️ **Demonstration Mode:** You can upload custom 13-band Sentinel-2 NetCDF (`.nc`) files for live model execution or select from pre-indexed held-out test glaciers. Pretrained weights are loaded directly from disk without mock inference.")

    # -----------------------------------------------------------------
    # STEP 1: UPLOAD SATELLITE DATA OR SELECT DEMO
    # -----------------------------------------------------------------
    st.markdown("""
    <div style="margin-top: 20px; margin-bottom: 12px;">
        <span class="badge-ice">STEP 1 OF 4</span>
        <h3 style="color: #F4FAFF; font-size: 1.3rem; font-weight: 700; margin: 6px 0;">
            1️⃣ Input Satellite Data Selection
        </h3>
    </div>
    """, unsafe_allow_html=True)

    c_mode, c_pick = st.columns([1.2, 1.8])
    with c_mode:
        input_type = st.radio(
            "Choose Input Pipeline:",
            [
                "🌟 Select Held-Out Test Glacier (DL4GAM Split 1)",
                "📤 Upload Custom Multispectral NetCDF (.nc)",
            ],
            index=0,
        )

    demo_cands = get_demo_netcdf_candidates()
    scene_data: Optional[Dict] = None

    if input_type == "🌟 Select Held-Out Test Glacier (DL4GAM Split 1)":
        with c_pick:
            demo_names = [f"{c['title']} ({c['id']})" for c in demo_cands]
            sel_idx = st.selectbox("Select Test Glacier Scene:", range(len(demo_names)), format_func=lambda i: demo_names[i])
            chosen_cand = demo_cands[sel_idx]
            st.caption(f"**Scene Description:** {chosen_cand['description']}")

        try:
            scene_data = read_and_validate_netcdf(chosen_cand["path"])
        except Exception as e:
            st.error(f"Error loading demo scene: {e}")

    else:
        with c_pick:
            uploaded_nc = st.file_uploader(
                "Upload 13-Channel NetCDF Patch (.nc):",
                type=["nc"],
                help="Requires NetCDF file containing 'band_data' array with 13 Sentinel-2 channels (and optional 'mask_crt_g').",
            )
            st.caption("ℹ️ *Format Constraint: Ordinary JPEG/PNG photos cannot be used for pixel-level segmentation because our models require all 13 multispectral channels. For standard RGB photos, use the Vision-Language CLIP/CoOp Lab.*")

        if uploaded_nc is not None:
            try:
                with st.spinner("Validating NetCDF multispectral structure..."):
                    scene_data = read_and_validate_netcdf(uploaded_nc)
                st.success(f"Validated '{uploaded_nc.name}' ({scene_data['metadata']['dimensions']}, 13 Bands).")
            except Exception as e:
                st.error(f"Invalid NetCDF format: {e}")

    if scene_data is None:
        st.warning("Please select a demo patch or upload a valid 13-channel NetCDF file to proceed.")
        return

    # Multispectral Channel Previews
    st.markdown("##### 🛰️ Multispectral Preview Channels")
    t1, t2, t3 = st.tabs(["True-Color RGB (B4-B3-B2)", "False-Color SWIR (B11-B8-B4)", "NDSI Snow/Ice Map"])
    with t1:
        st.image(scene_data["rgb_preview"], caption="True Color RGB (Percentile Stretched)", use_container_width=True)
    with t2:
        st.image(scene_data["swir_preview"], caption="False-Color SWIR Composite (Glaciers appear cyan; rock/debris appears brown)", use_container_width=True)
    with t3:
        fig_ndsi = px.imshow(scene_data["ndsi"], color_continuous_scale="Viridis", zmin=-0.8, zmax=0.9)
        fig_ndsi.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), template="plotly_dark")
        st.plotly_chart(fig_ndsi, use_container_width=True)

    meta = scene_data["metadata"]
    st.caption(f"**Dimensions:** `{meta['dimensions']}` | **Spectral Bands:** `{meta['channels']}` | **Ground Truth Available:** `{'Yes (mask_crt_g)' if meta['has_ground_truth'] else 'No'}`")

    st.markdown("---")

    # -----------------------------------------------------------------
    # STEP 2 & 3: CHOOSE MODEL ARCHITECTURE & CONFIGURE RUN
    # -----------------------------------------------------------------
    st.markdown("""
    <div style="margin-top: 10px; margin-bottom: 12px;">
        <span class="badge-ice">STEPS 2 & 3 OF 4</span>
        <h3 style="color: #F4FAFF; font-size: 1.3rem; font-weight: 700; margin: 6px 0;">
            2️⃣ Model Architecture & 3️⃣ Inference Configuration
        </h3>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
        <div class="glass-card" style="min-height: 200px;">
            <h4 style="color: #12DFF3; margin-top: 0;">🚀 Supervised 13-Band U-Net Baseline</h4>
            <p style="color: #A8BED0; font-size: 0.88rem; line-height: 1.6;">
                Conventional fully-supervised U-Net architecture trained directly on all 1,475 training patches with full backpropagation. 
                Produces high-contrast segmentations with strong spatial localization across all 13 Sentinel-2 bands.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="glass-card" style="min-height: 200px;">
            <h4 style="color: #65CDEB; margin-top: 0;">❄️ Prototypical Network (Few-Shot ProtoNet)</h4>
            <p style="color: #A8BED0; font-size: 0.88rem; line-height: 1.6;">
                Episodic meta-learning model. Extracts spatial prototypes from just $K \\in \\{1, 5, 10, 20\\}$ labeled support patches 
                sampled deterministically from <code>fold_train</code>, classifying query pixels via cosine similarity without weight updates.
            </p>
        </div>
        """, unsafe_allow_html=True)

    model_sel = st.radio(
        "Select Active Inference Engine:",
        [
            "Supervised 13-Band U-Net Baseline",
            "Prototypical Network (Few-Shot ProtoNet)",
        ],
        index=0,
    )

    k_shots = 5
    if "ProtoNet" in model_sel:
        k_shots = st.select_slider(
            "Select Few-Shot Support Budget (K labeled patches):",
            options=[1, 5, 10, 20],
            value=5,
            help="Number of labeled training patches sampled from fold_train to form class prototypes.",
        )
        st.caption(f"ProtoNet will extract prototypes using $K={k_shots}$ support patches without parameter updates.")

    run_inference = st.button("⚡ Run Live AI Inference", type="primary", use_container_width=True)

    # -----------------------------------------------------------------
    # STEP 4: VIEW RESULTS
    # -----------------------------------------------------------------
    if run_inference:
        st.markdown("---")
        st.markdown("""
        <div style="margin-top: 10px; margin-bottom: 12px;">
            <span class="badge-cyan">STEP 4 OF 4</span>
            <h3 style="color: #F4FAFF; font-size: 1.3rem; font-weight: 700; margin: 6px 0;">
                4️⃣ Live Inference Results & Scientific Evaluation
            </h3>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Executing live forward pass with PyTorch..."):
            raw_bands = scene_data["raw_bands"]
            gt_mask = scene_data["gt_mask"]

            if "U-Net Baseline" in model_sel:
                probs, pred_mask, latency = run_unet_inference(raw_bands)
                active_model_name = "Supervised 13-Band U-Net Baseline"
            else:
                probs, pred_mask, latency = run_protonet_inference(raw_bands, k_shots=k_shots)
                active_model_name = f"Few-Shot ProtoNet (K={k_shots})"

        st.success(f"Execution complete: **{active_model_name}** finished in **{latency:.1f} ms** on `{get_device().type.upper()}`.")

        # Metric Tiles (if ground truth available)
        if gt_mask is not None:
            metrics = compute_segmentation_metrics(pred_mask, gt_mask)
            seg_kpi_cards = [
                {
                    "label": "Dice Score",
                    "value": f"{metrics['dice']:.4f}",
                    "unit": "",
                    "badge": "Spatial Overlap",
                    "badge_type": "success" if metrics['dice'] >= 0.6 else "cyan",
                    "arrow": "▲" if metrics['dice'] >= 0.6 else None,
                    "value_color": "#34D399" if metrics['dice'] >= 0.6 else "#12DFF3",
                },
                {
                    "label": "IoU (Jaccard)",
                    "value": f"{metrics['iou']:.4f}",
                    "unit": "",
                    "badge": "Ice Pixel IoU",
                    "badge_type": "cyan",
                    "arrow": None,
                    "value_color": "#12DFF3",
                },
                {
                    "label": "Precision",
                    "value": f"{metrics['precision']:.4f}",
                    "unit": "",
                    "badge": "Positive Predictive",
                    "badge_type": "cyan",
                    "arrow": None,
                    "value_color": "#12DFF3",
                },
                {
                    "label": "Recall (Sensitivity)",
                    "value": f"{metrics['recall']:.4f}",
                    "unit": "",
                    "badge": "True Positive Rate",
                    "badge_type": "cyan",
                    "arrow": None,
                    "value_color": "#12DFF3",
                },
                {
                    "label": "Pixel Accuracy",
                    "value": f"{metrics['accuracy']:.2%}",
                    "unit": "",
                    "badge": "Global Correctness",
                    "badge_type": "success",
                    "arrow": "▲",
                    "value_color": "#34D399",
                },
            ]
            render_kpi_cards(seg_kpi_cards, grid_class="kpi-grid-5")

        # Multi-Panel Visualizations
        if gt_mask is not None:
            p1, p2, p3, p4, p5 = st.columns(5)
            with p1:
                overlay_pred = generate_contour_overlay(scene_data["rgb_preview"], pred_mask, contour_color=(18, 223, 243))
                st.image(overlay_pred, caption="1. RGB + Boundary (Cyan)", use_container_width=True)
            with p2:
                st.image((gt_mask * 255).astype(np.uint8), caption="2. Ground Truth (mask_crt_g)", use_container_width=True)
            with p3:
                st.image((pred_mask * 255).astype(np.uint8), caption="3. Predicted Mask (τ=0.5)", use_container_width=True)
            with p4:
                fig_heat = px.imshow(probs, color_continuous_scale="Plasma", zmin=0.0, zmax=1.0)
                fig_heat.update_layout(margin=dict(l=5, r=5, t=5, b=5), height=210, coloraxis_showscale=False)
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption("4. Probabilities [0, 1]")
            with p5:
                error_map = generate_error_classification_map(pred_mask, gt_mask)
                st.image(error_map, caption="5. Error Discrepancy Map", use_container_width=True)
        else:
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                overlay_pred = generate_contour_overlay(scene_data["rgb_preview"], pred_mask, contour_color=(18, 223, 243))
                st.image(overlay_pred, caption="1. RGB + Boundary (Cyan)", use_container_width=True)
            with p2:
                overlay_swir = generate_contour_overlay(scene_data["swir_preview"], pred_mask, contour_color=(18, 223, 243))
                st.image(overlay_swir, caption="2. SWIR + Boundary Overlay", use_container_width=True)
            with p3:
                st.image((pred_mask * 255).astype(np.uint8), caption="3. Predicted Mask (τ=0.5)", use_container_width=True)
            with p4:
                fig_heat = px.imshow(probs, color_continuous_scale="Plasma", zmin=0.0, zmax=1.0)
                fig_heat.update_layout(margin=dict(l=5, r=5, t=5, b=5), height=210, coloraxis_showscale=False)
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption("4. Probabilities [0, 1]")

        # Plain-English Panel Explanations
        st.markdown("""
        **Panel Guide for Scientific Reviewers:**
        - **Panel 1 — Satellite RGB + Predicted Boundary:** Sentinel-2 true-color composite ($B_4, B_3, B_2$) with AI-predicted boundary contour in glowing cyan.
        - **Panel 2 — Ground Truth Mask (`mask_crt_g`):** Expert glaciological reference clean-ice outline labeled by cryospheric scientists ($256 \\times 256$ binary).
        - **Panel 3 — Predicted Binary Mask:** AI-delineated glacier segmentation mask ($256 \\times 256$, standard threshold $\\tau = 0.5$).
        - **Panel 4 — Prediction Probability Heatmap:** Continuous sigmoid model confidence scores $[0.0, 1.0]$ across every image pixel.
        - **Panel 5 — Diagnostic Error Discrepancy Map:** Pixel-by-pixel spatial classification breakdown:
          - 🟩 **True Positive (Bright Green):** AI correctly detected glacier ice pixels ($TP$).
          - 🟥 **False Positive (Vivid Red):** False alarm — AI predicted glacier ice where ground truth indicates non-glacier terrain ($FP$).
          - 🟦 **False Negative (Vivid Blue):** Missed detection — ground truth indicates glacier ice, but AI failed to capture it ($FN$).
          - ⬛ **True Negative (Dark Slate):** AI correctly identified non-glacier bedrock, moraine, shadow, or valley background ($TN$).
        """)

        # Cryospheric Segmentation Metric Guide
        with st.expander("📖 Cryospheric Segmentation Metric Guide (Plain-English Explanations)", expanded=False):
            st.markdown("""
            - **Dice Score:** Measures the spatial overlap between the AI prediction and the expert ground truth ($2|A \\cap B| / (|A| + |B|)$). Values range from 0.0 (no overlap) to 1.0 (perfect match). Higher is better.
            - **Mean IoU (mIoU):** Mean Intersection over Union averaged across glacier and non-glacier background pixels. Higher is better.
            - **Foreground IoU:** Intersection over Union calculated strictly on glacier ice pixels, testing how accurately the glacier itself is identified. Higher is better.
            - **Precision:** Fraction of AI-predicted glacier pixels that are truly glaciers. High precision means very few false alarms.
            - **Recall:** Fraction of true glaciers that the AI successfully found. High recall means very few missed glaciers.
            - **F1 Score:** Harmonic mean of precision and recall ($2 \\cdot P \\cdot R / (P + R)$), balancing false alarms against missed detections. Higher is better.
            - **Boundary F1:** Evaluates how precisely the AI traced the outer edge of the glacier boundary at pixel tolerance. Higher means sharper boundaries.
            
            *Scientific Note on Fair Comparison:* Higher scores generally indicate better performance. However, models must be evaluated within their specific training setting: Few-Shot ProtoNet learns from only $K \\in \\{1, 5, 10, 20\\}$ support patches per episode without parameter updates, whereas the supervised U-Net baseline was trained directly on all 1,475 training patches with full backpropagation.
            """)

        # Export Controls
        st.markdown("---")
        cd1, cd2 = st.columns(2)
        with cd1:
            mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
            buf = io.BytesIO()
            mask_pil.save(buf, format="PNG")
            st.download_button(
                label="💾 Download Predicted Mask as PNG",
                data=buf.getvalue(),
                file_name=f"glaciervision_{meta['source_name']}_mask.png",
                mime="image/png",
                use_container_width=True,
            )
        with cd2:
            df_csv = pd.DataFrame(pred_mask)
            csv_bytes = df_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download Binary Mask Matrix as CSV",
                data=csv_bytes,
                file_name=f"glaciervision_{meta['source_name']}_matrix.csv",
                mime="text/csv",
                use_container_width=True,
            )
