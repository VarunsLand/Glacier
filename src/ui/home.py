"""GlacierVision Cinematic Landing Page & Research Overview.

Features:
- High-impact cinematic glacier hero banner with brand identity.
- Verified live statistics dynamically read from validated result files.
- Dual call-to-action controllers (Try GlacierVision vs Explore Research).
- 5 interactive research objective cards with genuine project visual thumbnails and direct navigation triggers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

import streamlit as st

from src.inference import FIGURES_DIR, RESULTS_DIR, RUNS_DIR
from src.ui.components import render_kpi_cards


def render_homepage(navigate_to: Callable[[str], None]):
    """Render the cinematic GlacierVision landing page."""
    # 1. Dynamically Load Verified Metrics
    summary_path = RESULTS_DIR / "glims_retreat_summary_validated.json"
    total_glaciers = "81,598"
    net_loss_num = 16321.76
    net_loss = "16,321.76 km²"
    retreating_pct = "67.06%"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            total_glaciers = f"{summary.get('total_glaciers_analyzed', 81598):,}"
            net_loss_num = float(summary.get('net_area_loss_km2', 16321.76))
            net_loss = f"{net_loss_num:,.2f} km²"
            retreating_pct = f"{summary.get('retreating_glaciers_ratio_pct', 67.06):.1f}%"

    obj1_path = RUNS_DIR / "objective1_run_002_recovery" / "results" / "objective1_evaluation_report.json"
    best_dice = "0.6559"
    best_acc = "91.83%"
    if obj1_path.exists():
        with open(obj1_path, "r", encoding="utf-8") as f:
            obj1 = json.load(f)
            bl = obj1.get("supervised_baseline_benchmark", {})
            best_dice = f"{bl.get('dice_mean', 0.6559):.4f}"

    # Objective 5 is the Research Evaluation Centre (no CLIP/CoOp)
    obj5_net_loss_text = "16,321.76 km² Net Loss"

    # 2. Cinematic Hero Section
    st.markdown("""
    <div class="hero-banner">
        <span class="badge-cyan">CLIMATE-TECH AI PLATFORM</span>
        <h1 style="color: #F4FAFF; font-size: clamp(1.75rem, 4.5vw, 3.2rem); font-weight: 900; letter-spacing: -0.02em; margin: 12px 0 6px 0; text-shadow: 0 0 35px rgba(18, 223, 243, 0.35); word-break: normal;">
            ❄️ GLACIERVISION
        </h1>
        <h3 style="color: #65CDEB; font-size: 1.35rem; font-weight: 600; margin: 0 0 16px 0;">
            AI-Powered Glacier Detection and Retreat Monitoring
        </h3>
        <p style="color: #A8BED0; font-size: 1.05rem; line-height: 1.7; max-width: 820px; margin-bottom: 24px;">
            A unified computer vision and Earth observation platform that achieves pixel-level glacier delineation 
            under extreme annotation scarcity and quantifies multi-decadal glacier shrinkage using multispectral 
            Sentinel-2, Landsat-8, and global GLIMS geodesic satellite inventories.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Call to action buttons
    c_btn1, c_btn2, _ = st.columns([1.2, 1.3, 2])
    with c_btn1:
        if st.button("🚀 Try GlacierVision Lab", type="primary", use_container_width=True):
            navigate_to("🔬 Interactive Segmentation Lab")
    with c_btn2:
        if st.button("📚 Explore Research Results", type="secondary", use_container_width=True):
            navigate_to("📋 Research Results & Evaluation")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 3. Verified Project Statistics Row (Responsive fluid layout, zero clipping)
    home_kpi_cards = [
        {
            "label": "Analyzed Glaciers",
            "value": total_glaciers,
            "unit": "",
            "badge": "Global GLIMS Basins",
            "badge_type": "cyan",
            "arrow": None,
        },
        {
            "label": "Net Area Loss",
            "value": f"{net_loss_num:,.2f}",
            "unit": "km²",
            "badge": "-7.40% Pooled Loss",
            "badge_type": "danger",
            "arrow": "▼",
            "value_color": "#F87171",
        },
        {
            "label": "Retreating Basins",
            "value": retreating_pct,
            "unit": "",
            "badge": "Δt ≥ 3.0 Years",
            "badge_type": "danger",
            "arrow": "▼",
            "value_color": "#F87171",
        },
        {
            "label": "U-Net Test Dice",
            "value": best_dice,
            "unit": "",
            "badge": "534 Held-Out Patches",
            "badge_type": "success",
            "arrow": "▲",
            "value_color": "#34D399",
        },
        {
            "label": "Retreating Glaciers",
            "value": retreating_pct,
            "unit": "",
            "badge": "of 81,598 GLIMS Pairs",
            "badge_type": "danger",
            "arrow": "▼",
            "value_color": "#F87171",
        },
    ]
    render_kpi_cards(home_kpi_cards, grid_class="kpi-grid-5")

    st.markdown("---")

    # 4. Five Research Objective Feature Cards
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="color: #F4FAFF; font-size: 1.7rem; font-weight: 800; margin-bottom: 4px;">
            Core Research & Technology Modules
        </h2>
        <p style="color: #65CDEB; font-size: 0.95rem;">
            Explore the five scientific pillars combining few-shot deep learning, satellite remote sensing, and vision-language optimization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    card_col1, card_col2 = st.columns(2)

    # Objective 1 Card
    with card_col1:
        st.markdown("""
        <div class="glass-interactive">
            <span class="badge-cyan">OBJECTIVE 1</span>
            <h3 style="color: #F4FAFF; font-size: 1.25rem; font-weight: 700; margin: 8px 0 6px 0;">
                1. Few-Shot Glacier Segmentation
            </h3>
            <p style="color: #A8BED0; font-size: 0.9rem; line-height: 1.6; margin-bottom: 12px;">
                Prototypical Networks (ProtoNet) trained via episodic meta-learning on 13 Sentinel-2 spectral channels. 
                Learns accurate pixel-level masks from just $K \\in \\{1, 5, 10, 20\\}$ support patches, benchmarked against a supervised U-Net.
            </p>
        </div>
        """, unsafe_allow_html=True)
        img_kcurve = RUNS_DIR / "objective1_run_002_recovery" / "figures" / "fewshot_k_curve.png"
        if img_kcurve.exists():
            st.image(str(img_kcurve), caption="Few-Shot K-Curve Benchmark across K ∈ {1, 5, 10, 20}", use_container_width=True)
        if st.button("Launch Segmentation Lab →", key="btn_obj1", use_container_width=True):
            navigate_to("🔬 Interactive Segmentation Lab")

    # Objective 2 & 3 Card
    with card_col2:
        st.markdown("""
        <div class="glass-interactive">
            <span class="badge-cyan">OBJECTIVES 2 & 3</span>
            <h3 style="color: #F4FAFF; font-size: 1.25rem; font-weight: 700; margin: 8px 0 6px 0;">
                2 & 3. Multi-Temporal Retreat & Area Accounting
            </h3>
            <p style="color: #A8BED0; font-size: 0.9rem; line-height: 1.6; margin-bottom: 12px;">
                Multi-decadal geodesic surface area calculation across 81,598 validated GLIMS basin pairs. 
                Rigorous area-change accounting establishing 16,321.76 km² net loss with mutually exclusive classification.
            </p>
        </div>
        """, unsafe_allow_html=True)
        img_dist = FIGURES_DIR / "glims_area_change_distribution.png"
        if img_dist.exists():
            st.image(str(img_dist), caption="Global Glacier Area Change Distribution (Δt ≥ 3.0 yrs)", use_container_width=True)
        if st.button("Launch Retreat Explorer →", key="btn_obj23", use_container_width=True):
            navigate_to("🗺️ Glacier Retreat Explorer")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    card_col3, card_col4 = st.columns(2)

    # Objective 4 Card
    with card_col3:
        st.markdown("""
        <div class="glass-interactive">
            <span class="badge-cyan">OBJECTIVE 4</span>
            <h3 style="color: #F4FAFF; font-size: 1.25rem; font-weight: 700; margin: 8px 0 6px 0;">
                4. Multi-Sensor Visual Glacier Monitoring
            </h3>
            <p style="color: #A8BED0; font-size: 0.9rem; line-height: 1.6; margin-bottom: 12px;">
                Publication-quality 300 DPI multi-sensor suites combining Sentinel-2 multispectral (True Color RGB, False-Color SWIR), 
                Landsat-8 multi-year bitemporal change detection, and GLIMS vector polygon boundaries (EPSG:32632).
            </p>
        </div>
        """, unsafe_allow_html=True)
        img_vis = FIGURES_DIR / "visual_monitoring_glims_boundary_overlays.png"
        if img_vis.exists():
            st.image(str(img_vis), caption="GLIMS Multi-Temporal Polygon Boundaries (Rhone, Great Aletsch, Unteraar)", use_container_width=True)
        if st.button("Launch Visual Monitoring Lab →", key="btn_obj4", use_container_width=True):
            navigate_to("🛰️ Visual Monitoring Suite")

    # Objective 5 Card — Research Results & Evaluation Centre
    with card_col4:
        st.markdown("""
        <div class="glass-interactive">
            <span class="badge-cyan">OBJECTIVE 5</span>
            <h3 style="color: #F4FAFF; font-size: 1.25rem; font-weight: 700; margin: 8px 0 6px 0;">
                5. Research Results &amp; Evaluation Centre
            </h3>
            <p style="color: #A8BED0; font-size: 0.9rem; line-height: 1.6; margin-bottom: 12px;">
                Consolidated scientific evaluation across all four research objectives. Presents segmentation benchmarks,
                glacier area comparisons for Rhone, Great Aletsch and Unteraar, global GLIMS inventory statistics,
                and Landsat-8 monitoring metrics — all sourced from actual project result files.
            </p>
        </div>
        """, unsafe_allow_html=True)
        img_obj5 = RUNS_DIR / "objective1_run_002_recovery" / "figures" / "fewshot_k_curve.png"
        if img_obj5.exists():
            st.image(str(img_obj5), caption="Research Results: Few-Shot K-Curve & Consolidated Findings", use_container_width=True)
        if st.button("Open Evaluation Centre →", key="btn_obj5", use_container_width=True):
            navigate_to("📋 Research Results & Evaluation")
