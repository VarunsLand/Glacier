"""GlacierVision Datasets, Scientific Methodology & Project Metadata.

Documents:
- Data source specifications: DL4GAM, GLIMS, Landsat-8, NSIDC.
- Strict data partitioning and zero-leakage protocols.
- System hardware verification (RTX 3050 Laptop GPU, CUDA 13.0, PyTorch 2.14.1+cu130).
- Academic citations and project documentation.
"""
from __future__ import annotations

import streamlit as st

from src.ui.theme import render_header


def render_datasets_about():
    """Render the Datasets, Methodology & About documentation page."""
    render_header(
        title="Datasets, Methodology & Academic Documentation",
        subtitle="Data Source Provenance, Partition Disjointness & Hardware Verification",
        badge="METADATA & DOCUMENTATION",
    )

    t_data, t_method, t_about = st.tabs([
        "💾 Earth Observation Datasets",
        "🔬 Methodology & Leakage Prevention",
        "ℹ️ Platform & Hardware Profile",
    ])

    with t_data:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #12DFF3; margin-top: 0;">1. DL4GAM (Deep Learning for Glacier Area Mapping)</h4>
            <p style="color: #A8BED0; font-size: 0.9rem; line-height: 1.6;">
                Multispectral Sentinel-2 MSI benchmark dataset covering the European Alps.
            </p>
            <ul style="color: #A8BED0; font-size: 0.88rem; line-height: 1.6; padding-left: 20px;">
                <li><b>Total Patches:</b> 2,251 extracted patches (256x256 resolution)</li>
                <li><b>Spectral Channels:</b> 13 bands (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12)</li>
                <li><b>Target Mask:</b> <code>mask_crt_g</code> (clean glacier ice binary segmentation mask)</li>
                <li><b>Benchmark Split:</b> Official <code>split_1</code>: 1,475 train (986 glaciers), 242 val (181 glaciers), 534 test (426 glaciers).</li>
            </ul>
        </div>

        <div class="glass-card">
            <h4 style="color: #12DFF3; margin-top: 0;">2. GLIMS (Global Land Ice Measurements from Space)</h4>
            <p style="color: #A8BED0; font-size: 0.9rem; line-height: 1.6;">
                Worldwide multi-temporal database of satellite-derived glacier outlines.
            </p>
            <ul style="color: #A8BED0; font-size: 0.88rem; line-height: 1.6; padding-left: 20px;">
                <li><b>Inventory Size:</b> 81,598 multi-temporal glacier basins analyzed globally.</li>
                <li><b>Observation Period:</b> 1970 to 2024 (Mean temporal gap: 11.65 years; minimum Δt = 3.0 years).</li>
                <li><b>Geodesic Verification:</b> WGS84 authalic ellipsoidal surface area calculation with regional UTM projections.</li>
            </ul>
        </div>

        <div class="glass-card">
            <h4 style="color: #12DFF3; margin-top: 0;">3. Landsat-8 Operational Land Imager (OLI)</h4>
            <p style="color: #A8BED0; font-size: 0.9rem; line-height: 1.6;">
                30-meter resolution multi-year imagery used for bitemporal surface change monitoring and retreat tracking.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with t_method:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #12DFF3; margin-top: 0;">Zero Data-Leakage Protocol</h4>
            <p style="color: #A8BED0; font-size: 0.9rem; line-height: 1.6;">
                In geospatial computer vision, spatial autocorrelation between adjacent tiles can lead to severe data leakage 
                if patches are partitioned randomly. GlacierVision guarantees:
            </p>
            <ol style="color: #A8BED0; font-size: 0.88rem; line-height: 1.8; padding-left: 20px;">
                <li><b>Glacier ID Partitioning:</b> Partitions are split strictly by <code>glac_id</code>. All patches covering a given glacier are assigned exclusively to train, val, or test.</li>
                <li><b>Support/Query Disjointness:</b> During episodic few-shot training and evaluation, support patches and query patches are sampled from disjoint glaciers.</li>
                <li><b>Normalization Independence:</b> Channel mean and standard deviation are computed strictly on the training partition and frozen for inference.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with t_about:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #12DFF3; margin-top: 0;">Hardware & Environment Profile</h4>
            <ul style="color: #A8BED0; font-size: 0.88rem; line-height: 1.7; padding-left: 20px;">
                <li><b>GPU:</b> NVIDIA GeForce RTX 3050 Laptop GPU (6.00 GB Dedicated VRAM)</li>
                <li><b>CUDA:</b> Version 13.0 | PyTorch 2.14.1+cu130 with FP16 Mixed Precision Support</li>
                <li><b>Operating System:</b> Windows 11 (64-bit)</li>
                <li><b>Web Framework:</b> Streamlit 1.51+ with custom CSS tokens</li>
                <li><b>Code Base:</b> Python 3.13.1</li>
            </ul>
        </div>

        <div class="glass-card">
            <h4 style="color: #12DFF3; margin-top: 0;">Academic Context & Citation</h4>
            <p style="color: #A8BED0; font-size: 0.88rem; line-height: 1.6; margin: 0;">
                GlacierVision: Few-Shot Computer Vision for Automated Glacier Detection and Retreat Monitoring.<br>
                Undergraduate Research Capstone Project in Computer Vision and Remote Sensing, 2026.
            </p>
        </div>
        """, unsafe_allow_html=True)
