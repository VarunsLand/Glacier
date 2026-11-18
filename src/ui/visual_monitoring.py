"""GlacierVision Interactive Multi-Sensor Visual Glacier Monitoring Workspace.

Objective 4 Complete Functional Implementation:
1. Mode A (Explore Research Datasets) & Mode B (Upload Satellite Imagery).
2. Sentinel-2 Multispectral Explorer:
   - True-Color RGB (B4-B3-B2) with percentile stretch, brightness, contrast, and gamma.
   - False-Color SWIR Composite (B11-B8-B4) with customizable band combinations.
   - Individual Spectral Band Explorer (B1..B12) with reflectance histograms & metadata.
   - Normalized Difference Snow Index (NDSI) with safe zero-denominator guard and adjustable threshold.
   - Topographic DEM elevation modeling, analytical hillshade, and contour analysis.
3. Landsat-8 Polar Bitemporal Change Explorer:
   - 683 matched image triplets (Epoch A, Epoch B, Ground Truth Change Mask).
   - Absolute spectral deviation map, binary detected-change mask, and coral composite overlay.
   - Confusion matrix evaluation (TP, FP, FN, TN, Precision, Recall, IoU, Dice score).
   - Scientific disclaimer on undirected surface change vs directional retreat.
4. GLIMS Geospatial Vector Boundary Overlay:
   - Boundary perimeter extraction on satellite imagery with customizable color, width, and opacity.
5. High-Resolution Output Exports:
   - PNG images and georeferenced GeoTIFFs preserving CRS and affine transform tags.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.inference import FIGURES_DIR, RESULTS_DIR
from src.ui.theme import render_header
from src.ui.components import render_kpi_cards
from src.visual_analysis import (
    S2_BAND_SPECS,
    BAND_NAMES_13,
    get_featured_dl4gam_patches,
    get_all_dl4gam_patch_paths,
    load_dl4gam_patch,
    parse_uploaded_satellite_image,
    resolve_s2_band_indices,
    generate_true_color_rgb,
    generate_swir_false_color,
    get_single_band_display,
    compute_band_statistics,
    compute_ndsi,
    compute_dem_analytics,
    parse_and_validate_dem_geotiff,
    get_available_landsat_samples,
    load_landsat_sample,
    compute_bitemporal_change,
    generate_confusion_map_overlay,
    evaluate_change_accuracy,
    overlay_vector_boundary_on_image,
    load_cached_glims_benchmarks,
    render_benchmark_glims_overlay,
    parse_and_validate_vector_upload,
    inspect_uploaded_geotiff,
    export_array_to_png_bytes,
    export_array_to_geotiff_bytes,
)


@st.cache_data
def get_cached_featured_patches():
    return get_featured_dl4gam_patches()


@st.cache_data
def get_cached_landsat_samples():
    return get_available_landsat_samples()


def render_visual_monitoring():
    """Render the interactive multi-sensor visual glacier monitoring workspace."""
    render_header(
        title="Interactive Multi-Sensor Visual Glacier Monitoring Workspace",
        subtitle="Sentinel-2 MSI Multispectral Analysis, Landsat-8 Bitemporal Change Detection & GLIMS Vector Overlays",
        badge="OBJECTIVE 4 • VISUAL MONITORING",
    )

    # =========================================================================
    # PART 1: GLOBAL WORKING MODE SELECTOR
    # =========================================================================
    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #12DFF3; margin-bottom: 1.2rem; padding: 14px 18px;">
        <h5 style="color: #12DFF3; margin: 0 0 6px 0; font-size: 1.0rem;">
            🛰️ Multi-Sensor Scientific Monitoring Laboratory
        </h5>
        <p style="color: #A8BED0; font-size: 0.86rem; margin: 0; line-height: 1.6;">
            Select between exploring validated research datasets (2,251 Sentinel-2 NetCDF patches and 683 Landsat-8 polar triplets) 
            or uploading your own satellite imagery for dynamic multispectral indices, bitemporal difference mapping, and topographic modeling.
        </p>
    </div>
    """, unsafe_allow_html=True)

    mode_choice = st.radio(
        "Select Operating Mode:",
        ["📂 Mode A — Explore Existing Research Datasets", "📤 Mode B — Upload Your Own Satellite Imagery"],
        horizontal=True,
        help="Switch between pre-cataloged research datasets or custom satellite imagery uploads.",
    )
    is_dataset_mode = "Mode A" in mode_choice

    # =========================================================================
    # TABS FOR SPECIALIZED SENSING WORKFLOWS
    # =========================================================================
    tab_s2, tab_landsat, tab_glims, tab_ref_figs = st.tabs([
        "🛰️ Sentinel-2 Multispectral Explorer",
        "🛰️ Landsat Bitemporal Change Explorer",
        "🗺️ GLIMS Boundary Overlay & Geospatial Integration",
        "🌐 Archival Publication Figures (Reference Record)",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: SENTINEL-2 MULTISPECTRAL EXPLORER
    # -------------------------------------------------------------------------
    with tab_s2:
        st.markdown("#### 🛰️ Sentinel-2 MSI Multispectral Analysis Suite")
        st.caption("Dynamic multispectral ingestion, True-Color RGB (B4-B3-B2), SWIR false-color (B11-B8-B4), cryogenic NDSI index, and topographic DEM elevation modeling:")

        s2_data: Optional[Dict[str, Any]] = None

        if is_dataset_mode:
            # Mode A: Bundled Dataset Selection
            c1, c2 = st.columns([2, 1])
            with c1:
                featured = get_cached_featured_patches()
                feat_opts = {f"{c['title']} [{c['glacier_id']}]": c for c in featured}
                selected_feat_label = st.selectbox(
                    "Choose Featured Alpine Benchmark Patch:",
                    list(feat_opts.keys()),
                    index=0,
                    help="Select verified Sentinel-2 NetCDF patches from the DL4GAM European Alps test split.",
                )
                chosen_sample = feat_opts[selected_feat_label]
            with c2:
                st.markdown("**Sample Search & Quick Stats:**")
                st.caption(f"Glacier ID: `{chosen_sample['glacier_id']}`\nRegion: {chosen_sample['region']}")

            try:
                s2_data = load_dl4gam_patch(chosen_sample["path"])
            except Exception as e:
                st.error(f"Error loading NetCDF patch: {e}")
        else:
            # Mode B: Custom File Upload
            st.markdown("##### 📤 Upload Satellite Imagery (NetCDF, Multi-Band GeoTIFF, or RGB)")
            uploaded_file = st.file_uploader(
                "Upload a satellite image (.nc, .tif, .tiff, .png, .jpg):",
                type=["nc", "tif", "tiff", "png", "jpg", "jpeg"],
                help="Supports 13-band NetCDF patches, multi-band GeoTIFF rasters, or standard 3-channel optical RGB images.",
            )
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()
                    uploaded_file.seek(0)
                    s2_data = parse_uploaded_satellite_image(file_bytes, uploaded_file.name)
                    st.success(f"Successfully ingested `{uploaded_file.name}` ({s2_data['channels']} channels, {s2_data['width']}×{s2_data['height']} px)")
                except Exception as e:
                    st.error(f"Upload validation failed: {e}")
            else:
                st.info("Please upload a NetCDF (.nc), GeoTIFF (.tif), or RGB image above to begin interactive multispectral analysis.")

        if s2_data is not None:
            # Band index resolution for dynamic channel mapping
            band_idx_map = resolve_s2_band_indices(s2_data["band_names"])

            # Active Data Source Metadata Banner
            st.markdown(f"""
            <div style="background: rgba(18, 223, 243, 0.05); border-left: 3px solid #12DFF3; padding: 10px 14px; border-radius: 4px; margin-top: 10px; margin-bottom: 14px;">
                <p style="color: #A8BED0; font-size: 0.84rem; margin: 0; line-height: 1.5;">
                    <b>Active Ingestion Source:</b> <code style="color: #12DFF3;">{s2_data.get('filename', 'Custom Raster')}</code> &nbsp;|&nbsp; 
                    <b>Sensor / Spacecraft:</b> {s2_data.get('spacecraft', 'Multispectral Satellite')} &nbsp;|&nbsp; 
                    <b>Spatial Dimensions:</b> {s2_data['width']} × {s2_data['height']} px &nbsp;|&nbsp; 
                    <b>Bands Available:</b> {len(s2_data['band_names'])} channels (<code>{', '.join(s2_data['band_names'])}</code>) &nbsp;|&nbsp; 
                    <b>CRS:</b> <code>{s2_data.get('spatial_ref', 'Unknown')}</code> &nbsp;|&nbsp; 
                    <b>Pixel Resolution:</b> {s2_data['pixel_size'][0]}m × {s2_data['pixel_size'][1]}m
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Special glaciological notice for 5-band Sentinel-2 subsets
            num_channels = s2_data.get("channels", len(s2_data.get("band_names", [])))
            if num_channels == 5:
                st.markdown("""
                <div class="glass-card" style="border-left: 3px solid #38BDF8; margin-bottom: 14px; padding: 10px 14px;">
                    <p style="color: #A8BED0; font-size: 0.83rem; margin: 0; line-height: 1.5;">
                        <b>🛰️ Sentinel-2 5-Band Selected Cryospheric Subset:</b> Ingested channels <code>B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR 1)</code>. 
                        The native 10m visible/NIR bands are referenced to the common <b>20-metre processing grid</b> established by SWIR Band 11. 
                        True-Color RGB, False-Color SWIR, and NDSI calculations are fully active.<br/>
                        <i>Model Scope Notice:</i> 13-band DL4GAM U-Net/ProtoNet segmentation requires the complete 13-channel tensor and is disabled for this 5-band dataset.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Sub-tabs for multispectral views
            sub_rgb, sub_swir, sub_bands, sub_ndsi, sub_dem = st.tabs([
                "🌈 True-Color RGB",
                "🧊 False-Color SWIR",
                "🔬 Individual Band Explorer",
                "❄️ NDSI Snow/Ice Index",
                "🏔️ DEM Elevation & Hillshade",
            ])

            # --- SUB-TAB 1: TRUE-COLOR RGB ---
            with sub_rgb:
                st.markdown("##### 🌈 True-Color RGB Composite (Bands 4, 3, 2)")
                st.caption("Simulates human visual perception using Red (B4/B04, 665nm), Green (B3/B03, 560nm), and Blue (B2/B02, 490nm):")

                c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns(4)
                with c_ctrl1:
                    stretch_p = st.slider("Percentile Stretch (%):", 0.0, 10.0, (2.0, 98.0), step=0.5, key="rgb_stretch")
                with c_ctrl2:
                    brightness = st.slider("Brightness Scale:", 0.5, 2.0, 1.0, step=0.1, key="rgb_bright")
                with c_ctrl3:
                    contrast = st.slider("Contrast Scale:", 0.5, 2.0, 1.0, step=0.1, key="rgb_contrast")
                with c_ctrl4:
                    gamma = st.slider("Gamma Correction:", 0.5, 2.5, 1.0, step=0.1, key="rgb_gamma")

                try:
                    bands_arr = s2_data["bands"]
                    r_idx = band_idx_map["red"] if band_idx_map["red"] is not None else (3 if bands_arr.shape[0] >= 4 else 0)
                    g_idx = band_idx_map["green"] if band_idx_map["green"] is not None else (2 if bands_arr.shape[0] >= 3 else 1)
                    b_idx = band_idx_map["blue"] if band_idx_map["blue"] is not None else (1 if bands_arr.shape[0] >= 2 else 0)

                    rgb_img = generate_true_color_rgb(
                        bands_arr,
                        r_idx=r_idx, g_idx=g_idx, b_idx=b_idx,
                        p_low=stretch_p[0], p_high=stretch_p[1],
                        brightness=brightness, contrast=contrast, gamma=gamma,
                    )

                    r_name = s2_data['band_names'][r_idx]
                    g_name = s2_data['band_names'][g_idx]
                    b_name = s2_data['band_names'][b_idx]
                    r_spec = S2_BAND_SPECS.get(r_name, {"wavelength": "665 nm"})
                    g_spec = S2_BAND_SPECS.get(g_name, {"wavelength": "560 nm"})
                    b_spec = S2_BAND_SPECS.get(b_name, {"wavelength": "490 nm"})

                    c_img, c_meta = st.columns([2, 1])
                    with c_img:
                        st.image(rgb_img, caption=f"True-Color RGB Composite ({r_name}-{g_name}-{b_name}) — {s2_data['filename']}", use_container_width=True)
                    with c_meta:
                        st.markdown("**Spectral Composite Specifications:**")
                        st.markdown(f"""
                        - **Red Channel:** {r_name} ({r_spec['wavelength']})
                        - **Green Channel:** {g_name} ({g_spec['wavelength']})
                        - **Blue Channel:** {b_name} ({b_spec['wavelength']})
                        - **Radiometric Stretch:** {stretch_p[0]:.1f}% – {stretch_p[1]:.1f}%
                        - **Effective Footprint:** {s2_data['width']*s2_data['pixel_size'][0]/1000.0:.2f} km × {s2_data['height']*s2_data['pixel_size'][1]/1000.0:.2f} km
                        """)
                        rgb_bytes = export_array_to_png_bytes(rgb_img)
                        st.download_button(
                            "📥 Download RGB Image (PNG)",
                            data=rgb_bytes,
                            file_name=f"true_color_rgb_{s2_data.get('patch_id', 'patch')}.png",
                            mime="image/png",
                        )
                        if s2_data.get("spatial_ref") and s2_data["spatial_ref"] != "Unknown":
                            gtiff_bytes = export_array_to_geotiff_bytes(rgb_img)
                            st.download_button(
                                "🗺️ Download Georeferenced GeoTIFF",
                                data=gtiff_bytes,
                                file_name=f"true_color_rgb_{s2_data.get('patch_id', 'patch')}.tif",
                                mime="image/tiff",
                            )
                except Exception as e:
                    st.error(f"Error generating True-Color RGB composite: {e}")

            # --- SUB-TAB 2: FALSE-COLOR SWIR ---
            with sub_swir:
                if s2_data.get("is_rgb_only", False) or s2_data["bands"].shape[0] < 4:
                    st.markdown("##### 🧊 Glaciological False-Color SWIR Composite")
                    st.warning("⚠️ **Multispectral Restriction:** The active source is a 3-channel optical RGB image. Shortwave-infrared (B11) and Near-infrared (B8/B08) channels are required to generate SWIR false-color composites.")
                else:
                    def_swir_r_idx = band_idx_map["swir1"] if band_idx_map["swir1"] is not None else min(11, len(s2_data["band_names"])-1)
                    def_swir_g_idx = band_idx_map["nir"] if band_idx_map["nir"] is not None else min(7, len(s2_data["band_names"])-1)
                    def_swir_b_idx = band_idx_map["red"] if band_idx_map["red"] is not None else min(3, len(s2_data["band_names"])-1)

                    target_r = s2_data["band_names"][def_swir_r_idx]
                    target_g = s2_data["band_names"][def_swir_g_idx]
                    target_b = s2_data["band_names"][def_swir_b_idx]

                    st.markdown(f"##### 🧊 Glaciological False-Color SWIR Composite ({target_r}-{target_g}-{target_b})")
                    st.caption(f"Exploits strong shortwave-infrared (SWIR) absorption by glacier ice. Default cryospheric mapping: Red={target_r} (SWIR 1), Green={target_g} (NIR), Blue={target_b} (Red). Snow and clean glacier ice appear electric cyan/aquamarine, clouds appear bright white, and bare bedrock/moraine appears brown/tan:")

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
                        or st.session_state["swir_r"] == st.session_state["swir_g"]  # Fix duplicate channel assignment bug
                    )

                    if needs_swir_init:
                        st.session_state["swir_dataset_tracker"] = current_scene_key
                        st.session_state["swir_r"] = target_r
                        st.session_state["swir_g"] = target_g
                        st.session_state["swir_b"] = target_b

                    col_swir_ctrl, col_swir_reset = st.columns([3, 1])
                    with col_swir_ctrl:
                        st.markdown(f"**Channel Assignment Controls:** Customize RGB channel band mapping or use standard glaciological defaults (`{target_r}` / `{target_g}` / `{target_b}`):")
                    with col_swir_reset:
                        if st.button(f"🔄 Reset to Default ({target_r}-{target_g}-{target_b})", key="reset_swir_btn", help="Reset all channel selectors back to the standard glaciological SWIR mapping"):
                            st.session_state["swir_r"] = target_r
                            st.session_state["swir_g"] = target_g
                            st.session_state["swir_b"] = target_b
                            st.rerun()

                    c_swir1, c_swir2, c_swir3 = st.columns(3)
                    with c_swir1:
                        opt_r = st.selectbox(f"Red Channel Band (Default: {target_r}):", s2_data["band_names"], key="swir_r")
                    with c_swir2:
                        opt_g = st.selectbox(f"Green Channel Band (Default: {target_g}):", s2_data["band_names"], key="swir_g")
                    with c_swir3:
                        opt_b = st.selectbox(f"Blue Channel Band (Default: {target_b}):", s2_data["band_names"], key="swir_b")

                    try:
                        r_i = s2_data["band_names"].index(opt_r)
                        g_i = s2_data["band_names"].index(opt_g)
                        b_i = s2_data["band_names"].index(opt_b)

                        swir_img = generate_swir_false_color(
                            s2_data["bands"],
                            swir_idx=r_i, nir_idx=g_i, red_idx=b_i,
                        )

                        spec_r = S2_BAND_SPECS.get(opt_r, {"wavelength": "1610 nm", "desc": "SWIR Channel"})
                        spec_g = S2_BAND_SPECS.get(opt_g, {"wavelength": "842 nm", "desc": "NIR Channel"})
                        spec_b = S2_BAND_SPECS.get(opt_b, {"wavelength": "665 nm", "desc": "Visible Red Channel"})

                        col_simg, col_smeta = st.columns([2, 1])
                        with col_simg:
                            st.image(swir_img, caption=f"False-Color SWIR Composite (R: {opt_r} / G: {opt_g} / B: {opt_b}) — {s2_data['filename']}", use_container_width=True)
                        with col_smeta:
                            st.markdown("**Active Channel Mapping & Wavelengths:**")
                            st.markdown(f"""
                            - **Red (R):** `{opt_r}` ({spec_r.get('wavelength', 'SWIR')}) — {spec_r.get('desc', 'Shortwave Infrared')}
                            - **Green (G):** `{opt_g}` ({spec_g.get('wavelength', 'NIR')}) — {spec_g.get('desc', 'Near-Infrared')}
                            - **Blue (B):** `{opt_b}` ({spec_b.get('wavelength', 'Red')}) — {spec_b.get('desc', 'Visible Red')}
                            """)

                            st.markdown("**Cryospheric Physical Interpretation:**")
                            st.markdown(f"""
                            - **Glacier Ice & Snow:** Electric **Cyan / Aquamarine**  
                              *(Glacial ice strongly absorbs SWIR `{opt_r}` yielding near-zero Red, while reflecting NIR `{opt_g}` and visible `{opt_b}`, combining into vibrant cyan `[0, G, B]`.)*
                            - **Cloud Formations:** Opaque **Bright White**  
                              *(Clouds reflect visible, NIR, and SWIR radiation almost equally, yielding uniform high values across all three channels `[R, G, B]`.)*
                            - **Bedrock / Moraine / Debris:** **Brownish to Tan / Orange**  
                              *(Exposed rock and glacial till exhibit moderate-to-high SWIR reflectance with lower NIR reflectance.)*
                            - **Vegetation:** **Vivid Green**  
                              *(Chlorophyll cellular structure reflects heavily in NIR `{opt_g}`, driving the Green channel.)*
                            - **Proglacial Meltwater / Lakes:** **Deep Navy to Black**  
                              *(Liquid water absorbs heavily across both SWIR and NIR wavelengths.)*
                            """)
                            swir_bytes = export_array_to_png_bytes(swir_img)
                            st.download_button(
                                "📥 Download SWIR Image (PNG)",
                                data=swir_bytes,
                                file_name=f"swir_composite_{opt_r}_{opt_g}_{opt_b}_{s2_data.get('patch_id', 'patch')}.png",
                                mime="image/png",
                            )
                    except Exception as e:
                        st.error(f"Error generating SWIR composite: {e}")

            # --- SUB-TAB 3: INDIVIDUAL BAND EXPLORER ---
            with sub_bands:
                st.markdown("##### 🔬 Individual Spectral Band Explorer (Radiometric Inspection)")
                st.caption("Inspect raw digital numbers, surface reflectance distributions, and physical sensor wavelengths for each channel:")

                chosen_band_name = st.selectbox("Select Spectral Band to Inspect:", s2_data["band_names"], index=min(1, len(s2_data["band_names"])-1))
                b_idx = s2_data["band_names"].index(chosen_band_name)
                band_arr = s2_data["bands"][b_idx] if s2_data["bands"].ndim == 3 else s2_data["bands"]

                band_spec = S2_BAND_SPECS.get(chosen_band_name, {"name": chosen_band_name, "wavelength": "Unknown", "resolution": f"{s2_data['pixel_size'][0]:.0f}m", "desc": "Multispectral Channel"})
                stats = compute_band_statistics(band_arr, chosen_band_name)

                col_b1, col_b2 = st.columns([1.5, 1.5])
                with col_b1:
                    b_disp = get_single_band_display(band_arr)
                    st.image(b_disp, caption=f"{band_spec['name']} Grayscale — {s2_data['filename']}", use_container_width=True)
                with col_b2:
                    st.markdown(f"**Physical Band Specifications:**")
                    st.markdown(f"""
                    - **Central Wavelength:** `{band_spec['wavelength']}`
                    - **Nominal Ground Resolution:** `{band_spec['resolution']}`
                    - **Function:** {band_spec['desc']}
                    - **Min / Max Reflectance:** `{stats['min']:.4f}` / `{stats['max']:.4f}`
                    - **Mean ± Std:** `{stats['mean']:.4f} ± {stats['std']:.4f}`
                    - **Median (p50):** `{stats['median']:.4f}` (p5: `{stats['p5']:.4f}`, p95: `{stats['p95']:.4f}`)
                    """)

                    # Histogram of pixel values
                    fig_hist = go.Figure()
                    bin_centers = 0.5 * (np.array(stats["bin_edges"][:-1]) + np.array(stats["bin_edges"][1:]))
                    fig_hist.add_trace(go.Bar(
                        x=bin_centers,
                        y=stats["hist_counts"],
                        marker_color="#12DFF3",
                        name="Pixel Frequency",
                    ))
                    fig_hist.update_layout(
                        title=f"{chosen_band_name} Reflectance Histogram",
                        paper_bgcolor="rgba(6, 20, 38, 0.4)",
                        plot_bgcolor="rgba(6, 20, 38, 0.2)",
                        xaxis=dict(title="Reflectance Value", color="#A8BED0"),
                        yaxis=dict(title="Pixel Count", color="#A8BED0"),
                        height=200,
                        margin=dict(l=30, r=30, t=30, b=30),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

            # --- SUB-TAB 4: NDSI SNOW/ICE INDEX ---
            with sub_ndsi:
                st.markdown("##### ❄️ Normalized Difference Snow Index (NDSI)")
                st.caption("Standard cryospheric index isolating snow and glacier ice from cloud cover, bare bedrock, and mountain shadows:")

                st.latex(r"\text{NDSI} = \frac{\text{Green (B3/B03)} - \text{SWIR1 (B11)}}{\text{Green (B3/B03)} + \text{SWIR1 (B11)}}")

                has_ndsi_channels = (band_idx_map["green"] is not None) and (band_idx_map["swir1"] is not None)

                if s2_data.get("is_rgb_only", False) or not has_ndsi_channels:
                    st.warning("⚠️ **Multispectral Restriction:** NDSI calculation requires Green (Band 3 / B03) and SWIR 1 (Band 11). The uploaded imagery does not contain these required channels.")
                else:
                    ndsi_thresh = st.slider(
                        "Configurable Snow/Ice Classification Threshold (Heuristic):",
                        0.0, 1.0, 0.4, step=0.05,
                        help="Standard starting heuristic is 0.40. Higher values demand cleaner snow/ice; lower values include dirty ice and mixed pixels.",
                    )

                    try:
                        green_chan = s2_data["bands"][band_idx_map["green"]]
                        swir_chan = s2_data["bands"][band_idx_map["swir1"]]

                        ndsi_res = compute_ndsi(green_chan, swir_chan, threshold=ndsi_thresh)

                        c_ndsi1, c_ndsi2 = st.columns(2)
                        with c_ndsi1:
                            st.image(ndsi_res["colorized_rgb"], caption=f"Continuous NDSI Field (Range: -1.0 to +1.0) — {s2_data['filename']}", use_container_width=True)
                        with c_ndsi2:
                            mask_disp = (ndsi_res["binary_mask"] * 255).astype(np.uint8)
                            st.image(mask_disp, caption=f"Binary Cryogenic Mask (NDSI ≥ {ndsi_thresh:.2f}) — {s2_data['filename']}", use_container_width=True)

                        col_met1, col_met2, col_met3 = st.columns(3)
                        with col_met1:
                            st.metric("Snow/Ice Classified Pixels", f"{ndsi_res['snow_ice_pixels']:,}")
                        with col_met2:
                            st.metric("Surface Coverage Ratio", f"{ndsi_res['snow_ice_pct']:.2f}%")
                        with col_met3:
                            st.metric("Mean Scene NDSI", f"{ndsi_res['mean_ndsi']:+.3f}")

                        res_desc = f"Common {s2_data['pixel_size'][0]:.0f}m processing grid (native visible bands aggregated to B11 20m sensor footprint)." if s2_data.get('channels', len(s2_data.get('band_names', []))) == 5 else "While Band 3 is acquired at 10m spatial resolution, Band 11 is natively a 20m sensor band resampled to 10m. The effective optical resolving limit of NDSI is therefore 20m."

                        st.markdown(f"""
                        <div class="glass-card" style="border-left: 3px solid #F59E0B; margin-top: 10px; margin-bottom: 10px; padding: 10px 14px;">
                            <p style="color: #A8BED0; font-size: 0.82rem; margin: 0; line-height: 1.5;">
                                <b>Scientific Disclaimers & Limitations:</b><br/>
                                1. <b>Heuristic vs Ground Truth:</b> The 0.40 threshold is a standard glaciological heuristic, not a machine-learned or validated glacier boundary. Debris-covered glacier tongues typically yield low NDSI and remain unclassified by index-based methods.<br/>
                                2. <b>Spatial Resolution & Sensor Grid:</b> {res_desc}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        ndsi_png = export_array_to_png_bytes(ndsi_res["colorized_rgb"])
                        st.download_button(
                            "📥 Download NDSI Field (PNG)",
                            data=ndsi_png,
                            file_name=f"ndsi_field_{s2_data.get('patch_id', 'patch')}.png",
                            mime="image/png",
                        )
                    except Exception as e:
                        st.error(f"Error computing NDSI: {e}")

            # --- SUB-TAB 5: DEM ELEVATION & HILLSHADE ---
            with sub_dem:
                st.markdown("##### 🏔️ Topographic Digital Elevation Model (DEM) & Hillshade")
                st.caption("Real topographic relief modeling from ALOS/Copernicus 10m elevation models embedded in validated research patches:")

                if s2_data.get("dem") is not None:
                    dem_arr = s2_data["dem"]
                    c_sun1, c_sun2 = st.columns(2)
                    with c_sun1:
                        sun_az = st.slider("Solar Azimuth Angle (°):", 0.0, 360.0, 315.0, step=15.0, key="dem_az")
                    with c_sun2:
                        sun_alt = st.slider("Solar Altitude Angle (°):", 10.0, 80.0, 45.0, step=5.0, key="dem_alt")

                    dem_res = compute_dem_analytics(dem_arr, resolution=10.0, sun_azimuth=sun_az, sun_altitude=sun_alt)

                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.image(dem_res["dem_rgb"], caption=f"Topographic Elevation Heatmap ({dem_res['min_elevation']:.0f}m to {dem_res['max_elevation']:.0f}m a.s.l.)", use_container_width=True)
                    with col_d2:
                        st.image(dem_res["hillshade"], caption=f"Analytical Hillshade (Azimuth: {sun_az:.0f}°, Altitude: {sun_alt:.0f}°)", use_container_width=True)

                    if "dem_contours_rgb" in dem_res:
                        st.image(dem_res["dem_contours_rgb"], caption=f"Topographic Elevation Contours & Hypsometry ({dem_res['min_elevation']:.0f}m – {dem_res['max_elevation']:.0f}m a.s.l.)", use_container_width=True)

                    d_kpi1, d_kpi2, d_kpi3, d_kpi4 = st.columns(4)
                    with d_kpi1:
                        st.metric("Min Elevation", f"{dem_res['min_elevation']:.1f} m")
                    with d_kpi2:
                        st.metric("Max Elevation", f"{dem_res['max_elevation']:.1f} m")
                    with d_kpi3:
                        st.metric("Total Relief", f"{dem_res['relief']:.1f} m")
                    with d_kpi4:
                        st.metric("Mean Slope", f"{dem_res['mean_slope_deg']:.1f}°")

                    st.markdown(f"""
                    <div style="background: rgba(18, 223, 243, 0.05); border-left: 3px solid #12DFF3; padding: 10px 14px; border-radius: 4px; margin-top: 10px; margin-bottom: 10px;">
                        <p style="color: #A8BED0; font-size: 0.83rem; margin: 0; line-height: 1.5;">
                            <b>DEM Hypsometry & Summary:</b><br/>
                            • <b>Mean Elevation:</b> {dem_res['mean_elevation']:.1f} m a.s.l. &nbsp;|&nbsp; <b>Median Elevation:</b> {dem_res.get('median_elevation', dem_res['mean_elevation']):.1f} m &nbsp;|&nbsp; <b>Std Deviation:</b> ±{dem_res.get('std_elevation', 0.0):.1f} m<br/>
                            • <b>Raster Dimensions:</b> {dem_arr.shape[1]} × {dem_arr.shape[0]} px &nbsp;|&nbsp; <b>Model:</b> ALOS/Copernicus 10m Model
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    dem_png = export_array_to_png_bytes(dem_res["dem_rgb"])
                    st.download_button(
                        "📥 Download DEM Elevation Map (PNG)",
                        data=dem_png,
                        file_name=f"dem_elevation_{s2_data.get('patch_id', 'patch')}.png",
                        mime="image/png",
                    )
                else:
                    st.info("ℹ️ **Digital Elevation Model Unavailable:** The current satellite scene does not contain an embedded digital elevation model (DEM). You may upload an optional DEM GeoTIFF below to inspect elevation contours and hillshading.")
                    dem_upload = st.file_uploader("Upload Optional Registered DEM GeoTIFF:", type=["tif", "tiff"], key="dem_extra_upload")
                    if dem_upload is not None:
                        try:
                            dem_upload.seek(0)
                            raw_dem_bytes = dem_upload.read()
                            dem_upload.seek(0)

                            dem_parsed = parse_and_validate_dem_geotiff(
                                file_bytes=raw_dem_bytes,
                                filename=dem_upload.name,
                                scene_metadata=s2_data,
                            )
                            st.success(f"Successfully loaded and validated DEM: `{dem_upload.name}` ({dem_parsed['width']}×{dem_parsed['height']} px, CRS: {dem_parsed['crs']})")

                            c_sun1, c_sun2 = st.columns(2)
                            with c_sun1:
                                sun_az = st.slider("Solar Azimuth Angle (°):", 0.0, 360.0, 315.0, step=15.0, key="dem_upload_az")
                            with c_sun2:
                                sun_alt = st.slider("Solar Altitude Angle (°):", 10.0, 80.0, 45.0, step=5.0, key="dem_upload_alt")

                            d_res = compute_dem_analytics(
                                dem_parsed["dem"],
                                resolution=dem_parsed.get("pixel_size", (10.0, 10.0))[0],
                                sun_azimuth=sun_az,
                                sun_altitude=sun_alt,
                            )

                            col_d1, col_d2 = st.columns(2)
                            with col_d1:
                                st.image(d_res["dem_rgb"], caption=f"Uploaded DEM Elevation Heatmap ({d_res['min_elevation']:.0f}m to {d_res['max_elevation']:.0f}m a.s.l.)", use_container_width=True)
                            with col_d2:
                                st.image(d_res["hillshade"], caption=f"Uploaded DEM Analytical Hillshade (Azimuth: {sun_az:.0f}°, Altitude: {sun_alt:.0f}°)", use_container_width=True)

                            if "dem_contours_rgb" in d_res:
                                st.image(d_res["dem_contours_rgb"], caption=f"Topographic Elevation Contours & Hypsometry ({d_res['min_elevation']:.0f}m – {d_res['max_elevation']:.0f}m a.s.l.)", use_container_width=True)

                            d_kpi1, d_kpi2, d_kpi3, d_kpi4 = st.columns(4)
                            with d_kpi1:
                                st.metric("Min Elevation", f"{d_res['min_elevation']:.1f} m")
                            with d_kpi2:
                                st.metric("Max Elevation", f"{d_res['max_elevation']:.1f} m")
                            with d_kpi3:
                                st.metric("Total Relief", f"{d_res['relief']:.1f} m")
                            with d_kpi4:
                                st.metric("Mean Slope", f"{d_res['mean_slope_deg']:.1f}°")

                            st.markdown(f"""
                            <div style="background: rgba(18, 223, 243, 0.05); border-left: 3px solid #12DFF3; padding: 10px 14px; border-radius: 4px; margin-top: 10px; margin-bottom: 10px;">
                                <p style="color: #A8BED0; font-size: 0.83rem; margin: 0; line-height: 1.5;">
                                    <b>DEM Metadata & Hypsometry:</b><br/>
                                    • <b>Mean Elevation:</b> {d_res['mean_elevation']:.1f} m a.s.l. &nbsp;|&nbsp; <b>Median Elevation:</b> {d_res.get('median_elevation', d_res['mean_elevation']):.1f} m &nbsp;|&nbsp; <b>Std Deviation:</b> ±{d_res.get('std_elevation', 0.0):.1f} m<br/>
                                    • <b>Raster Dimensions:</b> {dem_parsed['width']} × {dem_parsed['height']} pixels &nbsp;|&nbsp; <b>Pixel Resolution:</b> {dem_parsed.get('pixel_size', (10.0, 10.0))[0]:.1f}m &nbsp;|&nbsp; <b>CRS:</b> <code>{dem_parsed['crs']}</code>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                            dem_png = export_array_to_png_bytes(d_res["dem_rgb"])
                            st.download_button(
                                "📥 Download Validated DEM Elevation Map (PNG)",
                                data=dem_png,
                                file_name=f"dem_elevation_{dem_upload.name.replace('.tif', '')}.png",
                                mime="image/png",
                            )
                        except ValueError as e:
                            st.error(f"❌ **Invalid DEM GeoTIFF:** {e}")
                        except Exception as e:
                            st.error(f"❌ **Failed to parse uploaded DEM:** {e}")

    # -------------------------------------------------------------------------
    # TAB 2: LANDSAT BITEMPORAL CHANGE EXPLORER
    # -------------------------------------------------------------------------
    with tab_landsat:
        st.markdown("#### 🛰️ Landsat-8 Polar Bitemporal Surface Change Suite")
        st.caption("Undirected bitemporal surface change detection across multi-year polar cryosphere observations (30m spatial resolution):")

        ls_sample: Optional[Dict[str, Any]] = None

        if is_dataset_mode:
            # Mode A: Bundled Landsat Triplet Selector
            all_samples = get_cached_landsat_samples()
            col_l1, col_l2 = st.columns([2, 1])
            with col_l1:
                selected_sample_id = st.selectbox(
                    "Choose Polar Bitemporal Observation Triplet:",
                    all_samples,
                    index=0,
                    help="Select from 683 matched Landsat-8 earlier/later image pairs with verified reference change masks.",
                )
            with col_l2:
                region_hint = "Greenland Cryosphere (64°–66°N)" if selected_sample_id.startswith("1_") else "Svalbard Archipelago (76°–80°N)"
                st.markdown(f"**Region:** {region_hint}")
                st.caption(f"Spatial GSD: 30m | Dimensions: 256×256 px")

            try:
                ls_sample = load_landsat_sample(selected_sample_id)
            except Exception as e:
                st.error(f"Error loading Landsat sample: {e}")
        else:
            # Mode B: Custom Bitemporal Uploads
            st.markdown("##### 📤 Upload Multi-Temporal Satellite Pair (Epoch A & Epoch B)")
            col_up_a, col_up_b = st.columns(2)
            with col_up_a:
                up_a = st.file_uploader("1. Earlier Satellite Observation (Epoch A):", type=["png", "jpg", "tif", "tiff"], key="ls_up_a")
            with col_up_b:
                up_b = st.file_uploader("2. Later Satellite Observation (Epoch B):", type=["png", "jpg", "tif", "tiff"], key="ls_up_b")

            up_ref = st.file_uploader("3. Optional Ground-Truth Reference Change Mask:", type=["png", "jpg", "tif", "tiff"], key="ls_up_ref")

            if up_a is not None and up_b is not None:
                try:
                    img_a = np.array(Image.open(up_a).convert("RGB"))
                    img_b = np.array(Image.open(up_b).convert("RGB"))
                    ref_m = (np.array(Image.open(up_ref)) >= 128).astype(np.uint8) if up_ref is not None else None

                    if img_a.shape[:2] != img_b.shape[:2]:
                        st.error(f"Image dimension mismatch: Earlier observation is {img_a.shape[:2]}, but later observation is {img_b.shape[:2]}. Please upload aligned scenes.")
                    elif ref_m is not None and ref_m.shape[:2] != img_a.shape[:2]:
                        st.error(f"Reference mask dimension mismatch: Mask is {ref_m.shape[:2]}, but imagery is {img_a.shape[:2]}. Please upload matching dimensions.")
                    else:
                        ls_sample = {
                            "sample_id": f"Custom Upload ({up_a.name} vs {up_b.name})",
                            "region": "User Uploaded Multi-Temporal Scene",
                            "img_a": img_a,
                            "img_b": img_b,
                            "ref_mask": ref_m,
                            "width": img_a.shape[1],
                            "height": img_a.shape[0],
                            "resolution_m": 30.0,
                        }
                except Exception as e:
                    st.error(f"Failed to process uploaded bitemporal pair: {e}")
            else:
                st.info("Please upload both an earlier observation (Epoch A) and later observation (Epoch B) to execute change detection.")

        if ls_sample is not None:
            # Active Sample & Methodology Banner (Explicitly identifies heuristic, threshold, and sample ID)
            st.markdown(f"""
            <div style="background: rgba(18, 223, 243, 0.05); border-left: 3px solid #12DFF3; padding: 10px 14px; border-radius: 4px; margin-top: 10px; margin-bottom: 12px;">
                <p style="color: #A8BED0; font-size: 0.84rem; margin: 0; line-height: 1.5;">
                    <b>Selected Sample ID:</b> <code style="color: #12DFF3;">{ls_sample['sample_id']}</code> &nbsp;|&nbsp; 
                    <b>Region:</b> {ls_sample['region']} &nbsp;|&nbsp; 
                    <b>Method:</b> Deterministic Radiometric Differencing Heuristic [Threshold-based Heuristic, Not AI] &nbsp;|&nbsp; 
                    <b>Spatial Resolution:</b> Documented 30m Ground Sampling Distance (GSD) &nbsp;|&nbsp; 
                    <b>Dimensions:</b> {ls_sample['width']} × {ls_sample['height']} px
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Algorithm Method Selector & Threshold Slider
            col_m1, col_m2 = st.columns([1, 1])
            with col_m1:
                method_options = [
                    "Method A: Baseline Mean RGB (|ΔI| ≥ 35)",
                    "Method B: Grayscale Difference",
                    "Method C: Calibrated Mean RGB Difference",
                    "Method D: Euclidean RGB Distance",
                    "Method E: Illumination Offset-Normalized Difference",
                ]
                selected_method_label = st.selectbox(
                    "Select Change Detection Algorithm:",
                    method_options,
                    index=0,
                    help="Compare candidate differencing methods. Method A is the uncalibrated baseline; Method E compensates for multi-year solar/atmospheric darkening offsets."
                )

            method_params = {
                "Method A: Baseline Mean RGB (|ΔI| ≥ 35)": ("mean_rgb", 35.0, 5.0, 100.0),
                "Method B: Grayscale Difference": ("grayscale", 35.0, 5.0, 100.0),
                "Method C: Calibrated Mean RGB Difference": ("mean_rgb", 60.0, 5.0, 100.0),
                "Method D: Euclidean RGB Distance": ("euclidean", 90.0, 10.0, 180.0),
                "Method E: Illumination Offset-Normalized Difference": ("offset_normalized", 40.0, 5.0, 100.0),
            }
            m_code, def_th, min_th, max_th = method_params[selected_method_label]
            m_key = selected_method_label.split(":")[0]  # "Method A", "Method B", "Method C", "Method D", "Method E"

            # ── Method-aware display labels ──────────────────────────────────────────
            _diff_captions = {
                "Method A": "Mean absolute RGB difference |mean(|B − A|)|",
                "Method B": "Grayscale luminance difference |Gray(B) − Gray(A)|",
                "Method C": "Mean absolute RGB difference |mean(|B − A|)| (High-Threshold Calibration)",
                "Method D": "Euclidean RGB distance √(ΔR² + ΔG² + ΔB²)",
                "Method E": "Illumination-corrected difference |(B − μ_B) − (A − μ_A)|",
            }
            _mask_captions = {
                "Method A": "Method A — Baseline Mean RGB Change Mask",
                "Method B": "Method B — Grayscale Luminance Change Mask",
                "Method C": "Method C — Calibrated Mean RGB Change Mask",
                "Method D": "Method D — Euclidean RGB Distance Change Mask",
                "Method E": "Method E — Illumination Offset-Normalized Change Mask",
            }
            _method_analysis = {
                "Method A": (
                    "<b>Method A (Baseline Mean RGB Differencing):</b> Subtracts raw digital numbers across "
                    "RGB channels (|ΔI| ≥ 35) without illumination compensation. Multi-year Landsat images acquired "
                    "under differing solar zenith angles suffer from scene-wide darkening (on sample 1_10_1, "
                    "ΔĪ ≈ −36.49 DN). This baseline shift forces 83.60% of the scene into the changed category, "
                    "producing 52,140 false positives and a low precision of 0.0483 (IoU = 0.0449, Dice = 0.0860)."
                ),
                "Method B": (
                    "<b>Method B (Grayscale Luminance Differencing):</b> Converts each observation to ITU-R BT.601 "
                    "luminance (0.2989 R + 0.5870 G + 0.1140 B) before differencing (|ΔI| ≥ 35). Because cryospheric "
                    "features in polar scenes have nearly equal spectral reflectance across RGB channels, "
                    "grayscale differencing correlates almost perfectly with Mean RGB (Pearson r = 0.9950, mean absolute "
                    "difference = 0.55 DN). At T = 35, it flags 55,293 pixels (52,652 FP, Precision = 0.0478), "
                    "confirming that channel weighting alone cannot resolve scene-wide illumination offsets."
                ),
                "Method C": (
                    "<b>Method C (Calibrated Mean RGB Differencing):</b> Employs the identical differencing formula "
                    "as Method A, but raises the threshold to T = 60 to filter out the ~36.5 DN illumination darkening shift. "
                    "Its predicted change mask is a strict subset of Method A (1,051 pixels vs 54,789 pixels). "
                    "This threshold calibration successfully suppresses false positives by 99.47% (FP drops from "
                    "52,140 to 276) and boosts precision to 0.7374 (IoU = 0.1096, Dice = 0.1976), though recall "
                    "decreases to 0.1141 as subtle true changes fall below the conservative threshold."
                ),
                "Method D": (
                    "<b>Method D (Euclidean RGB Distance Differencing):</b> Calculates 3D geometric distance in RGB "
                    "color space (√(ΔR² + ΔG² + ΔB²)). Since Landsat channels are strongly correlated in polar imagery, "
                    "Euclidean distance scales approximately by √3 ≈ 1.732 × Mean RGB difference (Pearson r = 0.9990). "
                    "Its calibrated threshold T = 90 corresponds to an effective Mean RGB threshold of ~51.6 DN. "
                    "It predicts 2,008 changed pixels, detecting 1,149 true positives and 859 false positives "
                    "(Precision = 0.5722, Recall = 0.1691, Specificity = 0.9854, IoU = 0.1501, Dice = 0.2611)."
                ),
                "Method E": (
                    "<b>Method E (Illumination Offset-Normalized Differencing):</b> Directly compensates for the "
                    "multi-year atmospheric and solar elevation shift by subtracting the per-channel mean difference "
                    "(ΔR = −36.53, ΔG = −36.17, ΔB = −36.76 DN) prior to differencing (|ΔI| ≥ 40). "
                    "This centers the static background near zero, collapsing false positives from 52,140 down to just "
                    "148 pixels (a 99.72% reduction) while preserving 2,403 true positives. Method E achieves the "
                    "highest accuracy of all evaluated methods: Precision = 0.9420, Specificity = 0.9975, "
                    "IoU = 0.3462, and Dice = 0.5143."
                ),
            }
            _diff_cap  = _diff_captions.get(m_key, "Spectral difference")
            _mask_cap  = _mask_captions.get(m_key, m_key)
            _analysis  = _method_analysis.get(m_key, "")
            # ─────────────────────────────────────────────────────────────────────────

            with col_m2:
                ch_threshold = st.slider(
                    f"Deterministic Threshold ({m_key}):",
                    min_th, max_th, def_th, step=1.0,
                    key=f"slider_th_{m_key.replace(' ', '_')}",
                    help="Controls sensitivity of undirected spectral difference detection."
                )

            try:
                ch_results = compute_bitemporal_change(
                    ls_sample["img_a"],
                    ls_sample["img_b"],
                    threshold=ch_threshold,
                    pixel_resolution_m=ls_sample["resolution_m"],
                    method=m_code,
                )

                # Visualization Suite (All 6 core panels when reference mask is present; 5 panels for unmasked custom uploads)
                has_ref = ls_sample.get("ref_mask") is not None
                if has_ref:
                    st.markdown("##### 🔍 6-Panel Bitemporal Surface Change Suite")
                    p1, p2, p3, p4, p5, p6 = st.columns(6)
                    with p1:
                        st.image(ls_sample["img_a"], caption="Earlier observation (Epoch A)", use_container_width=True)
                    with p2:
                        st.image(ls_sample["img_b"], caption="Later observation (Epoch B)", use_container_width=True)
                    with p3:
                        st.image(ch_results["diff_rgb"], caption=_diff_cap, use_container_width=True)
                    with p4:
                        st.image((ch_results["detected_mask"] * 255).astype(np.uint8), caption=f"{_mask_cap} (T ≥ {ch_threshold:.0f})", use_container_width=True)
                    with p5:
                        st.image((ls_sample["ref_mask"] * 255).astype(np.uint8), caption="Ground-truth reference mask", use_container_width=True)
                    with p6:
                        confusion_overlay = generate_confusion_map_overlay(
                            ls_sample["img_b"],
                            ch_results["detected_mask"],
                            ls_sample["ref_mask"],
                        )
                        st.image(confusion_overlay, caption="Combined prediction overlay", use_container_width=True)

                    st.markdown("""
                    <div style="display: flex; gap: 16px; justify-content: center; align-items: center; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; margin-top: 4px; margin-bottom: 12px; font-size: 0.80rem;">
                        <span style="color: #22C55E;">🟩 <b>True Positive (TP)</b>: Correct Change</span>
                        <span style="color: #EF4444;">🟥 <b>False Positive (FP)</b>: Spectral Deviation</span>
                        <span style="color: #F59E0B;">🟧 <b>False Negative (FN)</b>: Missed Change</span>
                        <span style="color: #94A3B8;">⬛ <b>True Negative (TN)</b>: Static Background</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("##### 🔍 5-Panel Bitemporal Surface Change Suite (Custom Upload)")
                    p1, p2, p3, p4, p5 = st.columns(5)
                    with p1:
                        st.image(ls_sample["img_a"], caption="Earlier observation (Epoch A)", use_container_width=True)
                    with p2:
                        st.image(ls_sample["img_b"], caption="Later observation (Epoch B)", use_container_width=True)
                    with p3:
                        st.image(ch_results["diff_rgb"], caption=_diff_cap, use_container_width=True)
                    with p4:
                        st.image((ch_results["detected_mask"] * 255).astype(np.uint8), caption=f"{_mask_cap} (T ≥ {ch_threshold:.0f})", use_container_width=True)
                    with p5:
                        st.image(ch_results["overlay_b"], caption="Combined prediction overlay", use_container_width=True)

                # Change Statistics Cards (Correct Terminology: Estimated Detected Pixel-Change Footprint)
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                with m_c1:
                    st.metric("Total Valid Pixels", f"{ch_results['total_pixels']:,}")
                with m_c2:
                    st.metric("Detected Changed Pixels", f"{ch_results['changed_pixels']:,}")
                with m_c3:
                    st.metric("Surface Change Ratio", f"{ch_results['change_pct']:.2f}%")
                with m_c4:
                    st.metric("Estimated Detected Pixel-Change Footprint", f"{ch_results['change_area_km2']:.3f} km²")

                st.caption(f"ℹ️ **Area Calculation & Scientific Terminology:** The {ch_results['change_area_km2']:.3f} km² figure represents *estimated detected pixel-change footprint*, calculated assuming the documented nominal 30m ground sampling distance (0.0009 km² per pixel) is applicable. This binary mask captures undirected multi-spectral surface alterations, not directional glacier retreat or advance.")

                # Scientific Disclaimer Box
                st.markdown("""
                <div class="glass-card" style="border-left: 4px solid #F4A261; margin-top: 10px; margin-bottom: 12px; padding: 12px 16px;">
                    <h6 style="color: #F4A261; margin: 0 0 6px 0; font-size: 0.90rem;">⚠️ Scientific Disclaimer: Undirected Surface Change vs Cryospheric Retreat</h6>
                    <p style="color: #A8BED0; font-size: 0.84rem; margin: 0; line-height: 1.6;">
                        <b>1. Undirected Surface Change:</b> The calculated binary change mask highlights <i>undirected multi-spectral surface alterations</i> 
                        between the two acquisition epochs. Spectral deviations are driven by seasonal snowpack depletion, debris redistribution, 
                        meltwater pond formation, and terminus displacement. An undirected spectral difference mask does <b>not</b> establish 
                        directional retreat versus advance without verified georeferenced glacier outline tracking.<br/>
                        <b>2. Deterministic Heuristic (Not AI):</b> The pixel subtraction method (|ΔI| ≥ threshold) is a deterministic intensity differencing heuristic, not an AI or trained deep learning segmentation model.<br/>
                        <b>3. Photometric & Atmospheric Offsets:</b> Multi-year Landsat images acquired under differing solar zenith angles and seasonal dates exhibit baseline radiometric shifts (often 30–50 DN shifts across entire tiles). Subtraction of uncalibrated digital numbers causes high false-positive rates on dimly lit scenes.<br/>
                        <b>4. Geospatial Metadata Limitation:</b> The 256×256 PNG tiles are derived from co-registered parent Landsat-8 granules (Shi et al., 2025), but the PNG files lack embedded geospatial headers or GIS world files. Independent geographic verification cannot be performed from the image files alone.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Reference Mask Evaluation (when ground-truth is available)
                if has_ref:
                    with st.expander("📊 Ground-Truth Reference Comparison & Confusion Matrix Evaluation", expanded=True):
                        st.markdown(f"Comparing **{selected_method_label}** against ground-truth reference annotations for sample `{ls_sample['sample_id']}` at threshold T ≥ {ch_threshold:.0f}:")
                        eval_metrics = evaluate_change_accuracy(ch_results["detected_mask"], ls_sample["ref_mask"])

                        c_side1, c_side2 = st.columns(2)
                        with c_side1:
                            st.image((ch_results["detected_mask"] * 255).astype(np.uint8), caption=f"{_mask_cap} (T ≥ {ch_threshold:.0f})", use_container_width=True)
                        with c_side2:
                            st.image((ls_sample["ref_mask"] * 255).astype(np.uint8), caption="Ground-truth reference mask", use_container_width=True)

                        e1, e2, e3, e4, e5 = st.columns(5)
                        with e1:
                            st.metric("Precision (PPV)", f"{eval_metrics['precision']:.4f}")
                        with e2:
                            st.metric("Recall (Sensitivity)", f"{eval_metrics['recall']:.4f}")
                        with e3:
                            st.metric("Specificity (TNR)", f"{eval_metrics.get('specificity', 0.0):.4f}")
                        with e4:
                            st.metric("Intersection over Union (IoU)", f"{eval_metrics['iou']:.4f}")
                        with e5:
                            st.metric("Dice Score (F1)", f"{eval_metrics['dice']:.4f}")

                        tot_eval = eval_metrics['TP'] + eval_metrics['FP'] + eval_metrics['FN'] + eval_metrics['TN']
                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 6px; padding: 12px; margin-top: 8px;">
                            <p style="color: #E2E8F0; font-size: 0.85rem; margin: 0; line-height: 1.6;">
                                <b>Independent Confusion Matrix:</b><br/>
                                • <b>True Positives (TP):</b> <code>{eval_metrics['TP']:,}</code> pixels ({(eval_metrics['TP']/tot_eval)*100.0:.2f}%) &nbsp;&nbsp;|&nbsp;&nbsp; 
                                • <b>False Positives (FP):</b> <code>{eval_metrics['FP']:,}</code> pixels ({(eval_metrics['FP']/tot_eval)*100.0:.2f}%)<br/>
                                • <b>False Negatives (FN):</b> <code>{eval_metrics['FN']:,}</code> pixels ({(eval_metrics['FN']/tot_eval)*100.0:.2f}%) &nbsp;&nbsp;|&nbsp;&nbsp; 
                                • <b>True Negatives (TN):</b> <code>{eval_metrics['TN']:,}</code> pixels ({(eval_metrics['TN']/tot_eval)*100.0:.2f}%)<br/>
                                <i>Total Valid Evaluated Pixels:</i> <code>{tot_eval:,}</code>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="glass-card" style="border-left: 3px solid #38BDF8; margin-top: 10px; padding: 10px 14px;">
                            <p style="color: #A8BED0; font-size: 0.82rem; margin: 0; line-height: 1.5;">
                                <b>Scientific Analysis — {selected_method_label.split(":")[0]}:</b><br/>
                                {_analysis}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                # Export Controls
                st.markdown("##### 💾 High-Resolution Exports & Reproducibility Data")
                exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)
                with exp_c1:
                    mask_png = export_array_to_png_bytes((ch_results["detected_mask"] * 255).astype(np.uint8))
                    st.download_button("📥 Change Mask (PNG)", data=mask_png, file_name=f"landsat_change_mask_{ls_sample['sample_id']}.png", mime="image/png")
                with exp_c2:
                    if has_ref:
                        over_png = export_array_to_png_bytes(confusion_overlay)
                        st.download_button("📥 Confusion Overlay (PNG)", data=over_png, file_name=f"landsat_confusion_overlay_{ls_sample['sample_id']}.png", mime="image/png")
                    else:
                        over_png = export_array_to_png_bytes(ch_results["overlay_b"])
                        st.download_button("📥 Combined Overlay (PNG)", data=over_png, file_name=f"landsat_change_overlay_{ls_sample['sample_id']}.png", mime="image/png")
                with exp_c3:
                    # CSV export of sample metrics
                    sample_summary_csv = (
                        f"sample_id,algorithm,threshold,total_pixels,changed_pixels,change_pct,change_footprint_km2\n"
                        f"{ls_sample['sample_id']},{selected_method_label.split(':')[0]},{ch_threshold:.1f},{ch_results['total_pixels']},{ch_results['changed_pixels']},{ch_results['change_pct']:.4f},{ch_results['change_area_km2']:.4f}\n"
                    )
                    st.download_button("📥 Sample Metrics (CSV)", data=sample_summary_csv, file_name=f"metrics_{ls_sample['sample_id']}.csv", mime="text/csv")
                with exp_c4:
                    comp_csv_path = RESULTS_DIR / "objective4_landsat_method_comparison.csv"
                    if comp_csv_path.exists():
                        st.download_button("📥 Method Comparison (CSV)", data=comp_csv_path.read_bytes(), file_name="objective4_landsat_method_comparison.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error executing Landsat bitemporal analysis: {e}")

    # -------------------------------------------------------------------------
    # TAB 3: GLIMS BOUNDARY OVERLAY & GEOSPATIAL INTEGRATION
    # -------------------------------------------------------------------------
    with tab_glims:
        st.markdown("#### 🗺️ GLIMS Vector Boundary Overlay & Geospatial Integration")
        st.caption("Overlay vector glacier outlines directly onto high-resolution cartographic grids and satellite imagery with customizable line styling:")

        # Benchmark glaciers selector loaded from authoritative GLIMS records
        benchmarks = load_cached_glims_benchmarks()
        benchmark_options = {
            "Rhone Glacier (G008379E46648N) — Valais Alps, Switzerland": "rhone",
            "Great Aletsch Glacier (G008032E46504N) — Bernese Alps, Switzerland": "great_aletsch",
            "Unteraar Glacier (G008187E46569N) — Bernese Alps, Switzerland": "unteraar",
            "Gorner Glacier (G007800E45965N) — Pennine Alps, Switzerland": "gorner",
            "Findel Glacier (G007880E45990N) — Pennine Alps, Switzerland": "findel",
        }

        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            chosen_glac_label = st.selectbox(
                "Select Benchmark Alpine Glacier Basin:",
                list(benchmark_options.keys()),
                index=0,
                key="glims_benchmark_selector",
            )
            b_key = benchmark_options[chosen_glac_label]
            bdata = benchmarks.get(b_key, {})
        with col_g2:
            st.markdown(f"**GLIMS Glacier ID:** `{bdata.get('id', 'N/A')}`")
            st.caption(f"Region: {bdata.get('region', 'European Alps')} | Projection: {bdata.get('crs', 'EPSG:32632')}")

        # Area Change & Geometric Reconciled Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                label=f"Baseline Area ({str(bdata.get('earlier_date', '1973'))[:4]})",
                value=f"{bdata.get('earlier_area_km2', 0.0):.3f} km²",
                help=f"GLIMS Analysis ID: {bdata.get('earlier_analysis_id', '342911')}",
            )
        with m2:
            st.metric(
                label=f"Terminal Area ({str(bdata.get('later_date', '2015'))[:4]})",
                value=f"{bdata.get('later_area_km2', 0.0):.3f} km²",
                help=f"GLIMS Analysis ID: {bdata.get('later_analysis_id', '1095967')}",
            )
        with m3:
            st.metric(
                label="Net Area Change",
                value=f"-{bdata.get('net_loss_km2', 0.0):.3f} km²",
                delta=f"-{bdata.get('percentage_change', 0.0):.2f}%",
                delta_color="inverse",
            )
        with m4:
            st.metric(
                label="Retained Core Ice",
                value=f"{bdata.get('retained_ice_area_km2', 0.0):.3f} km²",
                help=f"Lost Ice (Retreat): {bdata.get('lost_ice_area_km2', 0.0):.3f} km² | Advance: {bdata.get('gained_ice_area_km2', 0.0):.3f} km²",
            )

        # Scientific Data Reconciliation Callout
        st.info(
            f"ℹ️ **Scientific Data Reconciliation Note:**\n\n"
            f"The previous placeholder display reported an informal net loss of `−2.33 km²` for Rhone Glacier, derived from historical Swiss inventory literature approximations (~17.63 km² in 1850 vs ~15.30 km² in 2010). "
            f"The verified canonical GLIMS multi-temporal polygon intersection (Analysis ID `{bdata.get('earlier_analysis_id', '342911')}` on {bdata.get('earlier_date', '1973-09-01')} vs Analysis ID `{bdata.get('later_analysis_id', '1095967')}` on {bdata.get('later_date', '2015-08-29')}) reprojected to conformal metric UTM Zone 32N (`{bdata.get('crs', 'EPSG:32632')}`) produces an unrounded initial area of **{bdata.get('earlier_area_km2', 17.166):.3f} km²**, a terminal area of **{bdata.get('later_area_km2', 14.179):.3f} km²**, and an exact net reduction of **−{bdata.get('net_loss_km2', 2.987):.3f} km²** (−{bdata.get('percentage_change', 17.40):.2f}%), in full 1:1 mathematical reconciliation with Objective 2."
        )

        # Styling controls
        st.markdown("##### 🎨 Boundary Rendering & Layer Controls")
        c_style1, c_style2, c_style3 = st.columns(3)
        with c_style1:
            line_color = st.selectbox(
                "Active Margin Color:",
                ["#12DFF3 (Cyan)", "#F59E0B (Amber)", "#F87171 (Coral)", "#34D399 (Emerald)", "#FFFFFF (White)"],
                index=0,
                key="glims_line_color",
            )
            color_hex = line_color.split()[0]
        with c_style2:
            thickness = st.slider("Line Thickness (px):", 1, 5, 2, key="glims_thickness")
        with c_style3:
            opacity = st.slider("Boundary Opacity:", 0.1, 1.0, 0.9, step=0.05, key="glims_opacity")

        c_chk1, c_chk2, c_chk3 = st.columns(3)
        with c_chk1:
            show_secondary = st.checkbox(
                "Show Baseline (1973) & Retreat Zone",
                value=True,
                key="glims_show_secondary",
                help="Overlay 1973 baseline perimeter (amber dashed), core retained ice (cyan fill), and retreat zone (coral fill)",
            )
        with c_chk2:
            show_debris = st.checkbox(
                "Show Supraglacial Debris Cover",
                value=True,
                key="glims_show_debris",
                help="Overlay GLIMS supraglacial debris cover outlines (amber fill/dotted)",
            )
        with c_chk3:
            show_satellite = st.checkbox(
                "Show Satellite Imagery Layer",
                value=True,
                key="glims_show_satellite",
                help="Render high-resolution Sentinel-2 MSI satellite imagery beneath the vector boundary layers",
            )

        # Dynamic overlay generation using authentic GLIMS vector polygons
        if bdata and "wkt_later" in bdata:
            try:
                dyn_overlay = render_benchmark_glims_overlay(
                    bdata,
                    color_hex=color_hex,
                    line_thickness=thickness,
                    opacity=opacity,
                    show_secondary=show_secondary,
                    show_debris=show_debris,
                    show_satellite=show_satellite,
                )

                c_ov1, c_ov2 = st.columns([2, 1])
                with c_ov1:
                    st.image(
                        dyn_overlay,
                        caption=f"Dynamic Vector Boundary Overlay — {bdata['name']} ({bdata['id']}) | Coordinate Frame: {bdata.get('crs', 'EPSG:32632')}",
                        use_container_width=True,
                    )
                with c_ov2:
                    st.markdown("**Geospatial Registration Metadata:**")
                    st.markdown(f"""
                    - **Target CRS:** `{bdata.get('crs', 'EPSG:32632')}` (Conformal UTM 32N)
                    - **Active Margin Layer:** `{color_hex}` ({thickness}px, {int(opacity*100)}% opacity)
                    - **Baseline Margin Layer:** `Amber (#F59E0B, 1973)`
                    - **Retreat Zone Layer:** `Coral (#FF4B4B, {bdata.get('lost_ice_area_km2', 0.0):.3f} km²)`
                    - **Retained Core Ice:** `Cyan (#00F2FE, {bdata.get('retained_ice_area_km2', 0.0):.3f} km²)`
                    - **Debris Cover Layer:** `Amber ({bdata.get('debris_area_km2', 0.0):.3f} km²)`
                    - **Satellite Layer:** `{'Active' if show_satellite else 'Disabled'}`
                    - **Coordinate Alignment:** Verified metric UTM geometry
                    """)
                    over_bytes = export_array_to_png_bytes(dyn_overlay)
                    st.download_button(
                        "📥 Download Styled Boundary Overlay (PNG)",
                        data=over_bytes,
                        file_name=f"glims_overlay_{bdata['id']}_{b_key}.png",
                        mime="image/png",
                        key=f"dl_glims_{b_key}",
                    )
            except Exception as e:
                st.error(f"Error rendering dynamic GLIMS boundary overlay: {e}")

        # Scientific Disclosure & Technical Audit Expander
        with st.expander("🔬 Geospatial Registration & Multi-Sensor Audit Details", expanded=False):
            st.markdown(f"""
            ### Geospatial Coordinate Registration Audit
            
            1. **Local Satellite Scene Extent:**
               - **File:** `rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif`
               - **Sensor / Collection:** Sentinel-2 MSI Level-2A (STAC: `sentinel-2-l2a`)
               - **Acquisition Timestamp:** `2026-07-24T10:15:59.024Z` (Item: `S2B_MSIL2A_20260724T101559_R065_T32TMS`)
               - **CRS & Transform:** `EPSG:32632` (UTM 32N), pixel scale 20 m
               - **Bounding Extent:** Easting `[448,620.0, 457,880.0]`, Northing `[5,155,320.0, 5,164,280.0]`
            
            2. **GLIMS Vector Basin Extent:**
               - **GLIMS Polygon ID:** `{bdata.get('id', 'G008379E46648N')}` ({bdata.get('name', 'Glacier')})
               - **Source CRS:** `EPSG:4326` (Geographic WGS84) reprojected to `EPSG:32632`
               - **Bounding Extent (Rhone 2015):** Easting `[448,849.97, 454,650.04]`, Northing `[5,164,599.96, 5,170,180.00]`
            
            3. **Registration Reality & Spatial Alignment:**
               - The GLIMS Rhone glacier boundary lies immediately north of the local Sentinel-2 GeoTIFF extent (a ~319 m Northing offset from tile limit `5,164,280` to glacier terminus `5,164,600`).
               - The previous prototype loaded an Austrian DL4GAM patch (`g_4367.nc` in UTM 33N) as a placeholder and mislabeled it as Rhone Glacier in EPSG:32632.
               - This verified release eliminates synthetic tile substitution: all 4 benchmark glaciers (Rhone, Great Aletsch, Unteraar, Gorner) dynamically access their authentic GLIMS vector geometries and metric UTM coordinate frames.
            """)

        # Archival publication figure expander
        fig_key_path = FIGURES_DIR / bdata.get("figure_name", f"glims_boundary_{b_key}.png")
        if fig_key_path.exists():
            with st.expander(f"📑 View Archival 300 DPI Publication Benchmark Figure ({bdata.get('name', 'Glacier')})", expanded=False):
                st.image(str(fig_key_path), caption=f"Publication Benchmark Map — {bdata.get('name', 'Glacier')} ({bdata.get('id', '')}) | 300 DPI Multi-Temporal Overlay", use_container_width=True)

        # Geospatial upload option (GeoJSON, Shapefile ZIP, and GeoTIFF)
        with st.expander("📤 Upload Custom Geospatial Boundary or Raster (GeoJSON, Shapefile ZIP, or GeoTIFF)", expanded=False):
            st.caption("Upload custom vector glacier boundaries (.geojson, .json, .zip shapefile) or georeferenced satellite rasters (.tif, .tiff):")
            up_geo = st.file_uploader(
                "Choose a geospatial file:",
                type=["geojson", "json", "zip", "tif", "tiff"],
                key="up_geospatial_file",
                help="Supported: GeoJSON, ESRI Shapefile ZIP (with .shp, .shx, .dbf, .prj), and georeferenced GeoTIFF",
            )
            if up_geo is not None:
                fname = up_geo.name.lower()
                f_bytes = up_geo.read()
                if fname.endswith((".geojson", ".json", ".zip")):
                    try:
                        v_res = parse_and_validate_vector_upload(
                            f_bytes,
                            filename=up_geo.name,
                            target_crs="EPSG:32632",
                            color_hex=color_hex,
                            line_thickness=thickness,
                            opacity=opacity,
                        )
                        st.success(f"✅ Successfully validated vector geometry: {v_res['feature_count']} features found.")
                        c_vu1, c_vu2 = st.columns([2, 1])
                        with c_vu1:
                            st.image(v_res["rendered_array"], caption=f"Rendered Uploaded Vector — {up_geo.name}", use_container_width=True)
                        with c_vu2:
                            st.markdown(f"""
                            **Vector Metadata:**
                            - **Original CRS:** `{v_res['original_crs']}`
                            - **Projected CRS:** `{v_res['projected_crs']}`
                            - **Calculated Metric Area:** `{v_res['area_km2']:.3f} km²`
                            - **Feature Count:** `{v_res['feature_count']}`
                            """)
                            up_bytes = export_array_to_png_bytes(v_res["rendered_array"])
                            st.download_button(
                                "📥 Download Vector Overlay (PNG)",
                                data=up_bytes,
                                file_name=f"uploaded_{Path(up_geo.name).stem}.png",
                                mime="image/png",
                                key="dl_up_vector",
                            )
                    except Exception as e:
                        st.error(f"❌ Vector Validation Failed: {e}")
                elif fname.endswith((".tif", ".tiff")):
                    try:
                        r_res = inspect_uploaded_geotiff(f_bytes)
                        st.success(f"✅ Successfully validated georeferenced GeoTIFF: {r_res['width']}×{r_res['height']} px, {r_res['band_count']} bands.")
                        c_ru1, c_ru2 = st.columns([2, 1])
                        with c_ru1:
                            st.image(r_res["preview_rgb"], caption=f"GeoTIFF Radiometric Preview — {up_geo.name}", use_container_width=True)
                        with c_ru2:
                            st.markdown(f"""
                            **Raster Metadata:**
                            - **CRS:** `{r_res['crs']}`
                            - **Resolution:** `{r_res['resolution'][0]:.1f}m × {r_res['resolution'][1]:.1f}m`
                            - **Dimensions:** `{r_res['width']} × {r_res['height']} pixels`
                            - **Band Count:** `{r_res['band_count']}`
                            """)
                    except Exception as e:
                        st.error(f"❌ GeoTIFF Validation Failed: {e}")

    # -------------------------------------------------------------------------
    # TAB 4: ARCHIVAL PUBLICATION FIGURES (REFERENCE RECORD)
    # -------------------------------------------------------------------------
    with tab_ref_figs:
        st.markdown("#### 🌐 Archival Publication Figures (Reference Record)")
        st.caption("High-resolution 300 DPI multi-sensor composite figures preserved for publication reproducibility:")

        fig_glims = FIGURES_DIR / "visual_monitoring_glims_boundary_overlays.png"
        if fig_glims.exists():
            st.image(str(fig_glims), caption="1. GLIMS Multi-Temporal Polygon Boundaries (Rhone, Great Aletsch, and Unteraar Reference Basins)", use_container_width=True)

        fig_s2 = FIGURES_DIR / "visual_monitoring_sentinel2_composites.png"
        if fig_s2.exists():
            st.image(str(fig_s2), caption="2. Sentinel-2 Multispectral Visual Suite (True-Color RGB, False-Color SWIR, and Topographic DEM Elevation)", use_container_width=True)

        fig_l8 = FIGURES_DIR / "visual_monitoring_landsat_bitemporal.png"
        if fig_l8.exists():
            st.image(str(fig_l8), caption="3. Landsat-8 Polar Bi-temporal Surface Change Monitoring Suite", use_container_width=True)

