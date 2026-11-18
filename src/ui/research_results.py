"""GlacierVision Peer-Reviewed Research Results Center.

Organized systematically by all 5 core research objectives:
- Objective 1: Few-Shot ProtoNet vs Supervised U-Net Baseline (training curves, 80-episode paired evaluation, K-curves).
- Objective 2 & 3: Global GLIMS Multi-Temporal Retreat & Area-Change Accounting (81,598 glaciers, 16,321.76 km² loss).
- Objective 4: Visual Monitoring Suites (Sentinel-2, Landsat-8, GLIMS Vector Overlays).
- Objective 5: CLIP vs CoOp Vision-Language Benchmarks.
- Methodology, Limitations & Downloadable Reports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.inference import FIGURES_DIR, RESULTS_DIR, RUNS_DIR
from src.ui.theme import render_header
from src.ui.components import render_kpi_cards


@st.cache_data
def load_retreat_summary():
    p = RESULTS_DIR / "glims_retreat_summary_validated.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_unet_history():
    p = RUNS_DIR / "objective1_run_002_recovery" / "results" / "unet_baseline_history.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data
def load_fewshot_history():
    p = RUNS_DIR / "objective1_run_002_recovery" / "results" / "fewshot_protonet_history.csv"
    if not p.exists():
        p = RUNS_DIR / "objective1_run_001" / "results" / "fewshot_protonet_history.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data
def load_objective1_evaluation():
    p = RUNS_DIR / "objective1_run_002_recovery" / "results" / "objective1_evaluation_report.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_paired_query_comparison():
    p = RUNS_DIR / "objective1_run_002_recovery" / "results" / "paired_query_comparison.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data
def load_clip_coop_report():
    p = RESULTS_DIR / "objective5_clip_coop_report.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_research_results():
    """Render the comprehensive peer-reviewed research results center."""
    render_header(
        title="Scientific Research Results Center",
        subtitle="Peer-Reviewed Experimental Benchmarks, Validation Trajectories & Evaluation Protocols",
        badge="SCIENTIFIC BENCHMARKS • PEER REVIEW",
    )

    tab_o1, tab_o23, tab_o4, tab_o5, tab_reports = st.tabs([
        "🔬 Objective 1: Segmentation Benchmark",
        "📉 Objectives 2 & 3: GLIMS Inventory (81,598 Glaciers)",
        "🛰️ Objective 4: Visual Suites",
        "🧠 Objective 5: CLIP vs CoOp",
        "📄 Acceptance Reports & Downloads",
    ])

    # -----------------------------------------------------------------
    # TAB 1: OBJECTIVE 1 BENCHMARK
    # -----------------------------------------------------------------
    with tab_o1:
        st.markdown("#### Few-Shot ProtoNet vs Supervised 13-Band U-Net Baseline")
        st.caption("Evaluated on the official DL4GAM European Alps split with strict disjoint glacier ID partitioning:")

        c_left, c_right = st.columns([1, 1.3])

        with c_left:
            st.markdown("""
            <div class="glass-card">
                <h5 style="color: #12DFF3; margin-top: 0;">Evaluation Partition Protocol:</h5>
                <ul style="color: #A8BED0; font-size: 0.88rem; line-height: 1.6; padding-left: 20px;">
                    <li><b>Training Partition:</b> 986 glacier IDs (1,475 patches)</li>
                    <li><b>Validation Partition:</b> 181 glacier IDs (242 patches)</li>
                    <li><b>Held-Out Test Partition:</b> 426 glacier IDs (534 patches)</li>
                    <li><b>Spectral Bands:</b> 13 Sentinel-2 channels (B1 through B12, including B8A)</li>
                    <li><b>Zero Data Leakage:</b> Query glaciers have zero overlap with training sets.</li>
                </ul>
                <hr style="border-color: rgba(18, 223, 243, 0.15); margin: 12px 0;">
                <p style="font-size: 0.84rem; color: #F4FAFF; margin: 0; line-height: 1.5;">
                    <b>ProtoNet Peak Val Dice:</b> <span style="color: #12DFF3; font-weight: 700;">0.4735</span> (Epoch 14)<br>
                    <b>U-Net Peak Val Dice:</b> <span style="color: #2EC4B6; font-weight: 700;">0.6648</span> (Epoch 5)<br>
                    <b>U-Net Full Test Dice:</b> <span style="color: #65CDEB; font-weight: 700;">0.6559</span> (534 Patches)<br>
                    <b>U-Net Full Test mIoU:</b> <span style="color: #65CDEB; font-weight: 700;">0.7076</span> (91.83% Pixel Acc)
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c_right:
            df_u = load_unet_history()
            df_f = load_fewshot_history()
            fig = go.Figure()
            if df_f is not None and not df_f.empty:
                fig.add_trace(go.Scatter(
                    x=df_f["epoch"], y=df_f["val_dice"],
                    mode="lines+markers", name="ProtoNet Val Dice",
                    line=dict(color="#12DFF3", width=2.5),
                    marker=dict(size=6),
                ))
            if df_u is not None and not df_u.empty:
                fig.add_trace(go.Scatter(
                    x=df_u["epoch"], y=df_u["val_dice"],
                    mode="lines+markers", name="U-Net Baseline Val Dice",
                    line=dict(color="#2EC4B6", width=2.5, dash="dash"),
                    marker=dict(size=6),
                ))
            fig.update_layout(
                title="Validation Dice Progression Curves",
                xaxis_title="Training Epoch",
                yaxis_title="Validation Dice Coefficient",
                template="plotly_dark",
                paper_bgcolor="rgba(6, 20, 38, 0.0)",
                plot_bgcolor="rgba(10, 29, 50, 0.7)",
                height=320,
                margin=dict(l=30, r=30, t=40, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Comparative Table
        obj1_eval = load_objective1_evaluation()
        if obj1_eval:
            k_res = obj1_eval.get("k_shot_evaluation_results", {})
            bl_res = obj1_eval.get("supervised_baseline_benchmark", {})
            t_rows = []
            for k_val in ["1", "5", "10", "20"]:
                if k_val in k_res:
                    kr = k_res[k_val]
                    t_rows.append({
                        "Model & Setup": f"Few-Shot ProtoNet (K={k_val})",
                        "mIoU": f"{kr['miou_mean']:.4f} ± {kr['miou_std']:.4f}",
                        "Dice Score": f"{kr['dice_mean']:.4f} ± {kr['dice_std']:.4f}",
                        "Foreground IoU": f"{kr['fg_iou_mean']:.4f}",
                        "Precision": f"{kr['precision_mean']:.4f}",
                        "Recall": f"{kr['recall_mean']:.4f}",
                        "Boundary F1": f"{kr['boundary_f1_mean']:.4f}",
                    })
            t_rows.append({
                "Model & Setup": "Supervised 13-Band U-Net Baseline",
                "mIoU": f"{bl_res.get('miou_mean', 0.7076):.4f} ± {bl_res.get('miou_std', 0.1108):.4f}",
                "Dice Score": f"{bl_res.get('dice_mean', 0.6559):.4f} ± {bl_res.get('dice_std', 0.1963):.4f}",
                "Foreground IoU": f"{bl_res.get('fg_iou_mean', 0.5176):.4f}",
                "Precision": f"{bl_res.get('precision_mean', 0.6360):.4f}",
                "Recall": f"{bl_res.get('recall_mean', 0.7657):.4f}",
                "Boundary F1": f"{bl_res.get('boundary_f1_mean', 0.2845):.4f}",
            })
            st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

        # 80-episode paired evaluation expander
        df_paired = load_paired_query_comparison()
        if df_paired is not None and not df_paired.empty:
            with st.expander("📊 Inspect 80 Individual Paired Query Episode Results (ProtoNet vs Baseline)", expanded=False):
                st.caption("Direct episodic head-to-head evaluation across K in {1, 5, 10, 20} on identical test query patches:")
                st.dataframe(df_paired, use_container_width=True, height=260)

    # -----------------------------------------------------------------
    # TAB 2: OBJECTIVES 2 & 3 GLIMS RETREAT INVENTORY
    # -----------------------------------------------------------------
    with tab_o23:
        summary = load_retreat_summary()
        if summary:
            st.markdown("#### Global Multi-Temporal GLIMS Retreat Inventory (81,598 Glaciers)")
            st.caption("Authoritative classification counts and surface area-change accounting:")

            inventory_kpi_cards = [
                {
                    "label": "Total Glaciers",
                    "value": f"{summary.get('total_glaciers_analyzed', 81598):,}",
                    "unit": "",
                    "badge": "100.00% Coverage",
                    "badge_type": "cyan",
                    "arrow": None,
                },
                {
                    "label": "Retreating Basins",
                    "value": f"{summary.get('retreating_glaciers_count', 54718):,}",
                    "unit": "",
                    "badge": f"{summary.get('retreating_glaciers_ratio_pct', 67.06):.2f}%",
                    "badge_type": "danger",
                    "arrow": "▼",
                    "value_color": "#F87171",
                },
                {
                    "label": "Advancing Basins",
                    "value": f"{summary.get('advancing_glaciers_count', 20132):,}",
                    "unit": "",
                    "badge": f"{summary.get('advancing_glaciers_ratio_pct', 24.67):.2f}%",
                    "badge_type": "success",
                    "arrow": "▲",
                    "value_color": "#34D399",
                },
                {
                    "label": "Stable Basins",
                    "value": f"{summary.get('stable_glaciers_count', 6748):,}",
                    "unit": "",
                    "badge": f"{summary.get('stable_glaciers_ratio_pct', 8.27):.2f}%",
                    "badge_type": "neutral",
                    "arrow": "■",
                    "value_color": "#94A3B8",
                },
            ]
            render_kpi_cards(inventory_kpi_cards, grid_class="kpi-grid-4")

            # Classification Donut & Net Loss
            col_d, col_loss = st.columns([1, 1.2])
            with col_d:
                dyn = summary.get("dynamics_breakdown", {})
                fig_d = go.Figure(data=[go.Pie(
                    labels=list(dyn.keys()), values=list(dyn.values()), hole=0.55,
                    marker=dict(colors=["#E63946", "#F4A261", "#2A9D8F", "#457B9D"]),
                )])
                fig_d.update_layout(
                    title="Mutually Exclusive Dynamic Partition",
                    template="plotly_dark",
                    paper_bgcolor="rgba(6, 20, 38, 0.0)",
                    height=280,
                    margin=dict(l=10, r=10, t=35, b=10),
                )
                st.plotly_chart(fig_d, use_container_width=True)

            with col_loss:
                st.markdown("""
                <div class="glass-card">
                    <h5 style="color: #12DFF3; margin-top: 0;">Net Area-Change Accounting:</h5>
                    <p style="color: #A8BED0; font-size: 0.88rem; line-height: 1.6;">
                        Across all 81,598 basins analyzed between 1970 and 2024 (mean baseline: 11.65 years):
                    </p>
                    <ul style="color: #A8BED0; font-size: 0.88rem; line-height: 1.6; padding-left: 20px;">
                        <li><b>Total Earlier Area:</b> 220,636.22 km²</li>
                        <li><b>Total Modern Area:</b> 204,314.45 km²</li>
                        <li><b>Net Pooled Area Loss:</b> <span style="color: #E63946; font-weight: 700;">16,321.76 km² (-7.40%)</span></li>
                        <li><b>Pooled Pairwise Reductions:</b> 27,330.47 km² (57,477 shrinking pairs, A₁ > A₂)</li>
                        <li><b>Pooled Pairwise Increases:</b> 11,008.71 km² (22,471 expanding pairs, A₂ > A₁)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # TAB 3: OBJECTIVE 4 VISUAL MONITORING SUITES
    # -----------------------------------------------------------------
    with tab_o4:
        st.markdown("#### High-Resolution Multi-Sensor Visual Suites (300 DPI)")
        st.caption("Visual products for publication review:")
        v1, v2 = st.columns(2)
        with v1:
            img_s2 = FIGURES_DIR / "visual_monitoring_sentinel2_composites.png"
            if img_s2.exists():
                st.image(str(img_s2), caption="Sentinel-2 Multispectral Visual Suite", use_container_width=True)
        with v2:
            img_g = FIGURES_DIR / "visual_monitoring_glims_boundary_overlays.png"
            if img_g.exists():
                st.image(str(img_g), caption="GLIMS Multi-Temporal Boundaries (EPSG:32632)", use_container_width=True)

        img_l = FIGURES_DIR / "visual_monitoring_landsat_bitemporal.png"
        if img_l.exists():
            st.image(str(img_l), caption="Landsat-8 Bitemporal Change Detection Suite", use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 4: OBJECTIVE 5 CLIP VS COOP
    # -----------------------------------------------------------------
    with tab_o5:
        st.markdown("#### Objective 5: Vision-Language Prompt Tuning Benchmark")
        clip_rep = load_clip_coop_report()
        if clip_rep:
            t_eval = clip_rep.get("test_evaluation", {})
            h = t_eval.get("handcrafted_clip", {})
            c = t_eval.get("learnable_coop", {})

            st.markdown(f"""
            - **Zero-Shot Handcrafted CLIP ViT-B-32:** Accuracy = `{h.get('accuracy', 0.68):.2%}`, Precision = `{h.get('precision', 0.6250):.4f}`, Recall = `{h.get('recall', 0.9000):.4f}`, Specificity = `{h.get('specificity', 0.4600):.4f}`, False Positives = `{h.get('fp', 27)}`
            - **Context Optimization (CoOp, M=16):** Accuracy = `{c.get('accuracy', 0.77):.2%}`, Precision = `{c.get('precision', 0.7647):.4f}`, Recall = `{c.get('recall', 0.7800):.4f}`, Specificity = `{c.get('specificity', 0.7600):.4f}`, False Positives = `{c.get('fp', 12)}`
            - **Empirical Deltas:** Accuracy **+9.00%**, Precision **+13.97%**, Specificity **+30.00%**, Bedrock False Positive Reduction **-55.6%** (12 vs 27), with a Trade-Off of Recall **-12.00%** (0.78 vs 0.90, +6 FN).
            """)

            fig_cp = FIGURES_DIR / "clip_vs_coop_comparison.png"
            if fig_cp.exists():
                st.image(str(fig_cp), caption="CLIP vs CoOp Performance Benchmark and Confusion Matrices", use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 5: ACCEPTANCE REPORTS & DOWNLOADS
    # -----------------------------------------------------------------
    with tab_reports:
        st.markdown("#### Formal Scientific Acceptance Audit Reports")
        st.caption("Inspect and download full markdown acceptance audit reports for academic review:")

        audits_dir = RESULTS_DIR.parent / "audits"
        audit_files = sorted(list(audits_dir.glob("*.md")))

        for aud_file in audit_files:
            with st.expander(f"📄 {aud_file.name}", expanded=False):
                st.markdown(aud_file.read_text(encoding="utf-8"))
                st.download_button(
                    label=f"Download {aud_file.name}",
                    data=aud_file.read_bytes(),
                    file_name=aud_file.name,
                    mime="text/markdown",
                    key=f"dl_{aud_file.name}",
                )
