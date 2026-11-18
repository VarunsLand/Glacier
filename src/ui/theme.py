"""GlacierVision Visual Design System and Styling Tokens.

Color Palette:
- Deep Navy Background: #061426
- Secondary Navy: #0A1D32
- Card Background: #0D243A
- Primary Glowing Cyan: #12DFF3
- Secondary Ice Blue: #65CDEB
- High-Contrast Text: #F4FAFF
- Muted Grey-Blue: #A8BED0
- Success Mint: #2EC4B6
- Warning Amber: #F4A261
- Danger Coral: #E63946
"""
from __future__ import annotations

import streamlit as st


def apply_custom_theme():
    """Inject custom CSS for the dark navy, ice-blue, and glowing cyan design system."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

        /* Root Typography & App Canvas */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            letter-spacing: -0.01em;
        }

        .stApp {
            background: radial-gradient(circle at 18% 12%, #0A1D32 0%, #061426 65%, #030B17 100%) !important;
            color: #F4FAFF;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #061221 !important;
            border-right: 1px solid rgba(18, 223, 243, 0.15) !important;
        }

        section[data-testid="stSidebar"] div.stRadio > label {
            color: #65CDEB !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            font-size: 0.78rem !important;
            letter-spacing: 0.08em;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #061426;
        }
        ::-webkit-scrollbar-thumb {
            background: #0D2D4B;
            border-radius: 4px;
            border: 1px solid rgba(18, 223, 243, 0.2);
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #12DFF3;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(13, 36, 58, 0.75);
            border: 1px solid rgba(18, 223, 243, 0.20);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 20px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
            transition: all 0.25s ease;
        }

        .glass-card:hover {
            border-color: rgba(18, 223, 243, 0.45);
            box-shadow: 0 12px 35px rgba(18, 223, 243, 0.08);
        }

        .glass-interactive {
            background: rgba(10, 29, 50, 0.85);
            border: 1px solid rgba(18, 223, 243, 0.25);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .glass-interactive:hover {
            transform: translateY(-2px);
            border-color: rgba(18, 223, 243, 0.6);
            box-shadow: 0 12px 32px rgba(18, 223, 243, 0.15);
        }

        /* Cinematic Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(10, 29, 50, 0.92) 0%, rgba(6, 20, 38, 0.95) 100%),
                        radial-gradient(ellipse at 80% 20%, rgba(18, 223, 243, 0.18) 0%, transparent 60%);
            border: 1px solid rgba(18, 223, 243, 0.3);
            border-radius: 20px;
            padding: 40px 36px;
            margin-bottom: 30px;
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }

        .hero-banner::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 380px;
            height: 100%;
            background: radial-gradient(circle, rgba(18, 223, 243, 0.15) 0%, transparent 70%);
            pointer-events: none;
        }

        /* Glowing Badges */
        .badge-cyan {
            background: linear-gradient(135deg, #12DFF3, #65CDEB);
            color: #030B17;
            padding: 4px 12px;
            border-radius: 24px;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            display: inline-block;
            box-shadow: 0 2px 10px rgba(18, 223, 243, 0.35);
        }

        .badge-ice {
            background: rgba(101, 205, 235, 0.15);
            color: #65CDEB;
            border: 1px solid rgba(101, 205, 235, 0.35);
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.74rem;
            font-weight: 700;
            display: inline-block;
        }

        .badge-amber {
            background: linear-gradient(135deg, #F4A261, #E76F51);
            color: #061426;
            padding: 4px 12px;
            border-radius: 24px;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            display: inline-block;
        }

        .badge-mint {
            background: linear-gradient(135deg, #2EC4B6, #20A497);
            color: #030B17;
            padding: 4px 12px;
            border-radius: 24px;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            display: inline-block;
        }

        /* ==========================================================================
           REUSABLE RESPONSIVE KPI CARDS (Fluid typography, no clipping, full units)
           ========================================================================== */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
            width: 100%;
        }

        .kpi-grid-5 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
            width: 100%;
        }

        @media (min-width: 1280px) {
            .kpi-grid-5 {
                grid-template-columns: repeat(5, 1fr);
            }
        }
        @media (max-width: 1279px) and (min-width: 860px) {
            .kpi-grid-5 {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        @media (max-width: 859px) and (min-width: 540px) {
            .kpi-grid-5 {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        @media (max-width: 539px) {
            .kpi-grid-5 {
                grid-template-columns: 1fr;
            }
        }

        .kpi-grid-4 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
            width: 100%;
        }

        @media (min-width: 1080px) {
            .kpi-grid-4 {
                grid-template-columns: repeat(4, 1fr);
            }
        }
        @media (max-width: 1079px) and (min-width: 640px) {
            .kpi-grid-4 {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        @media (max-width: 639px) {
            .kpi-grid-4 {
                grid-template-columns: 1fr;
            }
        }

        /* Mobile Responsive Viewport Tuning */
        @media (max-width: 640px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 2.2rem !important;
            }
            .hero-banner {
                padding: 24px 18px !important;
            }
            .glass-card {
                padding: 16px 14px !important;
            }
        }

        .kpi-card {
            background: rgba(13, 36, 58, 0.85);
            border: 1px solid rgba(18, 223, 243, 0.22);
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 115px;
            box-sizing: border-box;
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .kpi-card:hover {
            border-color: rgba(18, 223, 243, 0.55);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(18, 223, 243, 0.12);
        }

        .kpi-label {
            color: #A8BED0;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
            line-height: 1.35;
            white-space: normal;
            word-break: break-word;
        }

        .kpi-value-row {
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 4px;
            margin-bottom: 6px;
        }

        .kpi-value {
            color: #F4FAFF;
            font-size: clamp(1.35rem, 1.9vw, 1.75rem);
            font-weight: 800;
            font-family: 'Inter', sans-serif;
            line-height: 1.15;
            white-space: normal;
            word-break: normal;
            overflow-wrap: break-word;
        }

        .kpi-unit {
            font-size: 0.90rem;
            font-weight: 600;
            color: #94A3B8;
            line-height: 1.2;
        }

        .kpi-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 0.76rem;
            font-weight: 600;
            line-height: 1.3;
            padding: 3px 8px;
            border-radius: 6px;
            width: fit-content;
            max-width: 100%;
            white-space: normal;
            word-break: break-word;
        }

        .kpi-badge-cyan {
            background: rgba(18, 223, 243, 0.12);
            color: #12DFF3;
            border: 1px solid rgba(18, 223, 243, 0.30);
        }

        .kpi-badge-danger {
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }

        .kpi-badge-success {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        .kpi-badge-neutral {
            background: rgba(148, 163, 184, 0.14);
            color: #94A3B8;
            border: 1px solid rgba(148, 163, 184, 0.28);
        }

        .kpi-arrow {
            font-size: 0.72rem;
            line-height: 1;
        }

        /* ==========================================================================
           STREAMLIT NATIVE ST.METRIC OVERRIDES (Eliminates clipping app-wide)
           ========================================================================== */
        div[data-testid="stMetric"] {
            background: rgba(13, 36, 58, 0.85) !important;
            border: 1px solid rgba(18, 223, 243, 0.22) !important;
            border-radius: 12px !important;
            padding: 14px 16px !important;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35) !important;
            min-width: 0 !important;
            overflow: visible !important;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #A8BED0 !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: break-word !important;
            line-height: 1.35 !important;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F4FAFF !important;
            font-size: clamp(1.25rem, 1.8vw, 1.75rem) !important;
            font-weight: 800 !important;
            font-family: 'Inter', sans-serif !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: normal !important;
            overflow-wrap: break-word !important;
            line-height: 1.2 !important;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: break-word !important;
            font-size: 0.78rem !important;
            line-height: 1.35 !important;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] * {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: break-word !important;
        }

        /* Buttons */
        div.stButton > button {
            border-radius: 8px !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em !important;
            padding: 8px 10px !important;
            transition: all 0.2s ease !important;
            border: 1px solid rgba(18, 223, 243, 0.3) !important;
        }

        div.stButton > button p {
            font-size: 0.85rem !important;
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
            margin: 0 !important;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #12DFF3 0%, #00B4D8 100%) !important;
            color: #030B17 !important;
            border: 1px solid #12DFF3 !important;
            box-shadow: 0 4px 18px rgba(18, 223, 243, 0.45) !important;
        }

        div.stButton > button[kind="primary"] p {
            color: #030B17 !important;
            font-weight: 800 !important;
        }

        div.stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 24px rgba(18, 223, 243, 0.65) !important;
            transform: translateY(-1px) !important;
        }

        div.stButton > button[kind="secondary"] {
            background: rgba(13, 36, 58, 0.85) !important;
            color: #F4FAFF !important;
            border: 1px solid rgba(18, 223, 243, 0.25) !important;
            box-shadow: none !important;
        }

        div.stButton > button[kind="secondary"] p {
            color: #F4FAFF !important;
            font-weight: 700 !important;
        }

        div.stButton > button[kind="secondary"]:hover {
            border-color: #12DFF3 !important;
            color: #12DFF3 !important;
            background: rgba(18, 223, 243, 0.10) !important;
        }

        div.stButton > button[kind="secondary"]:hover p {
            color: #12DFF3 !important;
        }

        /* Tab Bar Styling */
        button[data-baseweb="tab"] {
            font-weight: 700 !important;
            color: #A8BED0 !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 10px 18px !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #12DFF3 !important;
            border-bottom: 2px solid #12DFF3 !important;
            background: rgba(18, 223, 243, 0.06) !important;
        }

        /* Code block & Pre formatting */
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
            background-color: rgba(6, 20, 38, 0.85) !important;
            border: 1px solid rgba(18, 223, 243, 0.15) !important;
            border-radius: 6px !important;
            color: #12DFF3 !important;
        }

        /* Responsive Tables for Dark Navy & Cyan Theme */
        .stMarkdown table {
            width: 100% !important;
            border-collapse: collapse !important;
            font-size: 0.86rem !important;
            color: #E2E8F0 !important;
            background: rgba(10, 29, 50, 0.75) !important;
            border: 1px solid rgba(18, 223, 243, 0.22) !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            margin: 16px 0 !important;
        }

        .stMarkdown th {
            background: rgba(13, 36, 58, 0.95) !important;
            color: #12DFF3 !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            font-size: 0.76rem !important;
            letter-spacing: 0.05em !important;
            padding: 10px 14px !important;
            border: 1px solid rgba(18, 223, 243, 0.18) !important;
            white-space: nowrap !important;
        }

        .stMarkdown td {
            padding: 9px 14px !important;
            border: 1px solid rgba(255, 255, 255, 0.07) !important;
            white-space: nowrap !important;
        }

        .stMarkdown tr:hover td {
            background: rgba(18, 223, 243, 0.06) !important;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: Optional[str] = None, badge: str = "RESEARCH PLATFORM"):
    """Render a standard header with ice-blue accents and subtle typography."""
    sub_html = f'<p style="color: #65CDEB; font-size: 1.05rem; margin-top: 6px; margin-bottom: 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <span class="badge-cyan">{badge}</span>
        <h1 style="color: #F4FAFF; font-size: 2.2rem; font-weight: 800; margin-top: 8px; margin-bottom: 0;">{title}</h1>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)
