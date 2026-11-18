"""GlacierVision Reusable UI Components.

Provides responsive, high-contrast, unclippable KPI cards and scientific layout blocks
adhering to the dark navy, ice-blue, and glowing cyan design system.
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional
import streamlit as st


def render_kpi_cards(
    cards: List[Dict[str, Any]],
    grid_class: str = "kpi-grid",
) -> None:
    """Render a responsive grid of KPI cards with fluid typography and no clipping.

    Each card in `cards` may contain:
    - label: str (e.g. "Earlier Area")
    - value: str or float (e.g. "17.166" or 17.166)
    - unit: Optional[str] (e.g. "km²")
    - badge: Optional[str] (e.g. "Baseline (1973)")
    - badge_type: "cyan" | "danger" | "success" | "neutral" (default "cyan")
    - arrow: Optional[str] (e.g. "▼", "▲", or None)
    - value_color: Optional[str] (e.g. "#EF4444")
    """
    card_elements = []
    for c in cards:
        label = str(c.get("label", ""))
        val = str(c.get("value", ""))
        unit = str(c.get("unit", ""))
        badge = str(c.get("badge", ""))
        badge_type = str(c.get("badge_type", "cyan"))
        arrow = str(c.get("arrow", "") or "")
        val_color = str(c.get("value_color", "#F4FAFF"))

        unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
        arrow_html = f'<span class="kpi-arrow">{arrow}</span> ' if arrow else ""
        badge_html = (
            f'<div class="kpi-badge kpi-badge-{badge_type}">{arrow_html}{badge}</div>'
            if badge
            else ""
        )

        card_html = (
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value-row">'
            f'<span class="kpi-value" style="color: {val_color};">{val}</span>'
            f'{unit_html}'
            f'</div>'
            f'{badge_html}'
            f'</div>'
        )
        card_elements.append(card_html)

    # Use st.html() to render raw HTML directly into DOM without CommonMark code-block parsing
    full_html = f'<div class="{grid_class}">{"".join(card_elements)}</div>'
    if hasattr(st, "html"):
        st.html(full_html)
    else:
        st.markdown(full_html, unsafe_allow_html=True)
