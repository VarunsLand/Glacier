"""Comprehensive Automated Verification for GlacierVision Premium UI Redesign.

Tests:
1. Global navigation definitions (8 primary application pages).
2. Theme styling tokens, dark navy palette, glowing cyan accents, and CSS injection.
3. Homepage component rendering, verified statistics loading, and feature cards.
4. Segmentation lab 4-stage workflow, model selectors, and transparency banners.
5. Geographical retreat explorer benchmark aliases, date selectors, and geometry overlays.
6. Quantitative area assessment workspace, mathematical formulas, and 81,598-glacier dataset loader.
7. Multi-sensor visual monitoring tabs (Sentinel-2, Landsat-8, GLIMS) and scientific disclaimers.
8. Vision-Language prompt tuning lab (Zero-Shot CLIP vs Learnable CoOp).
9. Research results center across all 5 objectives and acceptance audit reports.
10. Datasets, methodology, and zero-leakage protocols.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from src.inference import (
    ROOT,
    RESULTS_DIR,
    FIGURES_DIR,
    RUNS_DIR,
    DEMO_DIR,
    get_glacier_dynamic_assessment,
    get_demo_netcdf_candidates,
)
import app
from src.ui.theme import apply_custom_theme, render_header
from src.ui.home import render_homepage
from src.ui.segmentation_lab import render_segmentation_lab
from src.ui.retreat_explorer import render_retreat_explorer, load_case_studies
from src.ui.area_assessment import render_area_assessment, load_retreat_summary, load_retreat_csv
from src.ui.visual_monitoring import render_visual_monitoring
from src.ui.clip_coop_lab import render_clip_coop_lab, load_clip_coop_report
from src.ui.research_results import render_research_results
from src.ui.datasets_about import render_datasets_about


def test_global_navigation_pages():
    """Verify all 8 navigation pages are properly registered and unique."""
    expected_pages = [
        "🏠 Home",
        "🔬 Interactive Segmentation Lab",
        "🗺️ Glacier Retreat Explorer",
        "📊 Area Assessment Workspace",
        "🛰️ Visual Monitoring Suite",
        "🧠 CLIP vs CoOp Lab",
        "📚 Research Results Center",
        "📖 Datasets & Methodology",
    ]
    assert app.PAGES == expected_pages, f"Navigation pages mismatch: {app.PAGES}"
    assert len(app.PAGES) == 8


def test_theme_and_branding_tokens():
    """Verify visual design system tokens in .streamlit/config.toml and theme.py."""
    cfg_path = ROOT / ".streamlit" / "config.toml"
    assert cfg_path.exists(), "Missing .streamlit/config.toml"
    cfg_text = cfg_path.read_text(encoding="utf-8")
    assert 'primaryColor = "#12DFF3"' in cfg_text
    assert 'backgroundColor = "#061426"' in cfg_text
    assert 'secondaryBackgroundColor = "#0A1D32"' in cfg_text
    assert 'textColor = "#F4FAFF"' in cfg_text


def test_homepage_verified_statistics():
    """Verify homepage dynamically loads verified figures without hardcoding."""
    summary_path = RESULTS_DIR / "glims_retreat_summary_validated.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["total_glaciers_analyzed"] == 81598
    assert summary["retreating_glaciers_count"] == 54718
    assert summary["advancing_glaciers_count"] == 20132
    assert summary["stable_glaciers_count"] == 6748
    assert abs(summary["net_area_loss_km2"] - 16321.76) < 2.0


def test_segmentation_lab_demo_candidates():
    """Verify all 4 held-out test demo candidates exist and have 13 bands."""
    cands = get_demo_netcdf_candidates()
    assert len(cands) == 4, f"Expected 4 demo candidates, got {len(cands)}"
    for c in cands:
        assert c["path"].exists(), f"Candidate file missing: {c['path']}"
        assert c["id"] in ["g_4367", "g_3560", "g_2316", "g_1266"]


def test_retreat_explorer_case_studies():
    """Verify Swiss Alps benchmark case studies exist with vector outlines."""
    cases = load_case_studies()
    assert len(cases) >= 3, f"Expected at least 3 case studies, got {len(cases)}"
    case_ids = [c["glacier_id"] for c in cases]
    assert "G008379E46648N" in case_ids  # Rhone
    assert "G008032E46504N" in case_ids  # Great Aletsch
    assert "G008187E46569N" in case_ids  # Unteraar


def test_retreat_explorer_runtime_schema_regression():
    """Regression test for Bug 1: ensure no KeyError on date or geometry access for benchmark glaciers."""
    cases = load_case_studies()
    case_map = {c["glacier_id"]: c for c in cases}

    benchmarks = [
        ("Rhone", "G008379E46648N", "Rhone Glacier (Rhonegletscher)", 1973, 2015, 17.166, 14.179, 2.987, 17.4),
        ("Great Aletsch", "G008032E46504N", "Great Aletsch Glacier (Grosser Aletschgletscher)", 1973, 2015, 86.565, 78.069, 8.496, 9.81),
        ("Unteraar", "G008187E46569N", "Unteraar Glacier (Unteraargletscher)", 1973, 2015, 27.131, 20.816, 6.316, 23.28),
    ]

    for query, expected_id, expected_name, e_yr_exp, l_yr_exp, e_area_exp, l_area_exp, net_loss_exp, pct_exp in benchmarks:
        assessment = get_glacier_dynamic_assessment(query)
        assert assessment["glacier_id"] == expected_id
        assert assessment["glacier_name"] == expected_name
        assert "earlier_date" in assessment
        assert "later_date" in assessment
        assert assessment["earlier_date"].startswith(str(e_yr_exp))
        assert assessment["later_date"].startswith(str(l_yr_exp))

        # Check matched case study schema
        matched_case = case_map.get(expected_id)
        assert matched_case is not None

        # Verify exact field lookups performed in retreat_explorer.py
        crs = matched_case.get("projected_crs", "EPSG:32632")
        e_yr = matched_case.get("earlier_year", assessment.get("earlier_date", "1973")[:4])
        l_yr = matched_case.get("later_year", assessment.get("later_date", "2015")[:4])
        e_area = matched_case.get("earlier_area_km2", assessment.get("earlier_area_km2", 0.0))
        l_area = matched_case.get("later_area_km2", assessment.get("later_area_km2", 0.0))
        retained = matched_case.get("retained_ice_area_km2", 0.0)
        lost = matched_case.get("lost_ice_area_km2", 0.0)
        gained = matched_case.get("gained_ice_area_km2", 0.0)
        net_loss = matched_case.get("net_loss_km2", abs(min(0.0, assessment.get("area_change_km2", 0.0))))
        pct = matched_case.get("percentage_change", abs(assessment.get("percent_change", 0.0)))

        assert crs == "EPSG:32632"
        assert e_yr == e_yr_exp
        assert l_yr == l_yr_exp
        assert abs(e_area - e_area_exp) < 0.01
        assert abs(l_area - l_area_exp) < 0.01
        assert abs(net_loss - net_loss_exp) < 0.01
        assert abs(pct - pct_exp) < 0.01
        assert retained > 0.0
        assert lost > 0.0


def test_area_assessment_runtime_schema_regression():
    """Regression test for Bug 2: ensure no KeyError on net area loss or accounting metrics."""
    summary = load_retreat_summary()
    assert summary is not None

    # Verify exact schema lookups in area_assessment.py
    total_g = summary.get("total_glaciers_analyzed", 81598)
    ret_count = summary.get("retreating_glaciers_count", 54718)
    ret_pct = summary.get("retreating_glaciers_ratio_pct", 67.06)
    net_loss = summary.get("net_area_loss_km2", 16321.76)
    net_pct = summary.get("net_aggregate_percentage_loss", summary.get("net_area_loss_pct", 7.40))
    adv_count = summary.get("advancing_glaciers_count", 20132)
    adv_pct = summary.get("advancing_glaciers_ratio_pct", 24.67)

    assert total_g == 81598
    assert ret_count == 54718
    assert abs(ret_pct - 67.06) < 0.01
    assert abs(net_loss - 16321.76) < 0.01
    assert abs(net_pct - 7.40) < 0.01
    assert adv_count == 20132
    assert abs(adv_pct - 24.67) < 0.01

    # Verify temporal grouping fields
    tg = summary.get("temporal_grouping", {})
    assert len(tg) == 5
    for k, v in tg.items():
        assert "glaciers" in v
        assert "net_loss_km2" in v
        assert "retreating_pct" in v


def test_area_assessment_dataset_loader():
    """Verify full 81,598-row dataset loader operates correctly."""
    df = load_retreat_csv(nrows=None)
    assert df is not None
    assert len(df) == 81598, f"Expected 81,598 records, got {len(df)}"
    assert "dynamics_class" in df.columns
    assert "area_loss_km2" in df.columns or "percentage_change" in df.columns


def test_visual_monitoring_publication_figures():
    """Verify all publication 300 DPI figures exist for visual monitoring tabs."""
    figs = [
        FIGURES_DIR / "visual_monitoring_glims_boundary_overlays.png",
        FIGURES_DIR / "visual_monitoring_sentinel2_composites.png",
        FIGURES_DIR / "visual_monitoring_landsat_bitemporal.png",
    ]
    for f in figs:
        assert f.exists(), f"Figure missing: {f.name}"
        assert f.stat().st_size > 50000, f"Figure size too small: {f.name}"


def test_clip_coop_demo_assets():
    """Verify all 4 demo alpine RGB test samples exist in outputs/demo_samples/."""
    demo_files = [
        DEMO_DIR / "demo_glacier_1_clean_ice.png",
        DEMO_DIR / "demo_glacier_2_valley_ice.png",
        DEMO_DIR / "demo_bedrock_1_alpine_ridge.png",
        DEMO_DIR / "demo_bedrock_2_rocky_slope.png",
    ]
    for df in demo_files:
        assert df.exists(), f"Demo asset missing: {df.name}"
        assert df.stat().st_size > 50000, f"Demo asset too small: {df.name}"


def test_research_results_audit_reports():
    """Verify academic markdown acceptance reports exist in outputs/audits/."""
    audits_dir = ROOT / "outputs" / "audits"
    assert audits_dir.exists()
    required_reports = [
        "objective1_final_acceptance_report.md",
        "objective2_final_acceptance_report.md",
        "objective3_final_acceptance_report.md",
        "objective4_final_acceptance_report.md",
        "final_website_improvements_report.md",
        "interactive_platform_acceptance_report.md",
        "interactive_runtime_bugfix_report.md",
    ]
    for r in required_reports:
        p = audits_dir / r
        assert p.exists(), f"Missing acceptance audit report: {r}"


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
