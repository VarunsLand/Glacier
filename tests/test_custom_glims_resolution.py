import pytest
from pathlib import Path
from src.inference import get_glacier_dynamic_assessment, FIGURES_DIR


def test_custom_glims_mazia_resolution():
    """Verify exact resolution and metrics for custom glacier G010725E46789N (Mazia / Matscher)."""
    res = get_glacier_dynamic_assessment("G010725E46789N")
    assert res["found"] is True
    assert res["glacier_id"] == "G010725E46789N"
    assert "Mazia" in res["glacier_name"] or "Matscher" in res["glacier_name"]
    assert res["earlier_date"] == "2000-08-22"
    assert res["later_date"] == "2016-09-29"
    assert res["earlier_analysis_id"] == "1031480"
    assert res["later_analysis_id"] == "709596"
    assert round(res["earlier_area_km2"], 3) == 2.493
    assert round(res["later_area_km2"], 3) == 2.310
    assert round(res["net_loss_km2"], 3) == 0.184
    assert round(res["percent_change"], 2) == -7.37

    # Verify dedicated boundary figure exists
    fig_png = FIGURES_DIR / "glims_boundary_G010725E46789N.png"
    fig_svg = FIGURES_DIR / "glims_boundary_G010725E46789N.svg"
    assert fig_png.exists()
    assert fig_svg.exists()


def test_custom_glims_without_prerendered_map():
    """Verify that a valid GLIMS ID without a pre-rendered map resolves metrics without error."""
    res = get_glacier_dynamic_assessment("G057419E80930N")
    assert res["found"] is True
    assert res["glacier_id"] == "G057419E80930N"
    assert res["earlier_analysis_id"] == "120555"
    assert res["earlier_area_km2"] > 0
    assert res["later_area_km2"] > 0
    assert res["area_change_km2"] != 0


def test_invalid_glims_id_no_silent_fallback():
    """Verify that an invalid GLIMS ID returns found=False and never silently substitutes Rhone."""
    res = get_glacier_dynamic_assessment("INVALID_TEST_GLACIER_99999")
    assert res["found"] is False
    assert res["glacier_id"] == "INVALID_TEST_GLACIER_99999"
    assert "G008379E46648N" not in res["glacier_id"]
    assert "Rhone" not in res["glacier_name"]
    assert res["earlier_area_km2"] == 0.0
    assert res["later_area_km2"] == 0.0


def test_all_benchmarks_have_distinct_overlays():
    """Verify that each benchmark has its own unique, non-fallback boundary figure."""
    benchmarks = {
        "Rhone": ("G008379E46648N", "glims_boundary_rhone.png"),
        "Great Aletsch": ("G008032E46504N", "glims_boundary_great_aletsch.png"),
        "Unteraar": ("G008187E46569N", "glims_boundary_unteraar.png"),
        "Gorner": ("G007800E45965N", "glims_boundary_gorner.png"),
        "Findel": ("G007880E45990N", "glims_boundary_findel.png"),
        "Mazia": ("G010725E46789N", "glims_boundary_G010725E46789N.png"),
    }
    for name, (gid, fig_file) in benchmarks.items():
        res = get_glacier_dynamic_assessment(gid)
        assert res["found"] is True
        assert (FIGURES_DIR / fig_file).exists()


def test_findel_unification_and_identities():
    """Verify Findelgletscher (G007880E45990N) metrics unification in EPSG:32632 and geometric conservation."""
    res = get_glacier_dynamic_assessment("G007880E45990N")
    assert res["found"] is True
    assert res["is_case_study"] is True
    assert round(res["earlier_area_km2"], 3) == 18.603
    assert round(res["later_area_km2"], 3) == 16.072
    assert round(res["net_loss_km2"], 3) == 2.531
    assert round(res["percent_change"], 2) == -13.60
    assert res["earlier_analysis_id"] == "342320"
    assert res["later_analysis_id"] == "1096563"

    # Verify unrounded spatial conservation identities from case studies
    from src.ui.retreat_explorer import load_case_studies
    cases = load_case_studies()
    findel_case = next(c for c in cases if c["glacier_id"] == "G007880E45990N")
    ae = findel_case["earlier_area_km2"]
    al = findel_case["later_area_km2"]
    ret = findel_case["retained_ice_area_km2"]
    lost = findel_case["lost_ice_area_km2"]
    gain = findel_case["gained_ice_area_km2"]
    net = findel_case["net_loss_km2"]

    assert abs(ret + lost - ae) < 0.002
    assert abs(ret + gain - al) < 0.002
    assert abs(lost - gain - net) < 0.002

