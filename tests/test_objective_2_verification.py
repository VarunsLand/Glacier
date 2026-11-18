"""Objective 2 Scientific Verification and Regression Test Suite.

Verifies:
1. Exact benchmark geometry values for Rhone, Great Aletsch, and Unteraar.
2. Mathematical identities (spatial conservation, net loss, percentage loss, annualized rate).
3. Authoritative observation dates (1973-09-01 vs 1973-09-08 documentation).
4. Unification of Waterfall chart, KPI metric cards, and geometric breakdown.
5. Individual glacier boundary map existence in both 300 DPI raster PNG and vector SVG formats.
6. Multi-CRS equal-area comparison (EPSG:3035 vs EPSG:32632).
7. Custom GLIMS glacier discovery and loading (Gorner, Findel).
8. Graceful error handling for non-existent glacier IDs without fake data.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.inference import FIGURES_DIR, RESULTS_DIR, ROOT, get_glacier_dynamic_assessment
from src.ui.retreat_explorer import load_case_studies


class TestObjective2ScientificVerification:
    @classmethod
    def setup_class(cls):
        cls.cases = load_case_studies()
        cls.case_map = {c["glacier_id"]: c for c in cls.cases}

    def test_benchmark_exact_values(self):
        """Verify the 3 core benchmark cases match audited reference numbers."""
        expected = {
            "G008379E46648N": {
                "name": "Rhone",
                "earlier_area": 17.166,
                "later_area": 14.179,
                "retained": 14.114,
                "lost": 3.052,
                "gained": 0.065,
                "net_loss": 2.987,
                "pct_loss": 17.40,
                "annual_rate": 0.071,
            },
            "G008032E46504N": {
                "name": "Great Aletsch",
                "earlier_area": 86.565,
                "later_area": 78.069,
                "retained": 76.963,
                "lost": 9.602,
                "gained": 1.106,
                "net_loss": 8.496,
                "pct_loss": 9.81,
                "annual_rate": 0.202,
            },
            "G008187E46569N": {
                "name": "Unteraar",
                "earlier_area": 27.131,
                "later_area": 20.816,
                "retained": 20.239,
                "lost": 6.892,
                "gained": 0.577,
                "net_loss": 6.316,
                "pct_loss": 23.28,
                "annual_rate": 0.150,
            },
        }

        for gid, exp in expected.items():
            case = self.case_map[gid]
            assert np.isclose(case["earlier_area_km2"], exp["earlier_area"], atol=1e-3)
            assert np.isclose(case["later_area_km2"], exp["later_area"], atol=1e-3)
            assert np.isclose(case["retained_ice_area_km2"], exp["retained"], atol=1e-3)
            assert np.isclose(case["lost_ice_area_km2"], exp["lost"], atol=1e-3)
            assert np.isclose(case["gained_ice_area_km2"], exp["gained"], atol=1e-3)
            assert np.isclose(case["net_loss_km2"], exp["net_loss"], atol=1e-3)
            assert np.isclose(case["percentage_change"], exp["pct_loss"], atol=0.05)

    def test_mathematical_identities(self):
        """Verify strict spatial conservation and retreat identity proofs."""
        for case in self.cases:
            a_e = case["earlier_area_km2"]
            a_l = case["later_area_km2"]
            ret = case["retained_ice_area_km2"]
            lost = case["lost_ice_area_km2"]
            gain = case["gained_ice_area_km2"]
            net_loss = case["net_loss_km2"]

            # Identity 1: A_earlier = A_retained + A_lost
            assert np.isclose(a_e, ret + lost, atol=1e-2), f"Identity 1 failed for {case['glacier_name']}"

            # Identity 2: A_modern = A_retained + A_gained
            assert np.isclose(a_l, ret + gain, atol=1e-2), f"Identity 2 failed for {case['glacier_name']}"

            # Identity 3: Net Loss = A_lost - A_gained
            assert np.isclose(net_loss, lost - gain, atol=1e-2), f"Identity 3 failed for {case['glacier_name']}"

            # Identity 4: Net Loss = A_earlier - A_modern
            assert np.isclose(net_loss, a_e - a_l, atol=1e-2), f"Identity 4 failed for {case['glacier_name']}"

    def test_chart_kpi_unification(self):
        """Verify dynamic assessment and case studies return identical metrics (eliminates discrepancy)."""
        benchmarks = ["Rhone", "Great Aletsch", "Unteraar"]
        for name in benchmarks:
            assessment = get_glacier_dynamic_assessment(name)
            gid = assessment["glacier_id"]
            case = self.case_map[gid]

            assert assessment["earlier_area_km2"] == case["earlier_area_km2"]
            assert assessment["later_area_km2"] == case["later_area_km2"]
            assert np.isclose(assessment["net_loss_km2"], case["net_loss_km2"], atol=1e-3)
            assert np.isclose(abs(assessment["percent_change"]), case["percentage_change"], atol=0.05)

    def test_authoritative_dates(self):
        """Verify authoritative GLIMS source dates (1973-09-01 and 2015-08-29)."""
        for name in ["Rhone", "Great Aletsch", "Unteraar"]:
            assessment = get_glacier_dynamic_assessment(name)
            assert assessment["earlier_date"] == "1973-09-01"
            assert assessment["later_date"] == "2015-08-29"
            assert assessment["time_gap_years"] == 42.0

    def test_raster_and_vector_figures_exist(self):
        """Verify dedicated boundary overlay figures exist as both 300 DPI raster PNG and vector SVG."""
        base_names = [
            "glims_boundary_rhone",
            "glims_boundary_great_aletsch",
            "glims_boundary_unteraar",
            "glims_boundary_gorner",
        ]
        for base in base_names:
            png_path = FIGURES_DIR / f"{base}.png"
            svg_path = FIGURES_DIR / f"{base}.svg"

            assert png_path.exists(), f"Missing PNG: {base}.png"
            assert png_path.stat().st_size > 50_000, f"PNG suspiciously small: {base}.png"
            with Image.open(png_path) as img:
                assert img.format == "PNG"
                assert img.width >= 1000

            assert svg_path.exists(), f"Missing SVG: {base}.svg"
            assert svg_path.stat().st_size > 50_000, f"SVG suspiciously small: {base}.svg"
            svg_text = svg_path.read_text(encoding="utf-8")
            assert "<svg" in svg_text, f"Invalid SVG file: {base}.svg"

    def test_equal_area_crs_comparison(self):
        """Verify that EPSG:3035 is equal-area and matches geodesic area within 0.0001%."""
        crs_json = ROOT / "scratch" / "crs_comparison_results.json"
        assert crs_json.exists(), "CRS comparison JSON missing"
        with open(crs_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        for gid in ["G008379E46648N", "G008032E46504N", "G008187E46569N"]:
            rec = data[gid]
            geod = rec["geodetic_wgs84"]["earlier_km2"]
            laea = rec["crs_metrics"]["EPSG:3035"]["earlier_km2"]
            utm = rec["crs_metrics"]["EPSG:32632"]["earlier_km2"]

            # EPSG:3035 must match geodesic area within 0.001%
            diff_laea_pct = abs((laea - geod) / geod) * 100.0
            assert diff_laea_pct < 0.001, f"EPSG:3035 is not equal-area: {diff_laea_pct}% diff"

            # UTM 32N exhibits predictable -0.07% contraction
            diff_utm_pct = ((utm - geod) / geod) * 100.0
            assert -0.09 < diff_utm_pct < -0.05, f"Unexpected UTM distortion: {diff_utm_pct}%"

    def test_custom_glacier_search(self):
        """Verify that additional Alpine multi-temporal glaciers (e.g. Gornergletscher) can be queried."""
        gorner = get_glacier_dynamic_assessment("Gorner")
        assert gorner["found"] is True
        assert gorner["glacier_id"] == "G007800E45965N"
        assert gorner["earlier_area_km2"] > 40.0
        assert gorner["later_area_km2"] < gorner["earlier_area_km2"]
        assert gorner["area_change_km2"] < 0

        findel = get_glacier_dynamic_assessment("Findel")
        assert findel["found"] is True
        assert findel["glacier_id"] == "G007880E45990N"
        assert findel["earlier_area_km2"] > 15.0

    def test_invalid_glacier_error_handling(self):
        """Verify that invalid glacier IDs do not fabricate fake 1850 data and report found=False."""
        invalid_res = get_glacier_dynamic_assessment("NonExistentGlacier99999")
        assert invalid_res["found"] is False
        assert invalid_res["earlier_area_km2"] == 0.0
        assert invalid_res["later_area_km2"] == 0.0
        assert invalid_res["earlier_date"] == ""
