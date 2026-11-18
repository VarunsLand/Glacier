"""Automated Validation Test Suite for GlacierVision Objectives 2, 3, and 4.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"


# =====================================================================
# OBJECTIVE 2: GLACIER RETREAT DETECTION TESTS
# =====================================================================
class TestObjective2RetreatDetection:
    @classmethod
    def setup_class(cls):
        csv_path = RESULTS_DIR / "glims_retreat_assessment_validated.csv"
        assert csv_path.exists(), f"Validated CSV not found: {csv_path}"
        cls.df = pd.read_csv(csv_path)

        summary_path = RESULTS_DIR / "glims_retreat_summary_validated.json"
        assert summary_path.exists(), f"Summary JSON not found: {summary_path}"
        with open(summary_path, "r", encoding="utf-8") as f:
            cls.summary = json.load(f)

        cases_path = RESULTS_DIR / "glims_case_studies.json"
        assert cases_path.exists(), f"Cases JSON not found: {cases_path}"
        with open(cases_path, "r", encoding="utf-8") as f:
            cls.cases = json.load(f)

    def test_glacier_identity_matching(self):
        """Verify glacier IDs are non-null and match GLIMS ID standard format."""
        assert self.df["glacier_id"].notna().all()
        assert len(self.df) == self.df["glacier_id"].nunique(), "Duplicate glacier IDs found!"
        valid_ids = self.df["glacier_id"].str.match(r"^G\d{6}[EW]\d{5}[NS]$")
        assert valid_ids.mean() > 0.999, "Glacier IDs do not match GLIMS standard syntax."

    def test_observation_dates_filtering(self):
        """Verify dates are valid ISO format, satellite era, and exclude Jan 1 placeholders."""
        t1 = pd.to_datetime(self.df["earlier_date"], errors="coerce")
        t2 = pd.to_datetime(self.df["later_date"], errors="coerce")
        assert t1.notna().all()
        assert t2.notna().all()
        assert (t1.dt.year >= 1970).all()
        assert (t2.dt.year <= 2024).all()
        assert not ((t1.dt.month == 1) & (t1.dt.day == 1)).any(), "Jan 1 placeholder date found in t1!"
        assert not ((t2.dt.month == 1) & (t2.dt.day == 1)).any(), "Jan 1 placeholder date found in t2!"

    def test_temporal_interval_constraint(self):
        """Verify Delta t >= 3.0 years for multi-decadal glacier retreat signal."""
        assert (self.df["time_gap_years"] >= 3.0).all()
        assert (self.df["time_gap_years"] <= 50.0).all()

    def test_rejection_of_unverified_naive_loss(self):
        """Confirm that the flawed 21,758.94 km^2 figure is rejected in favor of validated 16,321.74 km^2."""
        net_loss = self.summary.get("net_area_loss_km2") or self.summary["mass_balance"]["net_area_loss_km2"]
        assert net_loss != 21758.94
        assert np.isclose(net_loss, 16321.76, atol=2.0)

    def test_mutually_exclusive_classification_partition(self):
        """Verify that classifications are strictly mutually exclusive and sum to total glaciers."""
        N = len(self.df)
        dyn = self.summary["dynamics_breakdown"]
        dyn_sum = sum(dyn.values())
        assert dyn_sum == N, f"Dynamics breakdown sum ({dyn_sum}) != total glaciers ({N})"
        
        # 3-class partition with 2% deadband:
        sig_ret = dyn.get("Significant Retreat", 0)
        mod_ret = dyn.get("Moderate Retreat", 0)
        tot_ret = sig_ret + mod_ret
        adv = dyn.get("Advance / Growth", 0)
        sta = dyn.get("Stable / Equilibrium", 0)
        assert tot_ret + adv + sta == N, "3-class partition sum != total glaciers"
        
        # Zero-threshold partition:
        strict_loss = (self.df["area_loss_km2"] > 0).sum()
        strict_gain = (self.df["area_loss_km2"] < 0).sum()
        strict_zero = (self.df["area_loss_km2"] == 0).sum()
        assert strict_loss + strict_gain + strict_zero == N, "Zero-threshold sum != total glaciers"

    def test_case_studies_geometry_conservation(self):
        """Verify metric area conservation in geometric overlay cases."""
        assert len(self.cases) >= 3
        for c in self.cases:
            a1 = c["earlier_area_km2"]
            a2 = c["later_area_km2"]
            retained = c["retained_ice_area_km2"]
            lost = c["lost_ice_area_km2"]
            gained = c["gained_ice_area_km2"]
            assert np.isclose(a1, retained + lost, atol=0.05), f"Conservation failed for {c['glacier_name']} t1"
            assert np.isclose(a2, retained + gained, atol=0.05), f"Conservation failed for {c['glacier_name']} t2"


# =====================================================================
# OBJECTIVE 3: QUANTITATIVE AREA ASSESSMENT TESTS
# =====================================================================
class TestObjective3AreaAssessment:
    @classmethod
    def setup_class(cls):
        csv_path = RESULTS_DIR / "glims_retreat_assessment_validated.csv"
        cls.df = pd.read_csv(csv_path)
        summary_path = RESULTS_DIR / "glims_retreat_summary_validated.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            cls.summary = json.load(f)

    def test_area_loss_and_percentage_formulae(self):
        """Verify mathematical consistency of area loss and percentage change."""
        a1 = self.df["earlier_area_km2"]
        a2 = self.df["later_area_km2"]
        expected_loss = (a1 - a2).round(4)
        assert np.isclose(self.df["area_loss_km2"], expected_loss, atol=1e-4).all()

        expected_pct = (((a1 - a2) / a1) * 100.0).round(2)
        assert np.isclose(self.df["percentage_change"], expected_pct, atol=1e-2).all()

    def test_annualized_rates_formula(self):
        """Verify annualized retreat rates (km^2/yr and %/yr)."""
        dt = self.df["time_gap_years"]
        expected_ann_loss = (self.df["area_loss_km2"] / dt).round(5)
        assert np.isclose(self.df["annual_loss_km2_yr"], expected_ann_loss, atol=1e-4).all()

    def test_area_ratio_plausibility_bounds(self):
        """Verify no impossible ratios (e.g. 99.9% collapse due to single-sliver drop_duplicates)."""
        ratios = self.df["later_area_km2"] / self.df["earlier_area_km2"]
        assert (ratios >= 0.149).all(), "Implausible shrinkage ratio (< 0.149) detected!"
        assert (ratios <= 3.001).all(), "Implausible growth ratio (> 3.001) detected!"

    def test_aggregate_mass_balance(self):
        """Verify gross loss, gross gain, and net loss consistency."""
        gross_loss = self.summary.get("gross_area_lost_km2") or self.summary["mass_balance"]["gross_area_lost_km2"]
        gross_gain = self.summary.get("gross_area_gained_km2") or self.summary["mass_balance"]["gross_area_gained_km2"]
        net_loss = self.summary.get("net_area_loss_km2") or self.summary["mass_balance"]["net_area_loss_km2"]
        assert np.isclose(gross_loss - gross_gain, net_loss, atol=0.1)

    def test_filter_sensitivity_results(self):
        """Verify filter sensitivity results file existence and stability."""
        sens_path = RESULTS_DIR / "filter_sensitivity_analysis.json"
        assert sens_path.exists(), "Sensitivity analysis JSON missing!"
        with open(sens_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "fragment_threshold_sensitivity" in data
        assert "ratio_bounds_sensitivity" in data
        # Confirm that retreat percentage remains stable across all reasonable thresholds (within 65%-68%)
        for run in data["fragment_threshold_sensitivity"]:
            assert 65.0 <= run["retreating_pct"] <= 68.0


# =====================================================================
# OBJECTIVE 4: VISUAL MONITORING SUITE TESTS
# =====================================================================
class TestObjective4VisualMonitoring:
    def test_figure_existence_and_integrity(self):
        """Verify publication figures exist, are non-empty, and valid PNGs."""
        figs = [
            FIGURES_DIR / "visual_monitoring_sentinel2_composites.png",
            FIGURES_DIR / "visual_monitoring_landsat_bitemporal.png",
            FIGURES_DIR / "visual_monitoring_glims_boundary_overlays.png",
            FIGURES_DIR / "glims_area_change_distribution.png",
            FIGURES_DIR / "glims_top_retreating_glaciers.png",
            FIGURES_DIR / "filter_sensitivity_curves.png",
        ]
        for f in figs:
            assert f.exists(), f"Figure missing: {f.name}"
            assert f.stat().st_size > 50_000, f"Figure too small / corrupt: {f.name}"
            with Image.open(f) as img:
                assert img.format == "PNG"
                assert img.width >= 1000, f"Figure width insufficient: {f.name}"
                assert img.height >= 500, f"Figure height insufficient: {f.name}"

    def test_landsat_labeling_disclaimer(self):
        """Confirm that script distinguishes undirected change mask from directed retreat."""
        script_path = ROOT / "scripts" / "generate_visual_monitoring.py"
        content = script_path.read_text(encoding="utf-8")
        assert "Surface Change Mask" in content
        assert "Undirected Binary Difference" in content
        assert "Scientific Note" in content
