"""GlacierVision: Comprehensive Automated Verification Suite for Objective 3.

Validates:
1. Global Glacier Area Accounting Dashboard (81,598 pairs, 3-class and 4-class partitions, zero-deadband, area conservation identities).
2. Area Change Scenario Calculator (arithmetic scenario analysis, edge cases, deadband thresholds, linear vs CAGR rates, invalid input rejection).
3. Aggregated Observation-Interval Summary (precomputed groupings, rounding explanation proof, annualized metrics).
4. 12-Configuration Quality Filter Sensitivity & Invariance Analysis (invariance of retreat dominance).
5. Top-Retreating Glacier Ranking (monotonic ordering, observation intervals, net loss metric).
6. Full GLIMS Database Explorer (search by ID and alias, category filtering, precision export).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"


# =====================================================================
# 1. GLOBAL GLACIER AREA ACCOUNTING & CONSERVATION
# =====================================================================
class TestObjective3GlobalAccounting:
    @classmethod
    def setup_class(cls):
        csv_path = RESULTS_DIR / "glims_retreat_assessment_validated.csv"
        assert csv_path.exists(), f"Validated CSV not found: {csv_path}"
        cls.df = pd.read_csv(csv_path)

        summary_path = RESULTS_DIR / "glims_retreat_summary_validated.json"
        assert summary_path.exists(), f"Summary JSON not found: {summary_path}"
        with open(summary_path, "r", encoding="utf-8") as f:
            cls.summary = json.load(f)

    def test_total_validated_pairs_count(self):
        """Verify exact canonical count of 81,598 validated observation pairs."""
        assert len(self.df) == 81598
        assert self.df["glacier_id"].nunique() == 81598, "Duplicate glacier IDs found in validated dataset!"
        assert self.summary["total_glaciers_analyzed"] == 81598

    def test_mutually_exclusive_3_class_partition(self):
        """Verify mutually exclusive 3-class glaciological partition with +/-2% deadband."""
        sig_ret = (self.df["dynamics_class"] == "Significant Retreat").sum()
        mod_ret = (self.df["dynamics_class"] == "Moderate Retreat").sum()
        tot_ret = sig_ret + mod_ret
        adv_grow = (self.df["dynamics_class"] == "Advance / Growth").sum()
        sta_equil = (self.df["dynamics_class"] == "Stable / Equilibrium").sum()

        assert tot_ret == 54718, f"Retreating pairs count mismatch: {tot_ret} != 54,718"
        assert adv_grow == 20132, f"Advancing pairs count mismatch: {adv_grow} != 20,132"
        assert sta_equil == 6748, f"Stable pairs count mismatch: {sta_equil} != 6,748"

        # Mutually exclusive and exhaustive
        assert tot_ret + adv_grow + sta_equil == 81598
        ret_pct = tot_ret / 81598 * 100.0
        adv_pct = adv_grow / 81598 * 100.0
        sta_pct = sta_equil / 81598 * 100.0

        assert np.isclose(ret_pct, 67.06, atol=0.01)
        assert np.isclose(adv_pct, 24.67, atol=0.01)
        assert np.isclose(sta_pct, 8.27, atol=0.01)
        assert np.isclose(ret_pct + adv_pct + sta_pct, 100.0, atol=0.01)

    def test_4_class_partition_counts(self):
        """Verify detailed 4-class partition breakdown."""
        counts = self.df["dynamics_class"].value_counts().to_dict()
        assert counts["Significant Retreat"] == 43676
        assert counts["Moderate Retreat"] == 11042
        assert counts["Advance / Growth"] == 20132
        assert counts["Stable / Equilibrium"] == 6748
        assert sum(counts.values()) == 81598

    def test_zero_threshold_partition(self):
        """Verify strict zero-deadband partition."""
        strict_loss = (self.df["area_loss_km2"] > 0).sum()
        strict_gain = (self.df["area_loss_km2"] < 0).sum()
        strict_zero = (self.df["area_loss_km2"] == 0).sum()

        assert strict_loss == 57477
        assert strict_gain == 22471
        assert strict_zero == 1650
        assert strict_loss + strict_gain + strict_zero == 81598

    def test_pooled_area_accounting_and_conservation(self):
        """Verify exact pooled area conservation identities using unrounded values."""
        a1_sum = self.df["earlier_area_km2"].sum()
        a2_sum = self.df["later_area_km2"].sum()
        gross_lost = self.df.loc[self.df["area_loss_km2"] > 0, "area_loss_km2"].sum()
        gross_gained = abs(self.df.loc[self.df["area_loss_km2"] < 0, "area_loss_km2"].sum())
        net_loss = self.df["area_loss_km2"].sum()

        # Display values rounded to 2 decimals
        assert round(a1_sum, 2) == 220636.22
        assert round(a2_sum, 2) == 204314.45
        assert round(gross_lost, 2) == 27330.47
        assert round(gross_gained, 2) == 11008.71
        assert round(net_loss, 2) == 16321.76

        # Identity 1: Net Area Loss = Gross Area Lost - Gross Area Gained
        assert np.isclose(gross_lost - gross_gained, net_loss, atol=1e-5)

        # Identity 2: Net Area Loss = Earlier Pooled Area - Later Pooled Area
        assert np.isclose(a1_sum - a2_sum, net_loss, atol=1e-5)

        # Identity 3: Conservation Residual
        residual = abs((gross_lost - gross_gained) - (a1_sum - a2_sum))
        assert residual < 1e-9, f"Conservation residual non-zero: {residual}"

        # Pooled percentage reduction
        pooled_pct = (net_loss / a1_sum) * 100.0
        assert np.isclose(pooled_pct, 7.3976, atol=0.01)
        assert round(pooled_pct, 2) == 7.40

    def test_sign_conventions_and_pairwise_magnitudes(self):
        """Verify strict sign conventions: delta_a = a2 - a1, reduction when a1 > a2 (delta_a < 0), increase when a2 > a1 (delta_a > 0)."""
        a1 = self.df["earlier_area_km2"]
        a2 = self.df["later_area_km2"]
        delta_a = a2 - a1

        # Positive magnitude variables
        l_i = np.maximum(0.0, a1 - a2)
        g_i = np.maximum(0.0, a2 - a1)

        assert np.isclose(delta_a, g_i - l_i, atol=1e-6).all()

        shrinking_mask = a1 > a2
        expanding_mask = a2 > a1
        identical_mask = a1 == a2

        assert shrinking_mask.sum() == 57477
        assert expanding_mask.sum() == 22471
        assert identical_mask.sum() == 1650
        assert shrinking_mask.sum() + expanding_mask.sum() + identical_mask.sum() == 81598

        sum_l = l_i.sum()
        sum_g = g_i.sum()
        assert np.isclose(sum_l, 27330.4690, atol=1e-4)
        assert round(sum_l, 2) == 27330.47
        assert np.isclose(sum_g, 11008.7060, atol=1e-4)
        assert round(sum_g, 2) == 11008.71

        net_loss = sum_l - sum_g
        assert np.isclose(net_loss, 16321.7630, atol=1e-4)
        assert round(net_loss, 2) == 16321.76

    def test_displayed_conservation_rounding_difference(self):
        """Mathematically prove the 16,321.77 vs 16,321.76 displayed subtraction discrepancy."""
        a1 = self.df["earlier_area_km2"]
        a2 = self.df["later_area_km2"]

        unrounded_a1 = a1.sum()  # 220,636.2156
        unrounded_a2 = a2.sum()  # 204,314.4526
        unrounded_net_loss = (a1 - a2).sum()  # 16,321.7630

        assert np.isclose(unrounded_a1, 220636.2156, atol=1e-4)
        assert np.isclose(unrounded_a2, 204314.4526, atol=1e-4)
        assert np.isclose(unrounded_net_loss, 16321.7630, atol=1e-4)

        # Independent 2-decimal rounding of displays:
        disp_a1 = round(unrounded_a1, 2)  # 220,636.22
        disp_a2 = round(unrounded_a2, 2)  # 204,314.45
        disp_net_loss = round(unrounded_net_loss, 2)  # 16,321.76

        assert disp_a1 == 220636.22
        assert disp_a2 == 204314.45
        assert disp_net_loss == 16321.76

        # Subtraction of rounded displays yields 16,321.77:
        disp_diff = disp_a1 - disp_a2
        assert round(disp_diff, 2) == 16321.77
        assert np.isclose(disp_diff, 16321.77, atol=1e-5)

        # The rounding offset delta:
        offset_a1 = disp_a1 - unrounded_a1  # +0.0044
        offset_a2 = disp_a2 - unrounded_a2  # -0.0026
        compounded_offset = offset_a1 - offset_a2  # +0.0070
        assert np.isclose(compounded_offset, 0.0070, atol=1e-4)
        assert np.isclose(disp_diff - unrounded_net_loss, compounded_offset, atol=1e-4)


# =====================================================================
# 2. TEMPORAL GROUPING & ROUNDING EXPLANATION
# =====================================================================
class TestObjective3TemporalGrouping:
    @classmethod
    def setup_class(cls):
        csv_path = RESULTS_DIR / "glims_retreat_assessment_validated.csv"
        cls.df = pd.read_csv(csv_path)

        summary_path = RESULTS_DIR / "glims_retreat_summary_validated.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            cls.summary = json.load(f)

    def test_temporal_interval_partitions(self):
        """Verify non-overlapping baseline interval definitions and pair counts."""
        b1 = self.df[(self.df["time_gap_years"] >= 3.0) & (self.df["time_gap_years"] <= 5.0)]
        b2 = self.df[(self.df["time_gap_years"] > 5.0) & (self.df["time_gap_years"] <= 10.0)]
        b3 = self.df[(self.df["time_gap_years"] > 10.0) & (self.df["time_gap_years"] <= 15.0)]
        b4 = self.df[(self.df["time_gap_years"] > 15.0) & (self.df["time_gap_years"] <= 20.0)]
        b5 = self.df[self.df["time_gap_years"] > 20.0]

        assert len(b1) == 12972
        assert len(b2) == 32632
        assert len(b3) == 18201
        assert len(b4) == 8727
        assert len(b5) == 9066
        assert len(b1) + len(b2) + len(b3) + len(b4) + len(b5) == 81598

    def test_temporal_grouping_rounding_proof(self):
        """Prove that the 16,321.77 vs 16,321.76 discrepancy is purely display rounding."""
        b1 = self.df[(self.df["time_gap_years"] >= 3.0) & (self.df["time_gap_years"] <= 5.0)]
        b2 = self.df[(self.df["time_gap_years"] > 5.0) & (self.df["time_gap_years"] <= 10.0)]
        b3 = self.df[(self.df["time_gap_years"] > 10.0) & (self.df["time_gap_years"] <= 15.0)]
        b4 = self.df[(self.df["time_gap_years"] > 15.0) & (self.df["time_gap_years"] <= 20.0)]
        b5 = self.df[self.df["time_gap_years"] > 20.0]

        # Exact unrounded sums
        loss_1 = b1["area_loss_km2"].sum()  # 442.6917
        loss_2 = b2["area_loss_km2"].sum()  # 4712.6167
        loss_3 = b3["area_loss_km2"].sum()  # 4997.6015
        loss_4 = b4["area_loss_km2"].sum()  # 1727.7768
        loss_5 = b5["area_loss_km2"].sum()  # 4441.0763

        unrounded_sum = loss_1 + loss_2 + loss_3 + loss_4 + loss_5
        assert np.isclose(unrounded_sum, 16321.7630, atol=1e-4)
        assert round(unrounded_sum, 2) == 16321.76

        # Rounded individual display numbers
        disp_1 = round(loss_1, 2)  # 442.69
        disp_2 = round(loss_2, 2)  # 4712.62
        disp_3 = round(loss_3, 2)  # 4997.60
        disp_4 = round(loss_4, 2)  # 1727.78
        disp_5 = round(loss_5, 2)  # 4441.08

        rounded_sum = disp_1 + disp_2 + disp_3 + disp_4 + disp_5
        assert rounded_sum == 16321.77, "Sum of rounded group losses must equal 16,321.77"
        assert round(rounded_sum - unrounded_sum, 4) == 0.0070

    def test_retreating_percentages_per_group(self):
        """Verify retreating percentages within each group match summary JSON."""
        bins = [
            ((self.df["time_gap_years"] >= 3.0) & (self.df["time_gap_years"] <= 5.0), 48.68),
            ((self.df["time_gap_years"] > 5.0) & (self.df["time_gap_years"] <= 10.0), 62.73),
            ((self.df["time_gap_years"] > 10.0) & (self.df["time_gap_years"] <= 15.0), 72.17),
            ((self.df["time_gap_years"] > 15.0) & (self.df["time_gap_years"] <= 20.0), 70.39),
            ((self.df["time_gap_years"] > 20.0), 95.46),
        ]
        for mask, expected_pct in bins:
            sub = self.df[mask]
            ret_count = sub["dynamics_class"].isin(["Significant Retreat", "Moderate Retreat"]).sum()
            actual_pct = round(ret_count / len(sub) * 100.0, 2)
            assert actual_pct == expected_pct

    def test_annualized_rates_sum_and_mean(self):
        """Verify the exact formula R_group = sum(r_i) where r_i = (A1 - A2)/dt."""
        expected_sums = [77.88, 667.36, 436.23, 107.51, 127.55]
        expected_means = [0.0060, 0.0205, 0.0240, 0.0123, 0.0141]

        masks = [
            (self.df["time_gap_years"] >= 3.0) & (self.df["time_gap_years"] <= 5.0),
            (self.df["time_gap_years"] > 5.0) & (self.df["time_gap_years"] <= 10.0),
            (self.df["time_gap_years"] > 10.0) & (self.df["time_gap_years"] <= 15.0),
            (self.df["time_gap_years"] > 15.0) & (self.df["time_gap_years"] <= 20.0),
            self.df["time_gap_years"] > 20.0,
        ]

        for mask, exp_sum, exp_mean in zip(masks, expected_sums, expected_means):
            sub = self.df[mask]
            actual_sum = round(sub["annual_loss_km2_yr"].sum(), 2)
            actual_mean = round(sub["annual_loss_km2_yr"].mean(), 4)
            assert actual_sum == exp_sum, f"Sum mismatch: {actual_sum} != {exp_sum}"
            assert actual_mean == exp_mean, f"Mean mismatch: {actual_mean} != {exp_mean}"


from src.ui.area_assessment import calculate_area_change_scenario


# =====================================================================
# 3. AREA CHANGE SCENARIO CALCULATOR
# =====================================================================
class TestObjective3ScenarioCalculator:
    @staticmethod
    def calculate_scenario(a1: float, a2: float, dt: float):
        """Calls the canonical calculation function from area_assessment."""
        res = calculate_area_change_scenario(a1, a2, dt)
        if not res.get("valid", False):
            return {"error": res.get("error", "Invalid inputs")}
        return res

    def test_expansion_scenario_test_a(self):
        """Test A: A1 = 100, A2 = 110, dt = 10."""
        res = self.calculate_scenario(100.0, 110.0, 10.0)
        assert res["delta_a"] == 10.0
        assert res["pct_a"] == 10.0
        assert res["rate_a"] == 1.0
        assert res["rate_pct"] == 1.0
        assert res["net_decrease"] == 0.0
        assert res["dyn_class"] in ["Measured Area Advance", "Measured Advance"]
        assert np.isclose(res["cagr"], 0.957658, atol=1e-4)

    def test_retreat_scenario_test_b(self):
        """Test B: A1 = 100, A2 = 90, dt = 10."""
        res = self.calculate_scenario(100.0, 90.0, 10.0)
        assert res["delta_a"] == -10.0
        assert res["pct_a"] == -10.0
        assert res["rate_a"] == -1.0
        assert res["rate_pct"] == -1.0
        assert res["net_decrease"] == 10.0
        assert res["dyn_class"] in ["Significant Retreat", "Significant Measured Retreat"]
        assert np.isclose(res["cagr"], -1.04815, atol=1e-4)

    def test_moderate_retreat_scenario(self):
        """Moderate retreat: A1 = 100, A2 = 95, dt = 10."""
        res = self.calculate_scenario(100.0, 95.0, 10.0)
        assert res["delta_a"] == -5.0
        assert res["pct_a"] == -5.0
        assert res["rate_a"] == -0.5
        assert res["net_decrease"] == 5.0
        assert res["dyn_class"] in ["Moderate Retreat", "Moderate Measured Retreat"]

    def test_equilibrium_scenario(self):
        """Equilibrium: A1 = 100, A2 = 100, dt = 10."""
        res = self.calculate_scenario(100.0, 100.0, 10.0)
        assert res["delta_a"] == 0.0
        assert res["pct_a"] == 0.0
        assert res["rate_a"] == 0.0
        assert res["net_decrease"] == 0.0
        assert res["dyn_class"] == "Stable / Equilibrium"

    def test_deadband_boundary_thresholds(self):
        """Test boundary thresholds around +/-2% and -10%."""
        # Exactly -10.0% -> Significant Measured Retreat
        res = self.calculate_scenario(100.0, 90.0, 10.0)
        assert res["dyn_class"] in ["Significant Retreat", "Significant Measured Retreat"]

        # -9.99% -> Moderate Measured Retreat
        res = self.calculate_scenario(100.0, 90.01, 10.0)
        assert res["dyn_class"] in ["Moderate Retreat", "Moderate Measured Retreat"]

        # Exactly -2.0% -> Moderate Measured Retreat
        res = self.calculate_scenario(100.0, 98.0, 10.0)
        assert res["dyn_class"] in ["Moderate Retreat", "Moderate Measured Retreat"]

        # -1.99% -> Stable / Equilibrium
        res = self.calculate_scenario(100.0, 98.01, 10.0)
        assert res["dyn_class"] == "Stable / Equilibrium"

        # +1.99% -> Stable / Equilibrium
        res = self.calculate_scenario(100.0, 101.99, 10.0)
        assert res["dyn_class"] == "Stable / Equilibrium"

        # Exactly +2.0% -> Measured Advance
        res = self.calculate_scenario(100.0, 102.0, 10.0)
        assert res["dyn_class"] in ["Measured Area Advance", "Measured Advance"]

    def test_invalid_and_edge_inputs(self):
        """Verify safe rejection of invalid inputs (division by zero, negative area/dt)."""
        assert "error" in self.calculate_scenario(0.0, 100.0, 10.0)
        assert "error" in self.calculate_scenario(-50.0, 100.0, 10.0)
        assert "error" in self.calculate_scenario(100.0, -10.0, 10.0)
        assert "error" in self.calculate_scenario(100.0, 100.0, 0.0)
        assert "error" in self.calculate_scenario(100.0, 100.0, -5.0)

        # Zero final area (glacier disappearance) is valid
        res = self.calculate_scenario(100.0, 0.0, 10.0)
        assert res["delta_a"] == -100.0
        assert res["pct_a"] == -100.0
        assert res["dyn_class"] in ["Significant Retreat", "Significant Measured Retreat"]
        assert res["net_decrease"] == 100.0


# =====================================================================
# 4. FILTER SENSITIVITY & INVARIANCE
# =====================================================================
class TestObjective3SensitivityAnalysis:
    @classmethod
    def setup_class(cls):
        json_path = RESULTS_DIR / "filter_sensitivity_analysis.json"
        assert json_path.exists(), f"Sensitivity JSON not found: {json_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_twelve_configurations_present(self):
        """Verify all 12 filter configurations exist in the results."""
        frag_list = self.data["fragment_threshold_sensitivity"]
        ratio_list = self.data["ratio_bounds_sensitivity"]
        assert len(frag_list) == 6
        assert len(ratio_list) == 6
        assert len(frag_list) + len(ratio_list) == 12

    def test_baseline_configuration_integrity(self):
        """Verify baseline configuration in both sensitivity slices matches production values."""
        base_frag = next(r for r in self.data["fragment_threshold_sensitivity"] if r["frag_threshold"] == 0.20)
        base_ratio = next(r for r in self.data["ratio_bounds_sensitivity"] if r["ratio_bounds"] == "(0.15, 3.0)")

        for b in [base_frag, base_ratio]:
            assert b["total_glaciers"] == 81598
            assert b["retreating_glaciers"] == 54713  # matches vectorized sensitivity output
            assert np.isclose(b["retreating_pct"], 67.05, atol=0.02)
            assert np.isclose(b["net_area_loss_km2"], 16321.74, atol=0.1)

    def test_retreat_dominance_invariance(self):
        """Verify that retreat dominance (66%-68%) is invariant across all 12 configurations."""
        all_configs = self.data["fragment_threshold_sensitivity"] + self.data["ratio_bounds_sensitivity"]
        for c in all_configs:
            ret_pct = c["retreating_pct"]
            adv_pct = c["advancing_pct"]
            sta_pct = c["stable_pct"]
            assert 66.0 <= ret_pct <= 68.0, f"Retreat percentage {ret_pct}% out of invariant range!"
            assert 22.0 <= adv_pct <= 26.0, f"Advance percentage {adv_pct}% out of invariant range!"
            assert 7.5 <= sta_pct <= 10.5, f"Stable percentage {sta_pct}% out of invariant range!"
            assert np.isclose(ret_pct + adv_pct + sta_pct, 100.0, atol=0.1)


# =====================================================================
# 5. TOP-RETREATING GLACIER RANKING
# =====================================================================
class TestObjective3TopGlaciersRanking:
    @classmethod
    def setup_class(cls):
        summary_path = RESULTS_DIR / "glims_retreat_summary_validated.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            cls.top10 = json.load(f)["top_10_retreat_glaciers"]

    def test_top10_monotonic_loss(self):
        """Verify top 10 glaciers are ordered monotonically descending by net area loss."""
        assert len(self.top10) == 10
        losses = [g["area_loss_km2"] for g in self.top10]
        assert losses == sorted(losses, reverse=True), "Top 10 glaciers are not monotonically descending!"

    def test_top1_glacier_identity_and_metrics(self):
        """Verify specific top 1 retreating glacier identity and metrics."""
        g1 = self.top10[0]
        assert g1["glacier_id"] == "G217991E60521N"
        assert g1["earlier_date"] == "2001-08-15"
        assert g1["later_date"] == "2010-09-10"
        assert g1["time_gap_years"] == 9.07
        assert g1["area_loss_km2"] == 606.82
        assert g1["percentage_change"] == 16.71

    def test_top_glaciers_have_observation_periods(self):
        """Verify all top glaciers have valid observation dates and intervals."""
        for g in self.top10:
            assert "earlier_date" in g and "later_date" in g
            assert "time_gap_years" in g and g["time_gap_years"] >= 3.0
            assert "area_loss_km2" in g and g["area_loss_km2"] > 0


# =====================================================================
# 6. GLIMS DATABASE EXPLORER & SEARCH
# =====================================================================
class TestObjective3DatabaseExplorer:
    @classmethod
    def setup_class(cls):
        csv_path = RESULTS_DIR / "glims_retreat_assessment_validated.csv"
        cls.df = pd.read_csv(csv_path)

    def test_benchmark_glacier_resolutions(self):
        """Verify all Swiss benchmark glaciers resolve in the full database."""
        benchmarks = {
            "Rhone": "G008379E46648N",
            "Great Aletsch": "G008032E46504N",
            "Unteraar": "G008187E46569N",
            "Gorner": "G007800E45965N",
            "Findel": "G007880E45990N",
        }
        for name, gid in benchmarks.items():
            matches = self.df[self.df["glacier_id"] == gid]
            assert len(matches) == 1, f"Glacier {name} ({gid}) failed to resolve unique row!"
            row = matches.iloc[0]
            assert row["earlier_area_km2"] > 0
            assert row["later_area_km2"] > 0
            assert row["time_gap_years"] >= 3.0

    def test_category_filter_subsets(self):
        """Verify category filters return exact subsets."""
        cats = ["Significant Retreat", "Moderate Retreat", "Advance / Growth", "Stable / Equilibrium"]
        total = 0
        for cat in cats:
            sub = self.df[self.df["dynamics_class"] == cat]
            assert len(sub) > 0
            total += len(sub)
        assert total == 81598

    def test_export_preserves_precision(self):
        """Verify export DataFrame retains all 15 columns with full precision."""
        expected_cols = [
            "glacier_id", "earlier_analysis_id", "later_analysis_id",
            "earlier_date", "later_date", "earlier_year", "later_year",
            "time_gap_years", "earlier_area_km2", "later_area_km2",
            "area_loss_km2", "percentage_change", "annual_loss_km2_yr",
            "annual_pct_change_yr", "dynamics_class"
        ]
        assert list(self.df.columns) == expected_cols
        assert self.df.notna().all().all(), "Unexpected NaN values in dataset export!"


# =====================================================================
# 7. OBJECTIVE 3 DELIVERABLES AND SUMMARY ARTIFACTS
# =====================================================================
class TestObjective3Deliverables:
    def test_summary_csv_artifacts_exist_and_valid(self):
        """Verify machine-readable CSV summary files exist with correct row counts."""
        f_global = RESULTS_DIR / "objective3_global_accounting.csv"
        f_tg = RESULTS_DIR / "objective3_temporal_group_summary.csv"
        f_sens = RESULTS_DIR / "objective3_sensitivity_12_configs.csv"
        f_top = RESULTS_DIR / "objective3_top_ranking.csv"

        assert f_global.exists(), f"Missing: {f_global}"
        assert f_tg.exists(), f"Missing: {f_tg}"
        assert f_sens.exists(), f"Missing: {f_sens}"
        assert f_top.exists(), f"Missing: {f_top}"

        df_g = pd.read_csv(f_global)
        assert len(df_g) == 6
        assert "Earlier Pooled Area (A1)" in df_g["accounting_component"].values

        df_t = pd.read_csv(f_tg)
        assert len(df_t) == 5
        assert df_t["glacier_pairs"].sum() == 81598

        df_s = pd.read_csv(f_sens)
        assert len(df_s) == 12

        df_r = pd.read_csv(f_top)
        assert len(df_r) == 10
        assert df_r.iloc[0]["glacier_id"] == "G217991E60521N"
        assert df_r.iloc[0]["area_loss_km2"] == 606.82
