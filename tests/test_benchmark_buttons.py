"""Regression test suite for Objective 2 Glacier Benchmark Buttons & Manual Search Synchronization."""
from __future__ import annotations

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from src.inference import get_glacier_dynamic_assessment


class TestBenchmarkButtonsRegression:
    def test_benchmark_data_recalculation(self):
        """Verify dynamic calculation for all 4 benchmark basins and Findel custom basin."""
        cases = {
            "G008379E46648N": {"name": "Rhone", "net_change": -2.987, "pct": -17.40},
            "G008032E46504N": {"name": "Great Aletsch", "net_change": -8.496, "pct": -9.81},
            "G008187E46569N": {"name": "Unteraar", "net_change": -6.316, "pct": -23.28},
            "G007800E45965N": {"name": "Gorner", "net_change": -15.718, "pct": -27.23},
            "G007880E45990N": {"name": "Findel", "net_change": -2.531, "pct": -13.60},
        }
        for gid, exp in cases.items():
            res = get_glacier_dynamic_assessment(gid)
            assert res["found"] is True, f"Failed to retrieve {gid}"
            assert res["glacier_id"] == gid
            assert abs(res["area_change_km2"] - exp["net_change"]) < 0.01, f"Mismatch on {gid} area_change"
            assert abs(abs(res["percent_change"]) - abs(exp["pct"])) < 0.05, f"Mismatch on {gid} pct_change"

    def test_streamlit_apptest_button_switching(self):
        """Verify AppTest interaction flow: clicking each button updates session state and search box."""
        app_path = Path(__file__).parent.parent / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()

        # Switch to Objective 2: Glacier Retreat Explorer
        at.sidebar.radio[0].set_value("🗺️ Glacier Retreat Explorer").run()

        # Initial default must be Rhone (G008379E46648N)
        assert at.session_state["selected_glacier_id"] == "G008379E46648N"
        assert at.session_state["retreat_search_input"] == "G008379E46648N"

        # Click Great Aletsch
        btn_aletsch = at.button(key="btn_quick_aletsch")
        btn_aletsch.click().run()
        assert at.session_state["selected_glacier_id"] == "G008032E46504N"
        assert at.session_state["retreat_search_input"] == "G008032E46504N"

        # Click Unteraar
        btn_unteraar = at.button(key="btn_quick_unteraar")
        btn_unteraar.click().run()
        assert at.session_state["selected_glacier_id"] == "G008187E46569N"
        assert at.session_state["retreat_search_input"] == "G008187E46569N"

        # Click Gorner
        btn_gorner = at.button(key="btn_quick_gorner")
        btn_gorner.click().run()
        assert at.session_state["selected_glacier_id"] == "G007800E45965N"
        assert at.session_state["retreat_search_input"] == "G007800E45965N"

        # Return to Rhone
        btn_rhone = at.button(key="btn_quick_rhone")
        btn_rhone.click().run()
        assert at.session_state["selected_glacier_id"] == "G008379E46648N"
        assert at.session_state["retreat_search_input"] == "G008379E46648N"

        # Manual search entry: G007880E45990N (Findel)
        at.text_input(key="retreat_search_input").input("G007880E45990N").run()
        assert at.session_state["selected_glacier_id"] == "G007880E45990N"
        assert at.session_state["retreat_search_input"] == "G007880E45990N"
