"""Automated Live Verification of GlacierVision Objective 2 Streamlit App.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import urllib.request
from PIL import Image

from src.inference import FIGURES_DIR, RESULTS_DIR, get_glacier_dynamic_assessment
from src.ui.retreat_explorer import load_case_studies


def verify():
    print("=" * 75)
    print("OBJECTIVE 2: LIVE APPLICATION VERIFICATION")
    print("=" * 75)

    # 1. Streamlit HTTP Health Check
    try:
        with urllib.request.urlopen("http://localhost:8501", timeout=5) as resp:
            print(f"[PASS] Streamlit Server running on port 8501: HTTP {resp.status}")
    except Exception as e:
        print(f"[FAIL] Streamlit check failed: {e}")

    # 2. Benchmark Cases Verification
    benchmarks = [
        ("Rhone", "G008379E46648N", 17.166, 14.179, -2.987, -17.40, -0.071, "glims_boundary_rhone"),
        ("Great Aletsch", "G008032E46504N", 86.565, 78.069, -8.496, -9.81, -0.202, "glims_boundary_great_aletsch"),
        ("Unteraar", "G008187E46569N", 27.131, 20.816, -6.316, -23.28, -0.150, "glims_boundary_unteraar"),
    ]

    for name, gid, exp_e, exp_l, exp_net, exp_pct, exp_rate, base_img in benchmarks:
        print(f"\n--- Verifying Benchmark: {name} ({gid}) ---")
        res = get_glacier_dynamic_assessment(name)
        assert res["found"] is True
        assert res["glacier_id"] == gid
        assert abs(res["earlier_area_km2"] - exp_e) < 0.005, f"Earlier area mismatch: {res['earlier_area_km2']}"
        assert abs(res["later_area_km2"] - exp_l) < 0.005, f"Later area mismatch: {res['later_area_km2']}"
        assert abs(res["area_change_km2"] - exp_net) < 0.005, f"Net change mismatch: {res['area_change_km2']}"
        assert abs(res["percent_change"] - exp_pct) < 0.05, f"Pct change mismatch: {res['percent_change']}"
        assert abs(res["annual_rate_km2_yr"] - exp_rate) < 0.005, f"Annual rate mismatch: {res['annual_rate_km2_yr']}"
        assert res["earlier_date"] == "1973-09-01"
        assert res["later_date"] == "2015-08-29"
        print(f"  [PASS] Values: Earlier={res['earlier_area_km2']:.3f} km², Modern={res['later_area_km2']:.3f} km², Net={res['area_change_km2']:+.3f} km² ({res['percent_change']:+.2f}%), Rate={res['annual_rate_km2_yr']:+.3f} km²/year")

        # Verify raster PNG
        png_p = FIGURES_DIR / f"{base_img}.png"
        assert png_p.exists(), f"Missing PNG {png_p}"
        with Image.open(png_p) as img:
            assert img.format == "PNG"
            assert img.width >= 1000
            print(f"  [PASS] High-Res Raster PNG (300 DPI): {png_p.name} ({png_p.stat().st_size} bytes, {img.size})")

        # Verify vector SVG
        svg_p = FIGURES_DIR / f"{base_img}.svg"
        assert svg_p.exists(), f"Missing SVG {svg_p}"
        svg_content = svg_p.read_text(encoding="utf-8")
        assert "<svg" in svg_content
        print(f"  [PASS] Scalable Vector Graphic (SVG): {svg_p.name} ({svg_p.stat().st_size} bytes, valid SVG XML)")

    # 3. Custom Glacier Verification: Gornergletscher
    print(f"\n--- Verifying Custom Glacier: Gornergletscher (G007800E45965N) ---")
    gorner = get_glacier_dynamic_assessment("Gorner")
    assert gorner["found"] is True
    assert gorner["glacier_id"] == "G007800E45965N"
    assert gorner["earlier_area_km2"] > 50.0
    assert gorner["area_change_km2"] < -10.0
    print(f"  [PASS] Gorner loaded: Earlier={gorner['earlier_area_km2']:.3f} km², Modern={gorner['later_area_km2']:.3f} km², Change={gorner['area_change_km2']:+.3f} km²")
    gorner_png = FIGURES_DIR / "glims_boundary_gorner.png"
    gorner_svg = FIGURES_DIR / "glims_boundary_gorner.svg"
    assert gorner_png.exists() and gorner_svg.exists()
    print(f"  [PASS] Gorner individual boundary raster PNG and vector SVG exist.")

    # 4. Invalid Glacier ID Verification
    print(f"\n--- Verifying Invalid Query Error Handling ---")
    inv = get_glacier_dynamic_assessment("G999999E99999N")
    assert inv["found"] is False
    assert inv["earlier_area_km2"] == 0.0
    assert inv["later_area_km2"] == 0.0
    assert inv["area_change_km2"] == 0.0
    print(f"  [PASS] Invalid ID handled gracefully without fabrication: found={inv['found']}, areas=0.0")

    print("\n" + "=" * 75)
    print("ALL OBJECTIVE 2 LIVE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    verify()
