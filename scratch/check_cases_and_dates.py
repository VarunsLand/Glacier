import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import json
from src.inference import get_glacier_dynamic_assessment, RESULTS_DIR

print("--- Testing get_glacier_dynamic_assessment ---")
for name in ["Rhone", "Great Aletsch", "Unteraar", "G008379E46648N"]:
    res = get_glacier_dynamic_assessment(name)
    print(f"{name}:")
    print(f"  ID: {res['glacier_id']}, Name: {res['glacier_name']}")
    print(f"  Earlier Date: {res['earlier_date']}, Later Date: {res['later_date']}, Gap: {res['time_gap_years']} yrs")
    print(f"  Earlier Area: {res['earlier_area_km2']}, Later Area: {res['later_area_km2']}")
    print(f"  Change: {res['area_change_km2']}, Class: {res['dynamics_class']}")

print("\n--- Testing glims_case_studies.json ---")
with open(RESULTS_DIR / "glims_case_studies.json", "r", encoding="utf-8") as f:
    cases = json.load(f)
for c in cases:
    print(f"Case {c['glacier_id']}: {c['glacier_name']}")
    print(f"  Years: {c.get('earlier_year')} -> {c.get('later_year')}")
    print(f"  Areas: {c.get('earlier_area_km2')} -> {c.get('later_area_km2')}")
    print(f"  Retained: {c.get('retained_ice_area_km2')}, Lost: {c.get('lost_ice_area_km2')}, Gained: {c.get('gained_ice_area_km2')}")
    print(f"  Net Loss: {c.get('net_loss_km2')}, Pct: {c.get('percentage_change')}")
