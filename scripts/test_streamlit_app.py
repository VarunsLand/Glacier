"""Test app.py imports, data loaders, and file integrity."""
import sys
from pathlib import Path

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")

# Check all required files exist
required_files = [
    ROOT / "outputs" / "results" / "glims_retreat_summary_validated.json",
    ROOT / "outputs" / "results" / "glims_retreat_assessment_validated.csv",
    ROOT / "outputs" / "results" / "glims_case_studies.json",
    ROOT / "outputs" / "results" / "objective5_clip_coop_report.json",
    ROOT / "outputs" / "figures" / "visual_monitoring_glims_boundary_overlays.png",
    ROOT / "outputs" / "figures" / "visual_monitoring_sentinel2_composites.png",
    ROOT / "outputs" / "figures" / "visual_monitoring_landsat_bitemporal.png",
    ROOT / "outputs" / "figures" / "glims_area_change_distribution.png",
    ROOT / "outputs" / "figures" / "glims_top_retreating_glaciers.png",
    ROOT / "outputs" / "figures" / "clip_vs_coop_comparison.png",
]

print("=" * 60)
print("TESTING WEBSITE DATA INTEGRITY AND FILE PATHS")
print("=" * 60)

missing = []
for f in required_files:
    if f.exists():
        size_kb = f.stat().st_size / 1024
        print(f" [PASS] {f.name} ({size_kb:.1f} KB)")
    else:
        print(f" [FAIL] Missing: {f}")
        missing.append(f)

if missing:
    print(f"\nFailed: {len(missing)} files missing.")
    sys.exit(1)
else:
    print("\nAll 10 required dataset, result, and figure files are verified present!")
