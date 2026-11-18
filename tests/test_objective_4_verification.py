"""Automated Validation and Verification Test Suite for Objective 4:
Interactive Multi-Sensor Visual Glacier Monitoring & Satellite-Based Change Interpretation.

Validates:
1. DL4GAM NetCDF sample ingestion, 13 Sentinel-2 spectral bands, quality masks, Copernicus 10m DEM.
2. Radiometric true-color (RGB) and false-color (SWIR) composite rendering with percentile stretching.
3. NDSI cryogenic index computation with safe zero-denominator handling and heuristic thresholding.
4. Real DEM analytical hillshade, elevation statistics, and terrain mapping.
5. Individual band radiometric stats and histogram generation.
6. Landsat-8 bitemporal change detection pipeline across 683 matched triplets.
7. Absolute pixel deviation, undirected surface change mask, coral overlay, and physical area in km².
8. Ground-truth reference mask evaluation: TP, FP, FN, TN, Precision, Recall, IoU, and Dice score.
9. Vector/morphological perimeter boundary overlays with custom styling.
10. Upload parsing for RGB, NetCDF, and GeoTIFF formats with graceful validation and capability flags.
11. High-resolution exports (PNG and georeferenced GeoTIFF).
"""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.visual_analysis import (
    S2_BAND_SPECS,
    BAND_NAMES_13,
    get_featured_dl4gam_patches,
    load_dl4gam_patch,
    parse_uploaded_satellite_image,
    resolve_s2_band_indices,
    generate_true_color_rgb,
    generate_swir_false_color,
    get_single_band_display,
    compute_band_statistics,
    compute_ndsi,
    compute_dem_analytics,
    parse_and_validate_dem_geotiff,
    get_available_landsat_samples,
    load_landsat_sample,
    compute_bitemporal_change,
    generate_confusion_map_overlay,
    evaluate_change_accuracy,
    overlay_vector_boundary_on_image,
    load_cached_glims_benchmarks,
    render_benchmark_glims_overlay,
    parse_and_validate_vector_upload,
    inspect_uploaded_geotiff,
    export_array_to_png_bytes,
    export_array_to_geotiff_bytes,
)

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
DATASETS_DIR = ROOT / "datasets"
DL4GAM_DIR = DATASETS_DIR / "dl4gam" / "extracted"
LANDSAT_DIR = DATASETS_DIR / "landsat"


class TestObjective4Sentinel2Explorer:
    """Test suite for Sentinel-2 multispectral and DEM exploration."""

    def test_dl4gam_featured_patches_existence(self):
        """Verify featured patches list is non-empty and points to existing NetCDF files."""
        patches = get_featured_dl4gam_patches()
        assert len(patches) >= 6, "Expected at least 6 featured DL4GAM patches."
        for p in patches:
            assert Path(p["path"]).exists(), f"Patch file missing: {p['path']}"
            assert p["glacier_id"].startswith("g_"), f"Invalid glacier id: {p['glacier_id']}"
            assert "title" in p and "region" in p

    def test_load_dl4gam_patch_structure(self):
        """Verify loading NetCDF patch returns 13 bands, DEM, masks, and metadata."""
        patches = get_featured_dl4gam_patches()
        sample_path = patches[0]["path"]
        data = load_dl4gam_patch(sample_path)

        assert "bands" in data
        assert "dem" in data
        assert "cloudless_mask" in data
        assert "fill_mask" in data
        assert "spatial_ref" in data
        assert "glacier_id" in data

        # Check 13 spectral bands
        bands = data["bands"]
        assert isinstance(bands, np.ndarray)
        assert bands.shape == (13, 256, 256), f"Unexpected band shape: {bands.shape}"
        assert bands.dtype in (np.float32, np.float64)

        # Check DEM
        dem = data["dem"]
        assert isinstance(dem, np.ndarray)
        assert dem.shape == (256, 256)
        # Elevation in meters a.s.l.
        assert np.nanmin(dem) > -100
        assert np.nanmax(dem) < 9000

    def test_true_color_rgb_generation(self):
        """Verify true-color RGB composite generation with percentile stretch."""
        patches = get_featured_dl4gam_patches()
        data = load_dl4gam_patch(patches[0]["path"])

        rgb = generate_true_color_rgb(data["bands"], p_low=2.0, p_high=98.0, brightness=1.0, contrast=1.0, gamma=1.0)
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (256, 256, 3)
        assert rgb.dtype == np.uint8
        assert rgb.min() >= 0 and rgb.max() <= 255
        # Ensure image is not entirely black or white
        assert rgb.mean() > 5.0
        assert rgb.mean() < 250.0

    def test_swir_false_color_generation(self):
        """Verify false-color SWIR (B11-B8-B4) generation and custom channel selection."""
        patches = get_featured_dl4gam_patches()
        data = load_dl4gam_patch(patches[0]["path"])

        swir = generate_swir_false_color(data["bands"], swir_idx=11, nir_idx=7, red_idx=3)
        assert isinstance(swir, np.ndarray)
        assert swir.shape == (256, 256, 3)
        assert swir.dtype == np.uint8
        assert swir.min() >= 0 and swir.max() <= 255

    def test_ndsi_computation_and_thresholding(self):
        """Verify NDSI computation, safe division guard, bounds [-1, 1], and mask output."""
        patches = get_featured_dl4gam_patches()
        data = load_dl4gam_patch(patches[0]["path"])

        green_band = data["bands"][2]  # B3
        swir_band = data["bands"][11]  # B11
        result = compute_ndsi(green_band, swir_band, threshold=0.40)

        ndsi = result["ndsi_raw"]
        mask = result["binary_mask"]
        colored = result["colorized_rgb"]

        assert isinstance(ndsi, np.ndarray)
        assert ndsi.shape == (256, 256)
        assert not np.isinf(ndsi).any(), "NDSI contains infinity!"
        # NDSI mathematically bounded by [-1, 1]
        valid = ~np.isnan(ndsi)
        if valid.any():
            assert ndsi[valid].min() >= -1.0 - 1e-5
            assert ndsi[valid].max() <= 1.0 + 1e-5

        assert mask.dtype == np.uint8
        assert mask.shape == (256, 256)
        assert colored.shape == (256, 256, 3)
        assert colored.dtype == np.uint8
        assert "effective_resolution" in result

    def test_dem_analytics_and_hillshade(self):
        """Verify DEM analytics calculate valid stats and hillshading bounded in [0, 255]."""
        patches = get_featured_dl4gam_patches()
        data = load_dl4gam_patch(patches[0]["path"])

        dem_info = compute_dem_analytics(data["dem"], sun_azimuth=315.0, sun_altitude=45.0)
        assert dem_info["valid"] is True
        assert dem_info["min_elevation"] <= dem_info["max_elevation"]
        assert dem_info["min_elevation"] <= dem_info["mean_elevation"] <= dem_info["max_elevation"]

        hs = dem_info["hillshade"]
        assert isinstance(hs, np.ndarray)
        assert hs.shape == (256, 256)
        assert hs.dtype == np.uint8
        assert hs.min() >= 0 and hs.max() <= 255

        colored = dem_info["dem_rgb"]
        assert isinstance(colored, np.ndarray)
        assert colored.shape == (256, 256, 3)
        assert colored.dtype == np.uint8

    def test_dem_unavailable_graceful_handling(self):
        """Verify None DEM handled gracefully with clear valid=False flag."""
        dem_info = compute_dem_analytics(np.array([], dtype=np.float32))
        assert dem_info["valid"] is False

    def test_band_statistics_and_histogram(self):
        """Verify single band extraction and radiometric stats computation."""
        patches = get_featured_dl4gam_patches()
        data = load_dl4gam_patch(patches[0]["path"])

        stats = compute_band_statistics(data["bands"][2], "B3 (Green)")
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "p5" in stats
        assert "p95" in stats
        assert "hist_counts" in stats and "bin_edges" in stats
        assert len(stats["hist_counts"]) == 30
        assert stats["min"] <= stats["max"]

        disp = get_single_band_display(data["bands"][2])
        assert disp.shape == (256, 256)
        assert disp.dtype == np.uint8


class TestObjective4LandsatChangeExplorer:
    """Test suite for Landsat bitemporal change detection across 683 triplets."""

    def test_landsat_triplet_inventory(self):
        """Confirm exactly 683 matched bitemporal triplets exist in A, B, and labels."""
        samples = get_available_landsat_samples()
        assert len(samples) == 683, f"Expected 683 matched Landsat triplets, found {len(samples)}"
        for s in samples[:10]:
            assert (LANDSAT_DIR / "A" / f"{s}.png").exists()
            assert (LANDSAT_DIR / "B" / f"{s}.png").exists()
            assert (LANDSAT_DIR / "labels" / f"{s}.png").exists()

    def test_load_landsat_sample(self):
        """Verify loading a Landsat sample loads A, B, and label images."""
        samples = get_available_landsat_samples()
        sample = load_landsat_sample(samples[0])

        assert sample["img_a"].shape == (256, 256, 3)
        assert sample["img_b"].shape == (256, 256, 3)
        assert sample["ref_mask"].shape == (256, 256)
        assert sample["img_a"].dtype == np.uint8
        assert sample["img_b"].dtype == np.uint8
        assert sample["ref_mask"].dtype == np.uint8

    def test_bitemporal_change_pipeline_five_panels(self):
        """Verify bitemporal change computation produces the required metrics and panels."""
        samples = get_available_landsat_samples()
        sample = load_landsat_sample(samples[0])

        result = compute_bitemporal_change(sample["img_a"], sample["img_b"], threshold=35.0, pixel_resolution_m=30.0)

        assert "diff_magnitude" in result
        assert "diff_rgb" in result
        assert "detected_mask" in result
        assert "overlay_b" in result
        assert "total_pixels" in result
        assert "changed_pixels" in result
        assert "change_pct" in result
        assert "change_area_km2" in result

        # Check shapes
        assert result["diff_magnitude"].shape == (256, 256)
        assert result["diff_rgb"].shape == (256, 256, 3)
        assert result["detected_mask"].shape == (256, 256)
        assert result["overlay_b"].shape == (256, 256, 3)

        # Check area formula: area_km2 == changed_pixels * (30 * 30 / 1e6)
        expected_area = result["changed_pixels"] * (30.0 * 30.0) / 1_000_000.0
        assert np.isclose(result["change_area_km2"], expected_area, atol=1e-5)
        assert result["total_pixels"] == 256 * 256

    def test_change_accuracy_evaluation(self):
        """Verify ground-truth comparison calculates TP, FP, FN, TN, Precision, Recall, IoU, Dice."""
        samples = get_available_landsat_samples()
        sample = load_landsat_sample(samples[0])
        result = compute_bitemporal_change(sample["img_a"], sample["img_b"], threshold=35.0)

        metrics = evaluate_change_accuracy(result["detected_mask"], sample["ref_mask"])
        assert "TP" in metrics
        assert "FP" in metrics
        assert "FN" in metrics
        assert "TN" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "iou" in metrics
        assert "dice" in metrics

        assert metrics["TP"] + metrics["FP"] + metrics["FN"] + metrics["TN"] == 256 * 256
        for key in ["precision", "recall", "iou", "dice"]:
            assert 0.0 <= metrics[key] <= 1.0, f"Metric {key} out of bounds: {metrics[key]}"

    def test_dimension_mismatch_handling(self):
        """Verify image dimension mismatch between A and B raises ValueError."""
        img_a = np.zeros((256, 256, 3), dtype=np.uint8)
        img_b = np.zeros((128, 128, 3), dtype=np.uint8)

        with pytest.raises(ValueError) as excinfo:
            compute_bitemporal_change(img_a, img_b)
        assert "dimension mismatch" in str(excinfo.value)

    def test_landsat_scientific_validation_metrics_and_thresholds(self):
        """Verify exact benchmark confusion matrix on 1_10_1 and multi-threshold sensitivity."""
        sample = load_landsat_sample("1_10_1")
        res_35 = compute_bitemporal_change(sample["img_a"], sample["img_b"], threshold=35.0, pixel_resolution_m=30.0)

        # Exact confusion matrix on 1_10_1
        metrics_35 = evaluate_change_accuracy(res_35["detected_mask"], sample["ref_mask"])
        assert metrics_35["TP"] == 2649
        assert metrics_35["FP"] == 52140
        assert metrics_35["FN"] == 4145
        assert metrics_35["TN"] == 6602
        assert np.isclose(metrics_35["precision"], 0.048349, atol=1e-4)
        assert np.isclose(metrics_35["recall"], 0.389903, atol=1e-4)
        assert np.isclose(metrics_35["iou"], 0.044949, atol=1e-4)
        assert np.isclose(metrics_35["dice"], 0.086030, atol=1e-4)

        # Verify area calculation at 30m resolution
        assert res_35["changed_pixels"] == 54789
        assert np.isclose(res_35["change_area_km2"], 49.3101, atol=1e-4)

        # Verify threshold sensitivity monotonicity
        t_low = compute_bitemporal_change(sample["img_a"], sample["img_b"], threshold=20.0)["changed_pixels"]
        t_mid = compute_bitemporal_change(sample["img_a"], sample["img_b"], threshold=50.0)["changed_pixels"]
        t_high = compute_bitemporal_change(sample["img_a"], sample["img_b"], threshold=100.0)["changed_pixels"]

        assert t_low > res_35["changed_pixels"] > t_mid > t_high, "Detected changed pixels must decrease monotonically with increasing threshold"

    def test_alternative_detection_methods_and_parameters(self):
        """Verify candidate change detection methods (Grayscale, Euclidean, Offset-Normalized)."""
        sample = load_landsat_sample("1_10_1")

        # Method A / Baseline
        res_a = compute_bitemporal_change(sample["img_a"], sample["img_b"], threshold=35.0, method="mean_rgb")
        assert res_a["changed_pixels"] == 54789

        # Method B / Grayscale
        res_b = compute_bitemporal_change(sample["img_a"], sample["img_b"], threshold=35.0, method="grayscale")
        assert res_b["diff_magnitude"].shape == (256, 256)
        assert res_b["changed_pixels"] > 0

        # Method D / Euclidean RGB
        res_d = compute_bitemporal_change(sample["img_a"], sample["img_b"], threshold=90.0, method="euclidean")
        assert res_d["diff_magnitude"].shape == (256, 256)
        assert res_d["changed_pixels"] < res_a["changed_pixels"]

        # Method E / Offset-Normalized Difference
        res_e = compute_bitemporal_change(sample["img_a"], sample["img_b"], threshold=40.0, method="offset_normalized")
        assert res_e["diff_magnitude"].shape == (256, 256)
        # Offset normalization reduces false positives dramatically on 1_10_1
        assert res_e["changed_pixels"] < 10000

    def test_confusion_map_overlay_generation(self):
        """Verify 4-color prediction vs reference overlay rendering."""
        base = np.full((256, 256, 3), 128, dtype=np.uint8)
        pred = np.zeros((256, 256), dtype=np.uint8)
        ref = np.zeros((256, 256), dtype=np.uint8)

        # TP at [50:100, 50:100]
        pred[50:100, 50:100] = 1
        ref[50:100, 50:100] = 1

        # FP at [120:150, 50:100]
        pred[120:150, 50:100] = 1

        # FN at [50:100, 120:150]
        ref[50:100, 120:150] = 1

        overlay = generate_confusion_map_overlay(base, pred, ref)
        assert overlay.shape == (256, 256, 3)
        assert overlay.dtype == np.uint8

        # TP region should have green tint (channel 1 highest)
        assert overlay[75, 75, 1] > overlay[75, 75, 0]
        # FP region should have red tint (channel 0 highest)
        assert overlay[135, 75, 0] > overlay[135, 75, 1]
        # FN region should have amber tint (channel 0 and 1 high)
        assert overlay[75, 135, 0] > 150 and overlay[75, 135, 1] > 100

    def test_scientific_inventory_and_split_integrity(self):
        """Verify dataset inventory and split manifest exist with exactly 683 triplets."""
        import csv
        inv_file = ROOT / "outputs" / "results" / "objective4_landsat_dataset_inventory.csv"
        split_file = ROOT / "outputs" / "results" / "objective4_landsat_split_manifest.csv"

        assert inv_file.exists(), "Inventory CSV must exist"
        assert split_file.exists(), "Split manifest CSV must exist"

        with open(inv_file, encoding="utf-8") as f:
            inv_rows = list(csv.DictReader(f))
        assert len(inv_rows) == 683

        with open(split_file, encoding="utf-8") as f:
            split_rows = list(csv.DictReader(f))
        assert len(split_rows) == 683

        splits = set(r["split"] for r in split_rows)
        assert splits == {"calibration_train", "validation", "held_out_test"}

        # Check active vs no-change counts
        active = sum(1 for r in inv_rows if r["is_active_change"] == "True")
        no_ch = sum(1 for r in inv_rows if r["is_active_change"] == "False")
        assert active == 560
        assert no_ch == 123

    def test_landsat_custom_upload_validation_and_processing(self):
        """Verify custom upload validation: matching dims, mismatched dims, and optional reference mask."""
        # 1. Matching pair
        a_mock = np.full((256, 256, 3), 70, dtype=np.uint8)
        b_mock = np.full((256, 256, 3), 40, dtype=np.uint8)
        ref_mock = np.zeros((256, 256), dtype=np.uint8)
        ref_mock[50:100, 50:100] = 1

        res = compute_bitemporal_change(a_mock, b_mock, threshold=20.0, method="mean_rgb")
        assert res["total_pixels"] == 65536
        assert res["changed_pixels"] == 65536  # All pixels changed by 30 DN >= 20.0
        
        # Test evaluation metrics with reference mask
        metrics = evaluate_change_accuracy(res["detected_mask"], ref_mock)
        assert metrics["TP"] == 2500  # 50x50 area
        assert metrics["FP"] == 65536 - 2500
        assert "specificity" in metrics
        
        # 2. Mismatched imagery dimensions
        mismatched_b = np.full((128, 128, 3), 40, dtype=np.uint8)
        with pytest.raises(ValueError) as exc:
            compute_bitemporal_change(a_mock, mismatched_b)
        assert "dimension mismatch" in str(exc.value)
        
        # 3. Mismatched reference mask dimensions
        mismatched_ref = np.zeros((128, 128), dtype=np.uint8)
        with pytest.raises(ValueError) as exc_ref:
            evaluate_change_accuracy(res["detected_mask"], mismatched_ref)
        assert "shape mismatch" in str(exc_ref.value)


class TestObjective4VectorBoundaryOverlay:
    """Test suite for vector perimeter extraction and overlay."""

    def test_overlay_vector_boundary(self):
        """Verify vector boundary overlay correctly draws contour perimeters."""
        base_img = np.full((256, 256, 3), 100, dtype=np.uint8)
        # Create a central square mask
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 1

        overlay = overlay_vector_boundary_on_image(
            base_img,
            mask,
            color_hex="#00FFFF",
            line_thickness=2,
            opacity=0.9,
        )

        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == (256, 256, 3)
        assert overlay.dtype == np.uint8
        # Pixels along perimeter should have been modified
        assert not np.array_equal(overlay, base_img)
        # Deep interior should remain unmodified
        assert np.array_equal(overlay[100, 100], base_img[100, 100])


class TestObjective4UploadHandling:
    """Test suite for Mode B upload parsing and validation."""

    def test_rgb_upload_disables_multispectral_and_dem(self):
        """Verify plain RGB image upload sets is_multispectral=False and has_dem=False."""
        # Create in-memory PNG
        buf = io.BytesIO()
        img = Image.new("RGB", (128, 128), color=(20, 180, 240))
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        parsed = parse_uploaded_satellite_image(png_bytes, "glacier_sample.png")
        assert parsed["is_multispectral"] is False
        assert parsed["is_rgb_only"] is True
        assert parsed["has_dem"] is False
        assert parsed["channels"] == 3
        assert parsed["bands"].shape == (3, 128, 128)
        assert parsed["dem"] is None

    def test_invalid_file_upload_rejected(self):
        """Verify corrupted or invalid binary input raises ValueError gracefully."""
        corrupt_bytes = b"NOT_A_VALID_IMAGE_OR_NETCDF"
        with pytest.raises(ValueError) as excinfo:
            parse_uploaded_satellite_image(corrupt_bytes, "bogus.dat")
        assert "Unsupported" in str(excinfo.value) or "Failed" in str(excinfo.value)

    def test_netcdf_upload_parsing(self):
        """Verify valid NetCDF upload correctly parsed into bands and DEM."""
        patches = get_featured_dl4gam_patches()
        nc_bytes = Path(patches[0]["path"]).read_bytes()

        parsed = parse_uploaded_satellite_image(nc_bytes, "test_patch.nc")
        assert parsed["is_multispectral"] is True
        assert parsed["has_dem"] is True
        assert parsed["bands"].shape[0] >= 13
        assert parsed["dem"] is not None

    def test_rhone_multiband_geotiff_upload_parsing(self):
        """Verify upload and parsing of the 5-band Sentinel-2 Rhone GeoTIFF."""
        rhone_path = DATASETS_DIR / "objective4" / "sentinel2_rhone" / "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
        assert rhone_path.exists(), f"Rhone GeoTIFF file not found: {rhone_path}"
        tif_bytes = rhone_path.read_bytes()

        parsed = parse_uploaded_satellite_image(tif_bytes, rhone_path.name)
        assert parsed["channels"] == 5
        assert parsed["width"] == 463
        assert parsed["height"] == 448
        assert parsed["spatial_ref"] == "EPSG:32632"
        assert parsed["pixel_size"] == (20.0, 20.0)
        assert parsed["band_names"] == ["B02", "B03", "B04", "B08", "B11"]
        assert parsed["is_multispectral"] is True
        assert parsed["is_rgb_only"] is False
        assert parsed["has_dem"] is False
        assert parsed["transform"] is not None
        assert parsed["nodata_vals"] == (-9999.0, -9999.0, -9999.0, -9999.0, -9999.0)
        assert parsed["spacecraft"] == "Sentinel-2 MSI"

    def test_rhone_band_resolution_and_composites(self):
        """Verify True-Color RGB, SWIR false-color, and NDSI generated from Rhone 5 bands."""
        rhone_path = DATASETS_DIR / "objective4" / "sentinel2_rhone" / "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
        parsed = parse_uploaded_satellite_image(rhone_path.read_bytes(), rhone_path.name)

        indices = resolve_s2_band_indices(parsed["band_names"])
        assert indices["blue"] == 0   # B02
        assert indices["green"] == 1  # B03
        assert indices["red"] == 2    # B04
        assert indices["nir"] == 3    # B08
        assert indices["swir1"] == 4  # B11

        # A. True-color RGB: Red = B04 (idx 2), Green = B03 (idx 1), Blue = B02 (idx 0)
        rgb = generate_true_color_rgb(
            parsed["bands"],
            r_idx=indices["red"],
            g_idx=indices["green"],
            b_idx=indices["blue"],
        )
        assert rgb.shape == (448, 463, 3)
        assert rgb.dtype == np.uint8
        assert 10 < rgb.mean() < 240

        # B. False-color SWIR: Red = B11 (idx 4), Green = B08 (idx 3), Blue = B04 (idx 2)
        swir = generate_swir_false_color(
            parsed["bands"],
            swir_idx=indices["swir1"],
            nir_idx=indices["nir"],
            red_idx=indices["red"],
        )
        assert swir.shape == (448, 463, 3)
        assert swir.dtype == np.uint8
        assert 10 < swir.mean() < 240

        # C. NDSI: (B03 - B11) / (B03 + B11)
        green_chan = parsed["bands"][indices["green"]]
        swir_chan = parsed["bands"][indices["swir1"]]
        ndsi_res = compute_ndsi(green_chan, swir_chan, threshold=0.40)
        assert ndsi_res["ndsi_raw"].shape == (448, 463)
        assert ndsi_res["binary_mask"].shape == (448, 463)
        assert ndsi_res["colorized_rgb"].shape == (448, 463, 3)
        assert ndsi_res["snow_ice_pixels"] > 0
        assert np.nanmin(ndsi_res["ndsi_raw"]) >= -1.0
        assert np.nanmax(ndsi_res["ndsi_raw"]) <= 1.0

        # D. Individual band stats and display
        for i, b_name in enumerate(parsed["band_names"]):
            stats = compute_band_statistics(parsed["bands"][i], b_name)
            assert stats["valid"] is True
            assert stats["min"] <= stats["max"]
            disp = get_single_band_display(parsed["bands"][i])
            assert disp.shape == (448, 463)

    def test_swir_default_channel_mapping_rhone_and_dl4gam(self):
        """Verify default SWIR mapping strictly assigns Red=B11, Green=B08/B8, Blue=B04/B4 and never duplicates."""
        # 1. Test 5-band Rhone Sentinel-2 GeoTIFF
        rhone_fp = DATASETS_DIR / "objective4" / "sentinel2_rhone" / "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
        with open(rhone_fp, "rb") as f:
            rhone_parsed = parse_uploaded_satellite_image(f.read(), "rhone.tif")

        rhone_map = resolve_s2_band_indices(rhone_parsed["band_names"])
        r_name = rhone_parsed["band_names"][rhone_map["swir1"]]
        g_name = rhone_parsed["band_names"][rhone_map["nir"]]
        b_name = rhone_parsed["band_names"][rhone_map["red"]]

        assert r_name == "B11", f"Expected Red=B11, got {r_name}"
        assert g_name == "B08", f"Expected Green=B08, got {g_name}"
        assert b_name == "B04", f"Expected Blue=B04, got {b_name}"
        assert r_name != g_name, "Red and Green must never be identical in SWIR composite!"
        assert g_name != b_name, "Green and Blue must never be identical in SWIR composite!"

        # Verify physical wavelength lookup
        assert S2_BAND_SPECS[r_name]["wavelength"] == "1610 nm"
        assert S2_BAND_SPECS[g_name]["wavelength"] == "842 nm"
        assert S2_BAND_SPECS[b_name]["wavelength"] == "665 nm"

        # 2. Test 13-band DL4GAM NetCDF patch
        dl4gam_patches = get_featured_dl4gam_patches()
        dl4gam_parsed = load_dl4gam_patch(dl4gam_patches[0]["path"])
        dl4gam_map = resolve_s2_band_indices(dl4gam_parsed["band_names"])
        dl_r_name = dl4gam_parsed["band_names"][dl4gam_map["swir1"]]
        dl_g_name = dl4gam_parsed["band_names"][dl4gam_map["nir"]]
        dl_b_name = dl4gam_parsed["band_names"][dl4gam_map["red"]]

        assert dl_r_name == "B11", f"Expected Red=B11, got {dl_r_name}"
        assert dl_g_name == "B8", f"Expected Green=B8, got {dl_g_name}"
        assert dl_b_name == "B4", f"Expected Blue=B4, got {dl_b_name}"
        assert dl_r_name != dl_g_name, "Red and Green must never be identical!"

    def test_swir_interactive_selection_and_reset_apptest(self):
        """Verify interactive SWIR custom channel selection and reset-to-default button in Streamlit AppTest."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("🛰️ Visual Monitoring Suite").run()

        # Check default session state
        assert at.session_state.get("swir_r") == "B11"
        assert at.session_state.get("swir_g") == "B8"
        assert at.session_state.get("swir_b") == "B4"

        # Change blue channel to B2
        sb_b = at.selectbox(key="swir_b")
        sb_b.select("B2").run()
        assert at.session_state.get("swir_b") == "B2"

        # Click Reset button
        rst_btn = at.button(key="reset_swir_btn")
        rst_btn.click().run()
        assert at.session_state.get("swir_r") == "B11"
        assert at.session_state.get("swir_g") == "B8"
        assert at.session_state.get("swir_b") == "B4"

    def test_invalid_geotiff_upload_rejected(self):
        """Verify corrupted TIFF upload raises ValueError with descriptive message."""
        corrupt_tiff = b"II*\x00\x08\x00\x00\x00CORRUPT_TIFF_DATA_STREAM"
        with pytest.raises(ValueError) as excinfo:
            parse_uploaded_satellite_image(corrupt_tiff, "corrupted_raster.tif")
        assert "Failed to parse uploaded GeoTIFF" in str(excinfo.value)

    def test_dem_geotiff_multispectral_rejection(self):
        """Verify 5-band Sentinel-2 file uploaded as DEM is rejected with exact clear explanation."""
        rhone_path = DATASETS_DIR / "objective4" / "sentinel2_rhone" / "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
        assert rhone_path.exists()
        with pytest.raises(ValueError) as excinfo:
            parse_and_validate_dem_geotiff(rhone_path.read_bytes(), rhone_path.name)
        assert "This is a multispectral satellite image, not an elevation model. Please upload a valid DEM GeoTIFF." in str(excinfo.value)

    def test_dem_geotiff_valid_upload_and_analytics(self):
        """Verify genuine compatible DEM GeoTIFF is parsed and computes contours, hillshade, and stats."""
        dem_path = DATASETS_DIR / "objective4" / "dem_rhone" / "rhone_copernicus_dem_30m.tif"
        assert dem_path.exists()

        rhone_s2 = DATASETS_DIR / "objective4" / "sentinel2_rhone" / "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
        s2_data = parse_uploaded_satellite_image(rhone_s2.read_bytes(), rhone_s2.name)

        parsed = parse_and_validate_dem_geotiff(dem_path.read_bytes(), dem_path.name, scene_metadata=s2_data)
        assert parsed["valid"] is True
        assert parsed["crs"] == "EPSG:32632"
        assert parsed["width"] == 463
        assert parsed["height"] == 448

        analytics = compute_dem_analytics(parsed["dem"], resolution=parsed["pixel_size"][0])
        assert analytics["valid"] is True
        assert 1500.0 < analytics["min_elevation"] < 1700.0
        assert 3400.0 < analytics["max_elevation"] < 3600.0
        assert analytics["relief"] > 1800.0
        assert "dem_contours_rgb" in analytics
        assert analytics["dem_contours_rgb"].ndim == 3
        assert analytics["hillshade"].shape == (448, 463)

    def test_dem_geotiff_spatial_overlap_rejection(self):
        """Verify DEM from non-overlapping glacier region is rejected with spatial extent error."""
        dem_other = DATASETS_DIR / "objective4" / "dem_rhone" / "g_3560_dem_copernicus_10m.tif"
        assert dem_other.exists()

        rhone_s2 = DATASETS_DIR / "objective4" / "sentinel2_rhone" / "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
        s2_data = parse_uploaded_satellite_image(rhone_s2.read_bytes(), rhone_s2.name)

        with pytest.raises(ValueError) as excinfo:
            parse_and_validate_dem_geotiff(dem_other.read_bytes(), dem_other.name, scene_metadata=s2_data)
        assert "does not spatially overlap" in str(excinfo.value)


class TestObjective4ExportCapabilities:
    """Test suite for high-resolution PNG and GeoTIFF exports."""

    def test_export_array_to_png_bytes(self):
        """Verify export to PNG generates valid PNG bytes."""
        arr = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        png_bytes = export_array_to_png_bytes(arr)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0

        # Verify Pillow can read it back
        img = Image.open(io.BytesIO(png_bytes))
        assert img.format == "PNG"
        assert img.size == (100, 100)

    def test_export_array_to_geotiff_bytes(self):
        """Verify export to GeoTIFF generates valid TIFF bytes."""
        arr = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        geotiff_bytes = export_array_to_geotiff_bytes(arr, epsg=32632)
        assert isinstance(geotiff_bytes, bytes)
        assert len(geotiff_bytes) > 0

        # Verify Pillow can read it back
        img = Image.open(io.BytesIO(geotiff_bytes))
        assert img.format == "TIFF"
        assert img.size == (100, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# Regression Tests — Method Consistency & Baseline Stability (added 2026-11-16)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMethodConsistencyRegression:
    """
    Regression tests verifying:
    1. Each detection method produces an independent prediction on sample 1_10_1.
    2. The ground-truth reference mask is not mutated by any method.
    3. Method A at T=35 always reproduces the documented baseline confusion matrix.
    4. Method E produces measurably fewer false positives than Method A on the
       illumination-shifted sample 1_10_1 (~36 DN scene-wide darkening in Epoch B).
    """

    def test_method_selection_produces_different_predictions(self):
        """Each method must produce a distinct prediction on the illumination-shifted 1_10_1 sample."""
        sample = load_landsat_sample("1_10_1")
        img_a, img_b = sample["img_a"], sample["img_b"]
        ref = sample["ref_mask"]

        res_a = compute_bitemporal_change(img_a, img_b, threshold=35.0, method="mean_rgb")
        res_e = compute_bitemporal_change(img_a, img_b, threshold=40.0, method="offset_normalized")

        m_a = evaluate_change_accuracy(res_a["detected_mask"], ref)
        m_e = evaluate_change_accuracy(res_e["detected_mask"], ref)

        # Method E must produce far fewer changed pixels (illumination offset removed)
        assert res_e["changed_pixels"] < res_a["changed_pixels"], (
            f"Method E changed_pixels ({res_e['changed_pixels']:,}) must be < "
            f"Method A ({res_a['changed_pixels']:,}) on illumination-shifted 1_10_1"
        )
        # Method E must produce fewer FP
        assert m_e["FP"] < m_a["FP"], (
            f"Method E FP ({m_e['FP']:,}) must be < Method A FP ({m_a['FP']:,}) on 1_10_1"
        )
        # Method E must have higher precision
        assert m_e["precision"] > m_a["precision"], (
            f"Method E precision ({m_e['precision']:.4f}) must exceed "
            f"Method A precision ({m_a['precision']:.4f}) on 1_10_1"
        )

    def test_ground_truth_unchanged_across_methods(self):
        """Reference mask must not be mutated by any method call."""
        sample = load_landsat_sample("1_10_1")
        original_ref = sample["ref_mask"].copy()

        for method in ("mean_rgb", "grayscale", "euclidean", "offset_normalized"):
            compute_bitemporal_change(
                sample["img_a"], sample["img_b"],
                threshold=35.0, method=method
            )

        np.testing.assert_array_equal(
            sample["ref_mask"], original_ref,
            err_msg="Reference mask was mutated by compute_bitemporal_change"
        )

    def test_method_a_exact_baseline_regression(self):
        """
        Method A at T=35 on sample 1_10_1 must always reproduce the documented
        baseline confusion matrix: TP=2649, FP=52140, FN=4145, TN=6602.
        """
        sample = load_landsat_sample("1_10_1")
        res = compute_bitemporal_change(
            sample["img_a"], sample["img_b"],
            threshold=35.0, method="mean_rgb"
        )
        m = evaluate_change_accuracy(res["detected_mask"], sample["ref_mask"])

        assert m["TP"] == 2649,  f"TP regression: expected 2649, got {m['TP']}"
        assert m["FP"] == 52140, f"FP regression: expected 52140, got {m['FP']}"
        assert m["FN"] == 4145,  f"FN regression: expected 4145, got {m['FN']}"
        assert m["TN"] == 6602,  f"TN regression: expected 6602, got {m['TN']}"


class TestObjective4GLIMSOverlay:
    """Validation test suite for Tab 3: GLIMS Vector Boundary Overlay & Geospatial Integration."""

    def test_glims_benchmarks_presence_and_ids(self):
        """All four primary benchmarks plus Findel must be present with authoritative GLIMS IDs."""
        benchmarks = load_cached_glims_benchmarks()
        assert len(benchmarks) >= 5, "Expected at least 5 cached GLIMS benchmarks."

        expected_ids = {
            "rhone": "G008379E46648N",
            "great_aletsch": "G008032E46504N",
            "unteraar": "G008187E46569N",
            "gorner": "G007800E45965N",
            "findel": "G007880E45990N",
        }
        for key, exp_id in expected_ids.items():
            assert key in benchmarks, f"Missing benchmark key '{key}'"
            assert benchmarks[key]["id"] == exp_id, f"ID mismatch for {key}: {benchmarks[key]['id']} vs {exp_id}"
            assert "wkt_later" in benchmarks[key], f"Missing wkt_later for {key}"
            assert "wkt_earlier" in benchmarks[key], f"Missing wkt_earlier for {key}"

    def test_glims_area_consistency_with_objective_2(self):
        """
        Verify area calculations for Rhone match Objective 2:
        Earlier: 17.166 km², Later: 14.179 km², Net Loss: 2.987 km² (-17.40%).
        The old informal placeholder was 2.33 km²; this must be completely replaced.
        """
        benchmarks = load_cached_glims_benchmarks()
        rhone = benchmarks["rhone"]

        assert rhone["earlier_area_km2"] == 17.166
        assert rhone["later_area_km2"] == 14.179
        assert rhone["net_loss_km2"] == 2.987
        assert rhone["percentage_change"] == 17.40
        assert rhone["earlier_date"] == "1973-09-01"
        assert rhone["later_date"] == "2015-08-29"

        # Great Aletsch
        aletsch = benchmarks["great_aletsch"]
        assert aletsch["earlier_area_km2"] == 86.565
        assert aletsch["later_area_km2"] == 78.069
        assert aletsch["net_loss_km2"] == 8.496

        # Unteraar
        unteraar = benchmarks["unteraar"]
        assert unteraar["earlier_area_km2"] == 27.131
        assert unteraar["later_area_km2"] == 20.816
        assert unteraar["net_loss_km2"] == 6.316

        # Gorner
        gorner = benchmarks["gorner"]
        assert gorner["earlier_area_km2"] == 57.715
        assert gorner["later_area_km2"] == 41.997
        assert gorner["net_loss_km2"] == 15.718

    def test_crs_transformation_and_geometry_validity(self):
        """Verify vector geometries are valid shapely objects in conformal UTM Zone 32N."""
        import shapely.wkt
        benchmarks = load_cached_glims_benchmarks()

        for key, bdata in benchmarks.items():
            assert bdata["crs"] == "EPSG:32632"
            geom_e = shapely.wkt.loads(bdata["wkt_earlier"])
            geom_l = shapely.wkt.loads(bdata["wkt_later"])
            assert geom_e.is_valid, f"Earlier geometry for {key} is invalid"
            assert geom_l.is_valid, f"Later geometry for {key} is invalid"
            # Bounds must be within Swiss UTM 32N coordinates
            minx, miny, maxx, maxy = geom_l.bounds
            assert 400_000 <= minx <= 600_000, f"Easting out of range for {key}: {minx}"
            assert 5_000_000 <= miny <= 5_300_000, f"Northing out of range for {key}: {miny}"

    def test_raster_vector_spatial_alignment_audit(self):
        """
        Document and verify the geospatial reality:
        Rhone GeoTIFF tile extent is [448620, 5155320, 457880, 5164280] in EPSG:32632.
        GLIMS Rhone polygon extent is [448849, 5164599, 454650, 5170180] in EPSG:32632.
        The glacier terminus is ~319 m north of the local GeoTIFF tile extent.
        """
        import rasterio
        tif_path = ROOT / "datasets" / "objective4" / "sentinel2_rhone" / "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
        assert tif_path.exists()

        with rasterio.open(tif_path) as src:
            assert str(src.crs) == "EPSG:32632"
            assert src.bounds.top == 5164280.0
            assert src.bounds.left == 448620.0

        benchmarks = load_cached_glims_benchmarks()
        rhone = benchmarks["rhone"]
        import shapely.wkt
        geom_l = shapely.wkt.loads(rhone["wkt_later"])
        minx, miny, maxx, maxy = geom_l.bounds

        # Min Northing of GLIMS polygon is above top of GeoTIFF
        northing_offset = miny - 5164280.0
        assert northing_offset > 0, "GLIMS terminus is expected to be north of GeoTIFF top"
        assert 300 < northing_offset < 350, f"Expected ~319m offset, got {northing_offset:.1f}m"

    def test_dynamic_rendering_controls_affect_output(self):
        """Verify that line color, thickness, opacity, and layer toggles modify the rendered overlay array."""
        benchmarks = load_cached_glims_benchmarks()
        rhone = benchmarks["rhone"]

        # Default render
        base = render_benchmark_glims_overlay(rhone, color_hex="#12DFF3", line_thickness=2, opacity=0.9, show_secondary=True)
        assert isinstance(base, np.ndarray)
        assert base.ndim == 3 and base.shape[2] in (3, 4)

        # Color change
        arr_color = render_benchmark_glims_overlay(rhone, color_hex="#FFFFFF", line_thickness=2, opacity=0.9, show_secondary=True)
        assert not np.array_equal(base, arr_color), "Changing color must modify the rendered image array."

        # Thickness change
        arr_thick = render_benchmark_glims_overlay(rhone, color_hex="#12DFF3", line_thickness=5, opacity=0.9, show_secondary=True)
        assert not np.array_equal(base, arr_thick), "Changing thickness must modify the rendered image array."

        # Opacity change
        arr_opacity = render_benchmark_glims_overlay(rhone, color_hex="#12DFF3", line_thickness=2, opacity=0.2, show_secondary=True)
        assert not np.array_equal(base, arr_opacity), "Changing opacity must modify the rendered image array."

        # Secondary layer toggle
        arr_no_secondary = render_benchmark_glims_overlay(rhone, color_hex="#12DFF3", line_thickness=2, opacity=0.9, show_secondary=False)
        assert not np.array_equal(base, arr_no_secondary), "Toggling secondary layer must modify the rendered image array."

    def test_distinct_imagery_per_benchmark(self):
        """Each benchmark glacier must render its own distinct geometry and figure array."""
        benchmarks = load_cached_glims_benchmarks()
        rendered_images = {}
        for key in ("rhone", "great_aletsch", "unteraar", "gorner"):
            rendered_images[key] = render_benchmark_glims_overlay(benchmarks[key])

        # Verify no two benchmarks produce the same image array
        keys = list(rendered_images.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                assert not np.array_equal(rendered_images[k1], rendered_images[k2]), (
                    f"Benchmarks '{k1}' and '{k2}' must produce distinct renderings!"
                )

    def test_vector_upload_validation_and_projection(self):
        """Verify vector upload parsing, CRS reprojection to EPSG:32632, and rejection of invalid data."""
        import geopandas as gpd
        from shapely.geometry import Polygon
        import io, json, zipfile

        # 1. Valid GeoJSON in EPSG:4326
        poly = Polygon([[8.35, 46.56], [8.38, 46.56], [8.38, 46.59], [8.35, 46.59], [8.35, 46.56]])
        gdf = gpd.GeoDataFrame([{"name": "Rhone Test", "geometry": poly}], crs="EPSG:4326")
        geojson_bytes = gdf.to_json().encode("utf-8")

        res = parse_and_validate_vector_upload(geojson_bytes, "test_rhone.geojson")
        assert res["valid"] is True
        assert res["feature_count"] == 1
        assert res["projected_crs"] == "EPSG:32632"
        assert res["area_km2"] > 0
        assert isinstance(res["rendered_array"], np.ndarray)

        # 2. Corrupt JSON
        with pytest.raises(ValueError, match="Corrupt GeoJSON"):
            parse_and_validate_vector_upload(b"{invalid json content", "bad.geojson")

        # 3. Shapefile ZIP missing .prj
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            z.writestr("test.shp", b"dummy")
            z.writestr("test.shx", b"dummy")
            z.writestr("test.dbf", b"dummy")
        zip_buf.seek(0)
        with pytest.raises(ValueError, match="Missing .prj"):
            parse_and_validate_vector_upload(zip_buf.getvalue(), "missing_prj.zip")

    def test_geotiff_upload_inspection(self):
        """Verify GeoTIFF raster inspection extracts CRS, bounds, resolution, and band preview."""
        tif_path = ROOT / "datasets" / "objective4" / "sentinel2_rhone" / "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
        tif_bytes = tif_path.read_bytes()

        res = inspect_uploaded_geotiff(tif_bytes)
        assert res["valid"] is True
        assert res["crs"] == "EPSG:32632"
        assert res["width"] == 463
        assert res["height"] == 448
        assert res["band_count"] == 5
        assert res["preview_rgb"].shape == (448, 463, 3)

    def test_export_array_to_png_bytes_supported_modes(self):
        """
        Verify export_array_to_png_bytes handles all array dimensionalities without
        raising 'ValueError: Invalid mode for data type'.
        """
        # 1. 2D grayscale
        arr_2d = np.zeros((64, 64), dtype=np.uint8)
        b_2d = export_array_to_png_bytes(arr_2d)
        assert len(b_2d) > 0

        # 2. 3D single-channel
        arr_1ch = np.zeros((64, 64, 1), dtype=np.uint8)
        b_1ch = export_array_to_png_bytes(arr_1ch)
        assert len(b_1ch) > 0

        # 3. 3D RGB
        arr_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        b_rgb = export_array_to_png_bytes(arr_rgb)
        assert len(b_rgb) > 0

        # 4. 3D RGBA (Prior bug: crashed with mode='RGB')
        arr_rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        b_rgba = export_array_to_png_bytes(arr_rgba)
        assert len(b_rgba) > 0
        im = Image.open(io.BytesIO(b_rgba))
        assert im.mode == "RGBA"

        # 5. Floating point array [0.0, 1.0]
        arr_flt = np.ones((64, 64, 4), dtype=np.float32) * 0.8
        b_flt = export_array_to_png_bytes(arr_flt)
        assert len(b_flt) > 0

    def test_dynamic_glims_overlay_with_satellite_and_debris(self):
        """
        Verify render_benchmark_glims_overlay successfully renders Rhone with satellite
        background and supraglacial debris, and can be exported to PNG without error.
        """
        benchmarks = load_cached_glims_benchmarks()
        rhone = benchmarks["rhone"]

        overlay = render_benchmark_glims_overlay(
            rhone,
            color_hex="#12DFF3",
            line_thickness=2,
            opacity=0.9,
            show_secondary=True,
            show_debris=True,
            show_satellite=True,
        )
        assert isinstance(overlay, np.ndarray)
        assert overlay.shape[2] == 4  # RGBA
        assert overlay.dtype == np.uint8

        # PNG export must succeed without error
        png_bytes = export_array_to_png_bytes(overlay)
        assert len(png_bytes) > 10_000

    def test_debris_toggle_affects_rendering(self):
        """Toggling supraglacial debris must alter the rendered image array."""
        benchmarks = load_cached_glims_benchmarks()
        rhone = benchmarks["rhone"]

        arr_with_deb = render_benchmark_glims_overlay(rhone, show_debris=True)
        arr_without_deb = render_benchmark_glims_overlay(rhone, show_debris=False)

        assert not np.array_equal(arr_with_deb, arr_without_deb), (
            "show_debris=True vs False must produce distinct image arrays!"
        )
