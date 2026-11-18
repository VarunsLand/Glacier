"""GlacierVision: Visual Glacier Monitoring & Multi-Spectral Composite Generator.

Objective 4 Implementation:
1. True-Color Sentinel-2 RGB Composite (B4/B3/B2).
2. False-Color Snow/Ice Composite (B11 SWIR / B8 NIR / B4 Red).
3. Precision Boundary Overlays on original satellite imagery.
4. Supraglacial Debris vs Clean Ice Contour Overlays.
5. Landsat-8 Polar Bitemporal Retreat Change Maps (Time A vs Time B).
"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import DL4GAMDataset
from src.metrics import extract_boundary
import torch

FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_sentinel2_monitoring_suite():
    print("Generating Sentinel-2 Multispectral Monitoring Suite...")
    ds = DL4GAMDataset(root=ROOT, fold="fold_test")
    # Select patch with interesting features (debris and distinct boundary)
    sample_path = Path(ds.patch_list[0][1])

    with xr.open_dataset(sample_path) as data:
        raw_bands = data["band_data"].values[:13] * ds.scale  # (13, 256, 256)
        mask_crt = data["mask_crt_g"].values  # (256, 256)
        mask_debris = data["mask_debris"].values if "mask_debris" in data else np.zeros_like(mask_crt)
        dem = data["dem"].values if "dem" in data else None

    # 1. True Color RGB (B4=Red, B3=Green, B2=Blue)
    rgb = np.stack([raw_bands[3], raw_bands[2], raw_bands[1]], axis=-1)
    p2, p98 = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)

    # 2. False Color Ice/Snow Composite (B11 SWIR1, B8 NIR, B4 Red)
    # Snow and ice absorb SWIR strongly while reflecting NIR, appearing vibrant cyan/electric blue!
    swir = raw_bands[11]  # B11 (1610nm)
    nir = raw_bands[7]    # B8 (842nm)
    red = raw_bands[3]    # B4 (665nm)
    fcolor = np.stack([swir, nir, red], axis=-1)
    p2_fc, p98_fc = np.nanpercentile(fcolor, (2, 98))
    fcolor = np.clip((fcolor - p2_fc) / max(p98_fc - p2_fc, 1e-6), 0.0, 1.0)

    # 3. Extract boundaries
    t_mask = torch.from_numpy(mask_crt[None, None, :, :]).float()
    b_mask = extract_boundary(t_mask)[0, 0].numpy() > 0.5

    t_deb = torch.from_numpy(mask_debris[None, None, :, :]).float()
    b_deb = extract_boundary(t_deb)[0, 0].numpy() > 0.5

    # 4. Overlays
    overlay_rgb = rgb.copy()
    overlay_rgb[b_mask] = [0.0, 1.0, 0.2]  # Bright green glacier margin
    if b_deb.any():
        overlay_rgb[b_deb] = [1.0, 0.6, 0.0]  # Amber debris margin

    overlay_fc = fcolor.copy()
    overlay_fc[b_mask] = [1.0, 0.0, 0.2]  # Red margin on false color

    # Build 5-panel comprehensive figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), dpi=150)

    axes[0].imshow(rgb)
    axes[0].set_title("1. True-Color RGB\n(Bands 4, 3, 2)", fontsize=11, fontweight="bold")

    axes[1].imshow(fcolor)
    axes[1].set_title("2. Glaciological False-Color\n(SWIR1 B11 / NIR B8 / Red B4)", fontsize=11, fontweight="bold")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("3. Glacier Boundary Overlay\n(Green: Clean Ice, Amber: Debris)", fontsize=11, fontweight="bold")

    axes[3].imshow(overlay_fc)
    axes[3].set_title("4. Margin on SWIR/NIR Composite\n(Delineates Ice vs Bedrock)", fontsize=11, fontweight="bold")

    if dem is not None:
        im_dem = axes[4].imshow(dem, cmap="terrain")
        axes[4].set_title("5. Topographic DEM (Elevation)\n(Copernicus 10m Model)", fontsize=11, fontweight="bold")
        plt.colorbar(im_dem, ax=axes[4], fraction=0.046, pad=0.04, label="Metres")
    else:
        axes[4].imshow(mask_crt, cmap="Blues_r")
        axes[4].set_title("5. Target Glacier Mask", fontsize=11, fontweight="bold")

    for ax in axes[:4]:
        ax.set_xticks([])
        ax.set_yticks([])
    if dem is None:
        axes[4].set_xticks([]); axes[4].set_yticks([])

    plt.suptitle(f"Sentinel-2 High-Resolution Visual Glacier Monitoring — Patch {sample_path.name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_s2 = FIG_DIR / "visual_monitoring_sentinel2_composites.png"
    plt.savefig(out_s2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Sentinel-2 monitoring suite: {out_s2}")


def generate_landsat_bitemporal_suite():
    print("Generating Landsat-8 Polar Bitemporal Retreat Monitoring Suite...")
    landsat_dir = ROOT / "datasets" / "landsat"
    sample_a_path = landsat_dir / "A" / "1_1_1.png"
    sample_b_path = landsat_dir / "B" / "1_1_1.png"
    sample_lbl_path = landsat_dir / "labels" / "1_1_1.png"

    if not (sample_a_path.exists() and sample_b_path.exists() and sample_lbl_path.exists()):
        print("Landsat files not found; skipping Landsat visual suite.")
        return

    img_a = np.array(Image.open(sample_a_path))
    img_b = np.array(Image.open(sample_b_path))
    lbl = np.array(Image.open(sample_lbl_path))
    binary_change = (lbl >= 128).astype(np.uint8)

    # Calculate absolute difference map
    diff_rgb = np.abs(img_b.astype(np.float32) - img_a.astype(np.float32)).mean(axis=-1)
    diff_rgb_norm = np.clip(diff_rgb / 60.0, 0.0, 1.0)

    # Change overlay on later image B
    overlay_b = img_b.copy()
    overlay_b[binary_change == 1] = [230, 30, 30]  # Bright red highlight on change areas

    # Contour of change
    t_ch = torch.from_numpy(binary_change[None, None, :, :]).float()
    b_ch = extract_boundary(t_ch)[0, 0].numpy() > 0.5
    contour_overlay = img_b.copy()
    contour_overlay[b_ch] = [255, 230, 0]  # Yellow boundary on perimeter

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), dpi=150)

    axes[0].imshow(img_a)
    axes[0].set_title("1. Earlier Timepoint (A)\n(Landsat-8 Polar Baseline)", fontsize=11, fontweight="bold")

    axes[1].imshow(img_b)
    axes[1].set_title("2. Later Timepoint (B)\n(Multi-Year Resurvey)", fontsize=11, fontweight="bold")

    axes[2].imshow(diff_rgb_norm, cmap="inferno")
    axes[2].set_title("3. Spectral Magnitude Difference\n(|B - A| Spectral Deviation)", fontsize=11, fontweight="bold")

    axes[3].imshow(binary_change, cmap="Reds_r", vmin=0, vmax=1)
    axes[3].set_title("4. Ground Truth Change Mask\n(1 = Verified Glacier Retreat)", fontsize=11, fontweight="bold")

    axes[4].imshow(overlay_b)
    axes[4].set_title("5. Bitemporal Retreat Overlay\n(Red: Verified Glacier Loss)", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Objective 4: Polar Glacier Bitemporal Retreat Monitoring (Greenland / Svalbard)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_ls = FIG_DIR / "visual_monitoring_landsat_bitemporal.png"
    plt.savefig(out_ls, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Landsat bitemporal monitoring suite: {out_ls}")


if __name__ == "__main__":
    generate_sentinel2_monitoring_suite()
    generate_landsat_bitemporal_suite()
    print("Visual monitoring suites generated successfully!")
