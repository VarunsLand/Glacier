"""GlacierVision: Visual Glacier Monitoring & Multi-Spectral Composite Generator.

Objective 4 Scientific Implementation:
1. True-Color Sentinel-2 RGB Composite (Bands 4, 3, 2) with 2%-98% percentile stretch.
2. Glaciological False-Color Ice/Snow Composite (Bands 11 SWIR1, 8 NIR, 4 Red).
3. Precision Boundary Overlays on original satellite imagery (Clean ice margin vs supraglacial debris).
4. Topographic DEM (Elevation in metres) from Copernicus 10m Model.
5. Landsat-8 Polar Bi-temporal Surface Change Suite with accurate scientific labeling
   (Distinguishes undirected surface change masks from directed cryospheric retreat).
6. High-resolution publication quality (300 DPI, dark navy UI styling).
"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import xarray as xr

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import DL4GAMDataset
from src.metrics import extract_boundary

FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_sentinel2_monitoring_suite():
    print("Generating Sentinel-2 Multispectral Monitoring Suite...")
    ds = DL4GAMDataset(root=ROOT, fold="fold_test")
    # Find a patch with glacier presence and debris
    chosen_path = None
    for gid, p_str in ds.patch_list:
        p = Path(p_str)
        with xr.open_dataset(p) as d:
            if "mask_crt_g" in d and d["mask_crt_g"].values.sum() > 2000:
                chosen_path = p
                break
    if chosen_path is None:
        chosen_path = Path(ds.patch_list[0][1])

    with xr.open_dataset(chosen_path) as data:
        raw_bands = data["band_data"].values[:13] * ds.scale  # (13, 256, 256)
        mask_crt = data["mask_crt_g"].values  # (256, 256)
        mask_debris = data["mask_debris"].values if "mask_debris" in data else np.zeros_like(mask_crt)
        dem = data["dem"].values if "dem" in data else None

    # 1. True Color RGB (B4=Red, B3=Green, B2=Blue)
    rgb = np.stack([raw_bands[3], raw_bands[2], raw_bands[1]], axis=-1)
    p2, p98 = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)

    # 2. Glaciological False Color (B11 SWIR1, B8 NIR, B4 Red)
    swir = raw_bands[11]  # B11 (1610nm)
    nir = raw_bands[7]    # B8 (842nm)
    red = raw_bands[3]    # B4 (665nm)
    fcolor = np.stack([swir, nir, red], axis=-1)
    p2_fc, p98_fc = np.nanpercentile(fcolor, (2, 98))
    fcolor = np.clip((fcolor - p2_fc) / max(p98_fc - p2_fc, 1e-6), 0.0, 1.0)

    # 3. Boundaries
    t_mask = torch.from_numpy(mask_crt[None, None, :, :]).float()
    b_mask = extract_boundary(t_mask)[0, 0].numpy() > 0.5

    t_deb = torch.from_numpy(mask_debris[None, None, :, :]).float()
    b_deb = extract_boundary(t_deb)[0, 0].numpy() > 0.5

    # 4. Overlays
    overlay_rgb = rgb.copy()
    overlay_rgb[b_mask] = [0.0, 0.95, 1.0]  # Electric cyan glacier boundary
    if b_deb.any():
        overlay_rgb[b_deb] = [1.0, 0.65, 0.0]  # Amber debris margin

    overlay_fc = fcolor.copy()
    overlay_fc[b_mask] = [1.0, 0.2, 0.2]  # Bright coral margin on SWIR composite

    # 5-panel figure with dark navy styling
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.0), dpi=300)
    fig.patch.set_facecolor("#0F172A")

    axes[0].imshow(rgb)
    axes[0].set_title("1. True-Color RGB\n(Bands 4, 3, 2)", fontsize=11, fontweight="bold", color="#F8FAFC")

    axes[1].imshow(fcolor)
    axes[1].set_title("2. False-Color Glacier\n(SWIR1 B11 / NIR B8 / Red B4)", fontsize=11, fontweight="bold", color="#F8FAFC")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("3. Glacier Margin Overlay\n(Cyan: Ice Margin, Amber: Debris)", fontsize=11, fontweight="bold", color="#F8FAFC")

    axes[3].imshow(overlay_fc)
    axes[3].set_title("4. Margin on False-Color\n(SWIR Ice Absorption Contrast)", fontsize=11, fontweight="bold", color="#F8FAFC")

    if dem is not None:
        im_dem = axes[4].imshow(dem, cmap="terrain")
        axes[4].set_title("5. Topographic DEM (Elevation)\n(Copernicus 10m Elevation Model)", fontsize=11, fontweight="bold", color="#F8FAFC")
        cbar = plt.colorbar(im_dem, ax=axes[4], fraction=0.046, pad=0.04)
        cbar.set_label("Metres a.s.l.", color="#F8FAFC", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="#94A3B8")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94A3B8")
    else:
        axes[4].imshow(mask_crt, cmap="Blues_r")
        axes[4].set_title("5. Ground Truth Mask (mask_crt_g)", fontsize=11, fontweight="bold", color="#F8FAFC")

    for ax in axes:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8", labelsize=8)
        # Add scale annotation: 256 pixels = 2.56 km
        ax.set_xlabel("10m/pixel (2.56 km footprint)", color="#94A3B8", fontsize=8)

    plt.suptitle(
        f"Sentinel-2 MSI Visual Glacier Monitoring Suite — Patch {chosen_path.name}\n"
        "Sensor: Sentinel-2 MSI (10m GSD) | Region: European Alps Cryosphere | Source: DL4GAM",
        fontsize=13, fontweight="bold", color="#F8FAFC", y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    out_s2 = FIG_DIR / "visual_monitoring_sentinel2_composites.png"
    plt.savefig(out_s2, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)
    plt.close(fig)
    print(f"Saved Sentinel-2 monitoring suite: {out_s2}")


def generate_landsat_bitemporal_suite():
    print("Generating Landsat-8 Polar Bi-temporal Surface Change Monitoring Suite...")
    landsat_dir = ROOT / "datasets" / "landsat"
    sample_a_path = landsat_dir / "A" / "1_1_1.png"
    sample_b_path = landsat_dir / "B" / "1_1_1.png"
    sample_lbl_path = landsat_dir / "labels" / "1_1_1.png"

    if not (sample_a_path.exists() and sample_b_path.exists() and sample_lbl_path.exists()):
        print("Landsat sample files not found; skipping Landsat visual suite.")
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
    overlay_b[binary_change == 1] = [230, 40, 40]  # Coral highlight on detected change areas

    # Boundary of change
    t_ch = torch.from_numpy(binary_change[None, None, :, :]).float()
    b_ch = extract_boundary(t_ch)[0, 0].numpy() > 0.5
    contour_overlay = img_b.copy()
    contour_overlay[b_ch] = [255, 230, 0]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.0), dpi=300)
    fig.patch.set_facecolor("#0F172A")

    axes[0].imshow(img_a)
    axes[0].set_title("1. Earlier Observation (Epoch A)\n(Landsat-8 Polar Baseline)", fontsize=11, fontweight="bold", color="#F8FAFC")

    axes[1].imshow(img_b)
    axes[1].set_title("2. Later Observation (Epoch B)\n(Multi-Year Resurvey)", fontsize=11, fontweight="bold", color="#F8FAFC")

    im_diff = axes[2].imshow(diff_rgb_norm, cmap="magma")
    axes[2].set_title("3. Spectral Deviation (|B - A|)\n(Multi-Band Pixel Magnitude)", fontsize=11, fontweight="bold", color="#F8FAFC")

    axes[3].imshow(binary_change, cmap="Reds_r", vmin=0, vmax=1)
    axes[3].set_title("4. Surface Change Mask\n(Undirected Binary Difference)", fontsize=11, fontweight="bold", color="#F8FAFC")

    axes[4].imshow(overlay_b)
    axes[4].set_title("5. Detected Change Overlay\n(Red: Bi-temporal Surface Change)", fontsize=11, fontweight="bold", color="#F8FAFC")

    for ax in axes:
        ax.set_facecolor("#1E293B")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Landsat-8 OLI (30m GSD)", color="#94A3B8", fontsize=8)

    plt.suptitle(
        "Objective 4: Polar Cryosphere Bi-Temporal Surface Change Monitoring (Landsat-8 OLI)\n"
        "Scientific Note: Binary change masks represent undirected spectral surface change (melt, moraine, snow shift) across multi-year epochs.",
        fontsize=12.5, fontweight="bold", color="#F8FAFC", y=0.98
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    out_ls = FIG_DIR / "visual_monitoring_landsat_bitemporal.png"
    plt.savefig(out_ls, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)
    plt.close(fig)
    print(f"Saved Landsat bi-temporal monitoring suite: {out_ls}")


if __name__ == "__main__":
    generate_sentinel2_monitoring_suite()
    generate_landsat_bitemporal_suite()
    print("Objective 4 visual monitoring suites generated successfully!")
