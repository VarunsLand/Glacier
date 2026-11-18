"""Radiometric Analysis of sample 1_10_1 and dataset."""
import numpy as np
from PIL import Image
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\datasets\landsat")
p_a = DATA_DIR / "A" / "1_10_1.png"
p_b = DATA_DIR / "B" / "1_10_1.png"

arr_a = np.array(Image.open(p_a))
arr_b = np.array(Image.open(p_b))

channels = ["Red", "Green", "Blue"]

print("=== SAMPLE 1_10_1 RADIOMETRIC PROFILE ===")
print("Channel | Obs | Min | Max | Mean | Std | Median | p2 | p98")
print("-" * 65)

for c_idx, c_name in enumerate(channels):
    ca = arr_a[:, :, c_idx].astype(np.float64)
    cb = arr_b[:, :, c_idx].astype(np.float64)
    
    print(f"{c_name:5s}   | A   | {np.min(ca):3.0f} | {np.max(ca):3.0f} | {np.mean(ca):5.2f} | {np.std(ca):5.2f} | {np.median(ca):5.1f}  | {np.percentile(ca, 2):4.1f} | {np.percentile(ca, 98):5.1f}")
    print(f"{c_name:5s}   | B   | {np.min(cb):3.0f} | {np.max(cb):3.0f} | {np.mean(cb):5.2f} | {np.std(cb):5.2f} | {np.median(cb):5.1f}  | {np.percentile(cb, 2):4.1f} | {np.percentile(cb, 98):5.1f}")
    delta_mean = np.mean(cb) - np.mean(ca)
    delta_median = np.median(cb) - np.median(ca)
    print(f"        | Δ   |       |       | {delta_mean:+5.2f} |       | {delta_median:+5.1f}  |      |")
    print("-" * 65)

# Overall 3-channel mean
mean_a_all = np.mean(arr_a)
mean_b_all = np.mean(arr_b)
print(f"Overall Mean A: {mean_a_all:.2f}")
print(f"Overall Mean B: {mean_b_all:.2f}")
print(f"Overall Shift (B - A): {mean_b_all - mean_a_all:+.2f} DN")
