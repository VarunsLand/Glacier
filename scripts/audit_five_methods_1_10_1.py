"""
audit_five_methods_1_10_1.py
Comprehensive mathematical and empirical audit of the 5 Landsat change detection methods on sample 1_10_1.
"""
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
sys.path.insert(0, str(PROJECT_ROOT))

from src.visual_analysis import load_landsat_sample, compute_bitemporal_change, evaluate_change_accuracy

def audit():
    sample_id = "1_10_1"
    sample = load_landsat_sample(sample_id)
    img_a = sample["img_a"]
    img_b = sample["img_b"]
    ref_mask = sample["ref_mask"]

    print(f"=== SAMPLE {sample_id} AUDIT ===")
    print(f"Dimensions: {img_a.shape}, Data type: {img_a.dtype}")
    print(f"Reference mask positives: {ref_mask.sum()} / {ref_mask.size} ({ref_mask.sum()/ref_mask.size*100:.2f}%)")

    # Method definitions
    methods = [
        {
            "name": "Method A",
            "desc": "Baseline Mean RGB (|ΔI| ≥ 35)",
            "code": "mean_rgb",
            "threshold": 35.0,
        },
        {
            "name": "Method B",
            "desc": "Grayscale Difference (|ΔI| ≥ 35)",
            "code": "grayscale",
            "threshold": 35.0,
        },
        {
            "name": "Method C",
            "desc": "Calibrated Mean RGB Difference (|ΔI| ≥ 60)",
            "code": "mean_rgb",
            "threshold": 60.0,
        },
        {
            "name": "Method D",
            "desc": "Euclidean RGB Distance (|ΔI| ≥ 90)",
            "code": "euclidean",
            "threshold": 90.0,
        },
        {
            "name": "Method E",
            "desc": "Illumination Offset-Normalized Difference (|ΔI| ≥ 40)",
            "code": "offset_normalized",
            "threshold": 40.0,
        },
    ]

    # Raw calculation inspection
    a_flt = img_a.astype(np.float32)
    b_flt = img_b.astype(np.float32)

    diff_arrays = {}
    diff_stats = {}
    change_masks = {}
    accuracies = {}

    for m in methods:
        res = compute_bitemporal_change(img_a, img_b, threshold=m["threshold"], method=m["code"])
        mask = res["detected_mask"]
        change_masks[m["name"]] = mask

        # Extract difference array directly to verify independence and compute stats
        code = m["code"]
        if code == "grayscale":
            gray_a = 0.2989 * a_flt[:, :, 0] + 0.5870 * a_flt[:, :, 1] + 0.1140 * a_flt[:, :, 2]
            gray_b = 0.2989 * b_flt[:, :, 0] + 0.5870 * b_flt[:, :, 1] + 0.1140 * b_flt[:, :, 2]
            diff_arr = np.abs(gray_b - gray_a)
        elif code == "euclidean":
            diff_arr = np.sqrt(np.sum((b_flt - a_flt) ** 2, axis=-1))
        elif code == "offset_normalized":
            offset = np.mean(b_flt, axis=(0, 1), keepdims=True) - np.mean(a_flt, axis=(0, 1), keepdims=True)
            diff_arr = np.abs((b_flt - offset) - a_flt).mean(axis=-1)
        else: # mean_rgb
            diff_arr = np.abs(b_flt - a_flt).mean(axis=-1)

        diff_arrays[m["name"]] = diff_arr
        diff_stats[m["name"]] = {
            "min": float(diff_arr.min()),
            "max": float(diff_arr.max()),
            "mean": float(diff_arr.mean()),
            "std": float(diff_arr.std()),
            "median": float(np.median(diff_arr)),
        }

        # Evaluate against reference mask
        eval_res = evaluate_change_accuracy(mask, ref_mask)
        accuracies[m["name"]] = eval_res

    print("\n--- DIFFERENCE ARRAY NUMERICAL RANGE & STATISTICS ---")
    for m in methods:
        name = m["name"]
        st = diff_stats[name]
        print(f"{name} ({m['desc']}):")
        print(f"  Range: [{st['min']:.4f}, {st['max']:.4f}] | Mean: {st['mean']:.4f} | Std: {st['std']:.4f} | Median: {st['median']:.4f}")

    print("\n--- PERFORMANCE & ACCURACY METRICS TABLE ---")
    header = f"| Method | Threshold | Changed pixels | TP | FP | FN | TN | Precision | Recall | Specificity | IoU | Dice |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(sep)
    for m in methods:
        name = m["name"]
        acc = accuracies[name]
        th = m["threshold"]
        ch_pix = int(change_masks[name].sum())
        print(f"| {name}: {m['desc'].split(' (')[0]} | {th:.1f} | {ch_pix:,} | {acc['TP']:,} | {acc['FP']:,} | {acc['FN']:,} | {acc['TN']:,} | {acc['precision']:.4f} | {acc['recall']:.4f} | {acc['specificity']:.4f} | {acc['iou']:.4f} | {acc['dice']:.4f} |")

    print("\n--- PIXEL-BY-PIXEL PREDICTION ARRAY COMPARISON ---")
    m_names = [m["name"] for m in methods]
    for i in range(len(m_names)):
        for j in range(i + 1, len(m_names)):
            n1, n2 = m_names[i], m_names[j]
            mask1 = change_masks[n1]
            mask2 = change_masks[n2]
            identical_pixels = int((mask1 == mask2).sum())
            pct_identical = (identical_pixels / mask1.size) * 100.0
            
            # Intersection and Union between predictions
            inter = int(np.logical_and(mask1 == 1, mask2 == 1).sum())
            union = int(np.logical_or(mask1 == 1, mask2 == 1).sum())
            jaccard = inter / max(union, 1)
            dice = (2.0 * inter) / max(mask1.sum() + mask2.sum(), 1)
            
            # Difference array correlation
            arr1 = diff_arrays[n1].flatten()
            arr2 = diff_arrays[n2].flatten()
            corr = float(np.corrcoef(arr1, arr2)[0, 1])
            mean_diff = float(np.mean(np.abs(arr1 - arr2)))

            print(f"{n1} vs {n2}:")
            print(f"  Pixel agreement: {identical_pixels:,} / {mask1.size:,} ({pct_identical:.2f}%)")
            print(f"  Prediction IoU: {jaccard:.4f} | Prediction Dice: {dice:.4f}")
            print(f"  Diff Array Pearson Correlation: {corr:.6f} | Mean Abs Diff: {mean_diff:.4f}")

    # Inspect why Methods A, B, C, D are visually similar or related
    print("\n--- MATHEMATICAL RELATIONSHIP ANALYSIS ---")
    print(f"1. Method A vs Method B:")
    print(f"   Max absolute pixel difference between diff arrays: {np.max(np.abs(diff_arrays['Method A'] - diff_arrays['Method B'])):.4f}")
    print(f"   Mean absolute pixel difference: {np.mean(np.abs(diff_arrays['Method A'] - diff_arrays['Method B'])):.4f}")
    
    print(f"\n2. Method A vs Method C:")
    print(f"   Are diff arrays identically computed? {np.array_equal(diff_arrays['Method A'], diff_arrays['Method C'])}")
    print(f"   Threshold A = 35.0 -> Changed pixels: {change_masks['Method A'].sum():,}")
    print(f"   Threshold C = 60.0 -> Changed pixels: {change_masks['Method C'].sum():,}")
    print(f"   Is Method C mask a strict subset of Method A mask? {bool(np.all(change_masks['Method A'][change_masks['Method C'] == 1] == 1))}")

    print(f"\n3. Method A vs Method D:")
    ratio = diff_arrays["Method D"] / np.maximum(diff_arrays["Method A"], 1e-6)
    print(f"   Ratio (Euclidean / Mean RGB) Mean: {ratio.mean():.4f}, Std: {ratio.std():.4f}, Min: {ratio.min():.4f}, Max: {ratio.max():.4f}")
    print(f"   Theoretical ratio for monochromatic channels: sqrt(3) ≈ {np.sqrt(3):.4f}")
    print(f"   Threshold D (90.0) / sqrt(3) ≈ {90.0 / np.sqrt(3):.2f}")
    print(f"   Threshold D effective equivalent in Mean RGB: ~{90.0 / ratio.mean():.2f}")

    print(f"\n4. Method E vs Others:")
    offset = np.mean(b_flt, axis=(0, 1)) - np.mean(a_flt, axis=(0, 1))
    print(f"   Epoch B - Epoch A mean channel offset: R={offset[0]:.2f}, G={offset[1]:.2f}, B={offset[2]:.2f}")
    print(f"   Method E FP count: {accuracies['Method E']['FP']} (vs Method A: {accuracies['Method A']['FP']})")

if __name__ == "__main__":
    audit()
