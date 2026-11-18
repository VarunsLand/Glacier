"""
compare_all_methods_1_10_1.py
Compare all 5 detection methods on sample 1_10_1 and print full confusion matrices.
"""
import sys
sys.path.insert(0, r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")

import numpy as np
from src.visual_analysis import compute_bitemporal_change, evaluate_change_accuracy, load_landsat_sample

SAMPLE_ID = "1_10_1"


def print_method_results(name, m_code, threshold, img_a, img_b, ref):
    res = compute_bitemporal_change(img_a, img_b, threshold=threshold, method=m_code)
    m = evaluate_change_accuracy(res["detected_mask"], ref)
    tot = m["TP"] + m["FP"] + m["FN"] + m["TN"]
    print(f"\n{'='*60}")
    print(f"  {name}  (method={m_code}, T={threshold})")
    print(f"{'='*60}")
    print(f"  Changed pixels : {res['changed_pixels']:>7,}  /  {res['total_pixels']:,}")
    print(f"  Change ratio   : {res['change_pct']:.2f}%")
    print(f"  TP : {m['TP']:>7,}  ({m['TP']/tot*100:.2f}%)")
    print(f"  FP : {m['FP']:>7,}  ({m['FP']/tot*100:.2f}%)")
    print(f"  FN : {m['FN']:>7,}  ({m['FN']/tot*100:.2f}%)")
    print(f"  TN : {m['TN']:>7,}  ({m['TN']/tot*100:.2f}%)")
    print(f"  Precision  : {m['precision']:.4f}")
    print(f"  Recall     : {m['recall']:.4f}")
    print(f"  Specificity: {m.get('specificity', float('nan')):.4f}")
    print(f"  IoU        : {m['iou']:.4f}")
    print(f"  Dice       : {m['dice']:.4f}")
    return m


if __name__ == "__main__":
    print(f"\nLoading sample {SAMPLE_ID}...")
    sample = load_landsat_sample(SAMPLE_ID)
    img_a = sample["img_a"]
    img_b = sample["img_b"]
    ref   = sample["ref_mask"]
    print(f"  img_a shape: {img_a.shape}, dtype: {img_a.dtype}")
    print(f"  img_b shape: {img_b.shape}, dtype: {img_b.dtype}")
    print(f"  ref   shape: {ref.shape},  positives: {ref.sum():,}")

    # Radiometric profile
    for ch, name in enumerate(["R", "G", "B"]):
        a_mean = img_a[:, :, ch].mean()
        b_mean = img_b[:, :, ch].mean()
        print(f"  {name}: A_mean={a_mean:.2f}, B_mean={b_mean:.2f}, ΔI={b_mean-a_mean:.2f}")

    methods = [
        ("Method A: Baseline Mean RGB",             "mean_rgb",        35.0),
        ("Method B: Grayscale Difference",           "grayscale",       35.0),
        ("Method C: Calibrated Mean RGB",            "mean_rgb",        60.0),
        ("Method D: Euclidean RGB Distance",         "euclidean",       90.0),
        ("Method E: Illumination Offset-Normalized", "offset_normalized", 40.0),
    ]

    results = {}
    for name, code, thr in methods:
        results[name] = print_method_results(name, code, thr, img_a, img_b, ref)

    # Summary comparison
    print(f"\n{'='*60}")
    print("  SUMMARY COMPARISON — Sample 1_10_1")
    print(f"{'='*60}")
    print(f"  {'Method':<45} {'TP':>6} {'FP':>7} {'FN':>6} {'TN':>7} {'Prec':>7} {'Rec':>7} {'IoU':>7}")
    print(f"  {'-'*105}")
    for name, code, thr in methods:
        m = results[name]
        print(f"  {name:<45} {m['TP']:>6,} {m['FP']:>7,} {m['FN']:>6,} {m['TN']:>7,} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['iou']:>7.4f}")

    print()
    m_a = results["Method A: Baseline Mean RGB"]
    m_e = results["Method E: Illumination Offset-Normalized"]
    if m_e["FP"] < m_a["FP"]:
        print(f"  ✓ Method E FP ({m_e['FP']:,}) < Method A FP ({m_a['FP']:,}) — illumination correction is effective on 1_10_1")
    else:
        print(f"  ✗ Method E FP ({m_e['FP']:,}) >= Method A FP ({m_a['FP']:,}) — normalization did NOT reduce FP on this sample")
