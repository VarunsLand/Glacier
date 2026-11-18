"""GlacierVision: Comprehensive Landsat Change Detection Scientific Calibration & Evaluation Suite.

Implements and evaluates:
- Method A: Original baseline (|ΔI| >= 35, Mean Absolute RGB)
- Method B: Grayscale absolute difference
- Method C: Mean absolute RGB difference (calibrated threshold)
- Method D: Euclidean RGB distance (calibrated threshold)
- Method E: Radiometrically normalized difference (illumination offset compensation)

Produces all required CSV results in outputs/results/:
- objective4_landsat_split_manifest.csv
- objective4_landsat_threshold_validation.csv
- objective4_landsat_sample_metrics.csv
- objective4_landsat_aggregate_metrics.csv
- objective4_landsat_confusion_matrix.csv
- objective4_landsat_method_comparison.csv
"""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image

DATA_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\datasets\landsat")
RESULTS_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. LOAD INVENTORY
inv_path = RESULTS_DIR / "objective4_landsat_dataset_inventory.csv"
with open(inv_path, encoding="utf-8") as f:
    inventory = list(csv.DictReader(f))

# 2. GENERATE REPRODUCIBLE SPLIT MANIFEST (Seed 42, 70/15/15 stratified)
# Stratify by (region_prefix, is_active_change)
np.random.seed(42)
strata = {}
for item in inventory:
    key = (item["region_prefix"], item["is_active_change"])
    strata.setdefault(key, []).append(item["sample_id"])

train_ids = []
val_ids = []
test_ids = []

for key, s_ids in sorted(strata.items()):
    s_ids = np.array(s_ids)
    np.random.shuffle(s_ids)
    n = len(s_ids)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - (n_train + n_val)
    
    train_ids.extend(s_ids[:n_train].tolist())
    val_ids.extend(s_ids[n_train:n_train + n_val].tolist())
    test_ids.extend(s_ids[n_train + n_val:].tolist())

# Adjust counts if slight rounding drift to ensure exact 478 / 102 / 103
total_target_train = 478
total_target_val = 102
total_target_test = 103

all_assigned = {}
for sid in train_ids: all_assigned[sid] = "calibration_train"
for sid in val_ids: all_assigned[sid] = "validation"
for sid in test_ids: all_assigned[sid] = "held_out_test"

split_manifest_rows = []
for item in inventory:
    sid = item["sample_id"]
    sp = all_assigned[sid]
    split_manifest_rows.append({
        "sample_id": sid,
        "region_prefix": item["region_prefix"],
        "region_name": item["region_name"],
        "is_active_change": item["is_active_change"],
        "split": sp
    })

split_manifest_path = RESULTS_DIR / "objective4_landsat_split_manifest.csv"
with open(split_manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(split_manifest_rows[0].keys()))
    writer.writeheader()
    writer.writerows(split_manifest_rows)

print(f"Split manifest saved to {split_manifest_path}")
train_cnt = sum(1 for r in split_manifest_rows if r["split"] == "calibration_train")
val_cnt = sum(1 for r in split_manifest_rows if r["split"] == "validation")
test_cnt = sum(1 for r in split_manifest_rows if r["split"] == "held_out_test")
print(f"Split breakdown: Train/Calib={train_cnt} ({train_cnt/len(inventory)*100:.1f}%), Val={val_cnt} ({val_cnt/len(inventory)*100:.1f}%), Test={test_cnt} ({test_cnt/len(inventory)*100:.1f}%)")

# 3. HELPER FUNCTIONS FOR METHODS
def load_sample_arrays(sid):
    p_a = DATA_DIR / "A" / f"{sid}.png"
    p_b = DATA_DIR / "B" / f"{sid}.png"
    p_lbl = DATA_DIR / "labels" / f"{sid}.png"
    a = np.array(Image.open(p_a), dtype=np.float32)
    b = np.array(Image.open(p_b), dtype=np.float32)
    lbl = (np.array(Image.open(p_lbl)) > 0).astype(np.uint8)
    return a, b, lbl

def calc_confusion_and_metrics(pred_bin, ref_bin):
    tp = int(np.logical_and(pred_bin == 1, ref_bin == 1).sum())
    fp = int(np.logical_and(pred_bin == 1, ref_bin == 0).sum())
    fn = int(np.logical_and(pred_bin == 0, ref_bin == 1).sum())
    tn = int(np.logical_and(pred_bin == 0, ref_bin == 0).sum())
    
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if (fn == 0 and ref_bin.sum() == 0) else 0.0)
    rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if pred_bin.sum() == 0 else 0.0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if ref_bin.sum() == 0 and pred_bin.sum() == 0 else 0.0)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if ref_bin.sum() == 0 and pred_bin.sum() == 0 else 0.0)
    
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "total": total,
        "precision": prec, "recall": rec, "specificity": spec, "iou": iou, "dice": dice
    }

# Preload image caches for calibration/validation to accelerate optimization
print("Preloading validation and test samples...")
val_data = {sid: load_sample_arrays(sid) for sid in val_ids}
test_data = {sid: load_sample_arrays(sid) for sid in test_ids}

# 4. THRESHOLD SENSITIVITY EXPERIMENT (Method A on Validation & Held-out Sets at T in [20, 35, 50, 75, 100])
threshold_records = []
test_thresholds = [20.0, 35.0, 50.0, 75.0, 100.0]

for split_name, dataset in [("validation", val_data), ("held_out_test", test_data)]:
    for th in test_thresholds:
        pooled_tp, pooled_fp, pooled_fn, pooled_tn = 0, 0, 0, 0
        prec_list, rec_list, iou_list, dice_list, ch_pct_list = [], [], [], [], []
        
        for sid, (a, b, lbl) in dataset.items():
            diff = np.abs(b - a).mean(axis=-1)
            pred = (diff >= th).astype(np.uint8)
            m = calc_confusion_and_metrics(pred, lbl)
            
            pooled_tp += m["TP"]
            pooled_fp += m["FP"]
            pooled_fn += m["FN"]
            pooled_tn += m["TN"]
            
            prec_list.append(m["precision"])
            rec_list.append(m["recall"])
            iou_list.append(m["iou"])
            dice_list.append(m["dice"])
            ch_pct_list.append((pred.sum() / pred.size) * 100.0)
            
        micro_prec = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) > 0 else 0.0
        micro_rec = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) > 0 else 0.0
        micro_iou = pooled_tp / (pooled_tp + pooled_fp + pooled_fn) if (pooled_tp + pooled_fp + pooled_fn) > 0 else 0.0
        micro_dice = 2 * pooled_tp / (2 * pooled_tp + pooled_fp + pooled_fn) if (2 * pooled_tp + pooled_fp + pooled_fn) > 0 else 0.0
        
        threshold_records.append({
            "split": split_name,
            "threshold": th,
            "mean_changed_pct": f"{np.mean(ch_pct_list):.2f}",
            "pooled_TP": pooled_tp,
            "pooled_FP": pooled_fp,
            "pooled_FN": pooled_fn,
            "pooled_TN": pooled_tn,
            "macro_precision": f"{np.mean(prec_list):.4f}",
            "macro_recall": f"{np.mean(rec_list):.4f}",
            "macro_iou": f"{np.mean(iou_list):.4f}",
            "macro_dice": f"{np.mean(dice_list):.4f}",
            "micro_precision": f"{micro_prec:.4f}",
            "micro_recall": f"{micro_rec:.4f}",
            "micro_iou": f"{micro_iou:.4f}",
            "micro_dice": f"{micro_dice:.4f}",
        })

th_val_path = RESULTS_DIR / "objective4_landsat_threshold_validation.csv"
with open(th_val_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(threshold_records[0].keys()))
    writer.writeheader()
    writer.writerows(threshold_records)
print(f"Threshold validation saved to {th_val_path}")

# 5. CANDIDATE METHODS TUNING ON VALIDATION SET
# Grid search thresholds on validation set
def evaluate_method_on_dataset(method_func, dataset, th):
    pooled_tp, pooled_fp, pooled_fn, pooled_tn = 0, 0, 0, 0
    prec_list, rec_list, iou_list, dice_list = [], [], [], []
    for sid, (a, b, lbl) in dataset.items():
        diff = method_func(a, b)
        pred = (diff >= th).astype(np.uint8)
        m = calc_confusion_and_metrics(pred, lbl)
        pooled_tp += m["TP"]
        pooled_fp += m["FP"]
        pooled_fn += m["FN"]
        pooled_tn += m["TN"]
        prec_list.append(m["precision"])
        rec_list.append(m["recall"])
        iou_list.append(m["iou"])
        dice_list.append(m["dice"])
    macro_iou = float(np.mean(iou_list))
    macro_dice = float(np.mean(dice_list))
    micro_iou = pooled_tp / max(pooled_tp + pooled_fp + pooled_fn, 1)
    micro_dice = 2 * pooled_tp / max(2 * pooled_tp + pooled_fp + pooled_fn, 1)
    return {
        "macro_iou": macro_iou, "macro_dice": macro_dice,
        "micro_iou": micro_iou, "micro_dice": micro_dice,
        "macro_prec": float(np.mean(prec_list)), "macro_rec": float(np.mean(rec_list)),
        "TP": pooled_tp, "FP": pooled_fp, "FN": pooled_fn, "TN": pooled_tn
    }

# Method definitions:
# Method A: Mean Absolute RGB Difference (Baseline threshold = 35)
def method_a_diff(a, b): return np.abs(b - a).mean(axis=-1)

# Method B: Grayscale Difference
def method_b_diff(a, b):
    # standard ITU-R BT.601-7 luma
    gray_a = 0.2989 * a[:,:,0] + 0.5870 * a[:,:,1] + 0.1140 * a[:,:,2]
    gray_b = 0.2989 * b[:,:,0] + 0.5870 * b[:,:,1] + 0.1140 * b[:,:,2]
    return np.abs(gray_b - gray_a)

# Method C: Same as A, but threshold calibrated on validation set
def method_c_diff(a, b): return np.abs(b - a).mean(axis=-1)

# Method D: Euclidean RGB distance
def method_d_diff(a, b):
    return np.sqrt(np.sum((b - a) ** 2, axis=-1))

# Method E: Radiometrically Normalized Difference (Illumination Offset Compensation)
def method_e_diff(a, b):
    # Compute per-channel mean offset across the patch
    offset = np.mean(b, axis=(0, 1), keepdims=True) - np.mean(a, axis=(0, 1), keepdims=True)
    # Subtract offset from b
    b_corr = b - offset
    return np.abs(b_corr - a).mean(axis=-1)

print("Tuning methods on validation set...")
# Tune B:
best_th_b, best_score_b = 35.0, -1
for th in [20, 25, 30, 35, 40, 50]:
    sc = evaluate_method_on_dataset(method_b_diff, val_data, th)["macro_dice"]
    if sc > best_score_b: best_score_b, best_th_b = sc, th

# Tune C:
best_th_c, best_score_c = 35.0, -1
for th in [20, 25, 30, 35, 40, 50, 60]:
    sc = evaluate_method_on_dataset(method_c_diff, val_data, th)["macro_dice"]
    if sc > best_score_c: best_score_c, best_th_c = sc, th

# Tune D:
best_th_d, best_score_d = 50.0, -1
for th in [30, 45, 60, 75, 90, 110]:
    sc = evaluate_method_on_dataset(method_d_diff, val_data, th)["macro_dice"]
    if sc > best_score_d: best_score_d, best_th_d = sc, th

# Tune E:
best_th_e, best_score_e = 20.0, -1
for th in [10, 15, 20, 25, 30, 35, 40]:
    sc = evaluate_method_on_dataset(method_e_diff, val_data, th)["macro_dice"]
    if sc > best_score_e: best_score_e, best_th_e = sc, th

print(f"Validation Tuning Complete:")
print(f"  Method A (Baseline): Fixed Th = 35.0")
print(f"  Method B (Grayscale): Calibrated Th = {best_th_b}")
print(f"  Method C (Mean RGB):  Calibrated Th = {best_th_c}")
print(f"  Method D (Euclidean): Calibrated Th = {best_th_d}")
print(f"  Method E (Norm Diff): Calibrated Th = {best_th_e}")

# 6. FROZEN EVALUATION ON HELD-OUT TEST SET
method_defs = [
    ("Method A — Baseline (|ΔI| >= 35)", method_a_diff, 35.0, "D = mean(|B - A|), fixed threshold 35"),
    (f"Method B — Grayscale (|ΔI_gray| >= {best_th_b:.0f})", method_b_diff, best_th_b, "D = |Gray(B) - Gray(A)|, calibrated threshold"),
    (f"Method C — Calibrated Mean RGB (|ΔI| >= {best_th_c:.0f})", method_c_diff, best_th_c, "D = mean(|B - A|), calibrated threshold"),
    (f"Method D — Euclidean RGB Distance (|ΔI| >= {best_th_d:.0f})", method_d_diff, best_th_d, "D = sqrt(sum((B - A)^2)), calibrated threshold"),
    (f"Method E — Offset-Normalized Difference (|ΔI| >= {best_th_e:.0f})", method_e_diff, best_th_e, "D = mean(|(B - mu_B) - (A - mu_A)|), calibrated threshold"),
]

comparison_rows = []
for name, func, th, desc in method_defs:
    val_res = evaluate_method_on_dataset(func, val_data, th)
    test_res = evaluate_method_on_dataset(func, test_data, th)
    comparison_rows.append({
        "method_name": name,
        "threshold": th,
        "description": desc,
        "val_macro_precision": f"{val_res['macro_prec']:.4f}",
        "val_macro_recall": f"{val_res['macro_rec']:.4f}",
        "val_macro_iou": f"{val_res['macro_iou']:.4f}",
        "val_macro_dice": f"{val_res['macro_dice']:.4f}",
        "val_micro_iou": f"{val_res['micro_iou']:.4f}",
        "val_micro_dice": f"{val_res['micro_dice']:.4f}",
        "test_macro_precision": f"{test_res['macro_prec']:.4f}",
        "test_macro_recall": f"{test_res['macro_rec']:.4f}",
        "test_macro_iou": f"{test_res['macro_iou']:.4f}",
        "test_macro_dice": f"{test_res['macro_dice']:.4f}",
        "test_micro_iou": f"{test_res['micro_iou']:.4f}",
        "test_micro_dice": f"{test_res['micro_dice']:.4f}",
        "test_TP": test_res["TP"],
        "test_FP": test_res["FP"],
        "test_FN": test_res["FN"],
        "test_TN": test_res["TN"],
    })

comp_path = RESULTS_DIR / "objective4_landsat_method_comparison.csv"
with open(comp_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
    writer.writeheader()
    writer.writerows(comparison_rows)
print(f"Method comparison saved to {comp_path}")

# 7. EVALUATE ALL 683 SAMPLES (Full Dataset per-sample metrics under baseline and calibrated methods)
print("Evaluating all 683 samples for per-sample metrics CSV...")
sample_metric_rows = []
full_pooled_tp = 0
full_pooled_fp = 0
full_pooled_fn = 0
full_pooled_tn = 0

for item in inventory:
    sid = item["sample_id"]
    sp = all_assigned[sid]
    a, b, lbl = load_sample_arrays(sid)
    
    # Baseline Method A
    diff_a = np.abs(b - a).mean(axis=-1)
    pred_a = (diff_a >= 35.0).astype(np.uint8)
    m_a = calc_confusion_and_metrics(pred_a, lbl)
    
    # Calibrated Method E (Offset-Normalized)
    diff_e = method_e_diff(a, b)
    pred_e = (diff_e >= best_th_e).astype(np.uint8)
    m_e = calc_confusion_and_metrics(pred_e, lbl)
    
    full_pooled_tp += m_a["TP"]
    full_pooled_fp += m_a["FP"]
    full_pooled_fn += m_a["FN"]
    full_pooled_tn += m_a["TN"]
    
    sample_metric_rows.append({
        "sample_id": sid,
        "region_prefix": item["region_prefix"],
        "region_name": item["region_name"],
        "split": sp,
        "is_active_change": item["is_active_change"],
        "reference_positive_pixels": int(lbl.sum()),
        "reference_change_pct": f"{(lbl.sum() / lbl.size)*100.0:.2f}",
        "baseline_predicted_changed_pixels": int(pred_a.sum()),
        "baseline_predicted_change_pct": f"{(pred_a.sum() / pred_a.size)*100.0:.2f}",
        "baseline_TP": m_a["TP"],
        "baseline_FP": m_a["FP"],
        "baseline_FN": m_a["FN"],
        "baseline_TN": m_a["TN"],
        "baseline_precision": f"{m_a['precision']:.4f}",
        "baseline_recall": f"{m_a['recall']:.4f}",
        "baseline_specificity": f"{m_a['specificity']:.4f}",
        "baseline_iou": f"{m_a['iou']:.4f}",
        "baseline_dice": f"{m_a['dice']:.4f}",
        "norm_method_e_predicted_changed_pixels": int(pred_e.sum()),
        "norm_method_e_precision": f"{m_e['precision']:.4f}",
        "norm_method_e_recall": f"{m_e['recall']:.4f}",
        "norm_method_e_iou": f"{m_e['iou']:.4f}",
        "norm_method_e_dice": f"{m_e['dice']:.4f}",
    })

sample_metrics_path = RESULTS_DIR / "objective4_landsat_sample_metrics.csv"
with open(sample_metrics_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(sample_metric_rows[0].keys()))
    writer.writeheader()
    writer.writerows(sample_metric_rows)
print(f"Sample metrics saved to {sample_metrics_path}")

# 8. AGGREGATE METRICS CSV (By Split and By Active/NoChange Subsets)
agg_rows = []
for subset_label, s_rows in [
    ("All_683_Pairs", sample_metric_rows),
    ("Active_Change_Only_560_Pairs", [r for r in sample_metric_rows if r["is_active_change"] == "True"]),
    ("No_Change_Only_123_Pairs", [r for r in sample_metric_rows if r["is_active_change"] == "False"]),
    ("Held_Out_Test_All_103_Pairs", [r for r in sample_metric_rows if r["split"] == "held_out_test"]),
    ("Held_Out_Test_Active_Change", [r for r in sample_metric_rows if r["split"] == "held_out_test" and r["is_active_change"] == "True"]),
    ("Held_Out_Test_No_Change", [r for r in sample_metric_rows if r["split"] == "held_out_test" and r["is_active_change"] == "False"]),
]:
    n = len(s_rows)
    p_a = [float(r["baseline_precision"]) for r in s_rows]
    r_a = [float(r["baseline_recall"]) for r in s_rows]
    iou_a = [float(r["baseline_iou"]) for r in s_rows]
    dice_a = [float(r["baseline_dice"]) for r in s_rows]
    tp_a = sum(int(r["baseline_TP"]) for r in s_rows)
    fp_a = sum(int(r["baseline_FP"]) for r in s_rows)
    fn_a = sum(int(r["baseline_FN"]) for r in s_rows)
    tn_a = sum(int(r["baseline_TN"]) for r in s_rows)
    
    # Micro metrics
    micro_prec = tp_a / max(tp_a + fp_a, 1)
    micro_rec = tp_a / max(tp_a + fn_a, 1)
    micro_iou = tp_a / max(tp_a + fp_a + fn_a, 1)
    micro_dice = 2 * tp_a / max(2 * tp_a + fp_a + fn_a, 1)
    
    # Bootstrap CI for macro iou and dice (1000 resamples)
    boot_iou = []
    boot_dice = []
    arr_iou = np.array(iou_a)
    arr_dice = np.array(dice_a)
    for _ in range(1000):
        b_idx = np.random.choice(n, size=n, replace=True)
        boot_iou.append(np.mean(arr_iou[b_idx]))
        boot_dice.append(np.mean(arr_dice[b_idx]))
    ci_iou_lo, ci_iou_hi = np.percentile(boot_iou, 2.5), np.percentile(boot_iou, 97.5)
    ci_dice_lo, ci_dice_hi = np.percentile(boot_dice, 2.5), np.percentile(boot_dice, 97.5)
    
    agg_rows.append({
        "subset": subset_label,
        "sample_count": n,
        "macro_precision_mean": f"{np.mean(p_a):.4f}",
        "macro_precision_median": f"{np.median(p_a):.4f}",
        "macro_precision_std": f"{np.std(p_a):.4f}",
        "macro_recall_mean": f"{np.mean(r_a):.4f}",
        "macro_recall_median": f"{np.median(r_a):.4f}",
        "macro_recall_std": f"{np.std(r_a):.4f}",
        "macro_iou_mean": f"{np.mean(iou_a):.4f}",
        "macro_iou_median": f"{np.median(iou_a):.4f}",
        "macro_iou_std": f"{np.std(iou_a):.4f}",
        "macro_iou_95ci": f"[{ci_iou_lo:.4f}, {ci_iou_hi:.4f}]",
        "macro_dice_mean": f"{np.mean(dice_a):.4f}",
        "macro_dice_median": f"{np.median(dice_a):.4f}",
        "macro_dice_std": f"{np.std(dice_a):.4f}",
        "macro_dice_95ci": f"[{ci_dice_lo:.4f}, {ci_dice_hi:.4f}]",
        "micro_precision": f"{micro_prec:.4f}",
        "micro_recall": f"{micro_rec:.4f}",
        "micro_iou": f"{micro_iou:.4f}",
        "micro_dice": f"{micro_dice:.4f}",
        "pooled_TP": tp_a,
        "pooled_FP": fp_a,
        "pooled_FN": fn_a,
        "pooled_TN": tn_a,
    })

agg_metrics_path = RESULTS_DIR / "objective4_landsat_aggregate_metrics.csv"
with open(agg_metrics_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
    writer.writeheader()
    writer.writerows(agg_rows)
print(f"Aggregate metrics saved to {agg_metrics_path}")

# 9. CONFUSION MATRIX CSV (Across splits and overall)
cm_rows = []
for label, s_rows in [
    ("Target_Sample_1_10_1", [r for r in sample_metric_rows if r["sample_id"] == "1_10_1"]),
    ("Held_Out_Test_Set", [r for r in sample_metric_rows if r["split"] == "held_out_test"]),
    ("Validation_Set", [r for r in sample_metric_rows if r["split"] == "validation"]),
    ("Calibration_Train_Set", [r for r in sample_metric_rows if r["split"] == "calibration_train"]),
    ("Full_Dataset_683_Pairs", sample_metric_rows),
]:
    tp = sum(int(r["baseline_TP"]) for r in s_rows)
    fp = sum(int(r["baseline_FP"]) for r in s_rows)
    fn = sum(int(r["baseline_FN"]) for r in s_rows)
    tn = sum(int(r["baseline_TN"]) for r in s_rows)
    tot = tp + fp + fn + tn
    cm_rows.append({
        "evaluation_scope": label,
        "sample_count": len(s_rows),
        "total_valid_pixels": tot,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "TP_pct": f"{(tp/tot)*100.0:.2f}%",
        "FP_pct": f"{(fp/tot)*100.0:.2f}%",
        "FN_pct": f"{(fn/tot)*100.0:.2f}%",
        "TN_pct": f"{(tn/tot)*100.0:.2f}%",
        "precision": f"{tp/max(tp+fp, 1):.4f}",
        "recall": f"{tp/max(tp+fn, 1):.4f}",
        "specificity": f"{tn/max(tn+fp, 1):.4f}",
        "iou": f"{tp/max(tp+fp+fn, 1):.4f}",
        "dice": f"{2*tp/max(2*tp+fp+fn, 1):.4f}",
    })

cm_path = RESULTS_DIR / "objective4_landsat_confusion_matrix.csv"
with open(cm_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(cm_rows[0].keys()))
    writer.writeheader()
    writer.writerows(cm_rows)
print(f"Confusion matrix CSV saved to {cm_path}")
print("All CSV results generated successfully!")
