"""Verify baseline on sample 1_10_1."""
import numpy as np
from PIL import Image
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\datasets\landsat")
p_a = DATA_DIR / "A" / "1_10_1.png"
p_b = DATA_DIR / "B" / "1_10_1.png"
p_lbl = DATA_DIR / "labels" / "1_10_1.png"

im_a = np.array(Image.open(p_a))
im_b = np.array(Image.open(p_b))
im_lbl = np.array(Image.open(p_lbl))

diff_mag = np.abs(im_b.astype(np.float32) - im_a.astype(np.float32)).mean(axis=-1)
pred_mask = (diff_mag >= 35.0).astype(np.uint8)
ref_mask = (im_lbl > 0).astype(np.uint8)

tp = int(np.logical_and(pred_mask == 1, ref_mask == 1).sum())
fp = int(np.logical_and(pred_mask == 1, ref_mask == 0).sum())
fn = int(np.logical_and(pred_mask == 0, ref_mask == 1).sum())
tn = int(np.logical_and(pred_mask == 0, ref_mask == 0).sum())

total = tp + fp + fn + tn
prec = tp / (tp + fp)
rec = tp / (tp + fn)
iou = tp / (tp + fp + fn)
dice = 2 * tp / (2 * tp + fp + fn)
spec = tn / (tn + fp)

print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TN: {tn}")
print(f"Total: {total}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"IoU: {iou:.4f}")
print(f"Dice: {dice:.4f}")
print(f"Specificity: {spec:.4f}")
