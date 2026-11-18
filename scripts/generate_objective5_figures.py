"""
generate_objective5_figures.py
Generates the publication-grade 3-panel comparison figure for Objective 5:
1. Performance Metrics Bar Chart (Accuracy, Precision, Recall, Specificity, F1-Score)
2. Zero-Shot Handcrafted CLIP Confusion Matrix (TN=23, FP=27, FN=5, TP=45)
3. Learnable CoOp (M=16) Confusion Matrix (TN=38, FP=12, FN=11, TP=39)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Canonical metrics
    metrics_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]
    clip_vals = [0.6800, 0.6250, 0.9000, 0.4600, 0.7377]
    coop_vals = [0.7700, 0.7647, 0.7800, 0.7600, 0.7723]

    # Confusion matrices: [[TN, FP], [FN, TP]]
    cm_clip = np.array([[23, 27], [5, 45]])
    cm_coop = np.array([[38, 12], [11, 39]])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # 1. Bar Chart
    x = np.arange(len(metrics_names))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, clip_vals, width, label="Zero-Shot Handcrafted CLIP", color="#4ECDC4", edgecolor="#1A535C", alpha=0.9)
    rects2 = axes[0].bar(x + width/2, coop_vals, width, label="Learnable CoOp (M=16 Context)", color="#00F2FE", edgecolor="#0077B6", alpha=0.95)

    axes[0].set_ylabel("Score", fontsize=12, fontweight="bold")
    axes[0].set_title("Evaluation Metrics (N=100 Held-Out Patches)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    axes[0].set_ylim(0.0, 1.08)
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # Add data labels on bars
    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1A535C")
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0077B6")

    # 2. CLIP Confusion Matrix
    im1 = axes[1].imshow(cm_clip, cmap="Purples", vmin=0, vmax=50)
    axes[1].set_title("Zero-Shot CLIP Confusion Matrix\n(Accuracy: 68.0%, 27 False Alarms)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Non-Glacier\n(Bedrock)", "Glacier Ice"], fontweight="bold", fontsize=10)
    axes[1].set_yticklabels(["Non-Glacier\n(Bedrock)", "Glacier Ice"], fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Predicted Class", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Ground Truth Class", fontweight="bold", fontsize=11)
    
    cell_labels_clip = [
        [f"TN = 23\n(46.0%)", f"FP = 27\n(54.0%)"],
        [f"FN = 5\n(10.0%)", f"TP = 45\n(90.0%)"]
    ]
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, cell_labels_clip[i][j], ha="center", va="center",
                         color="white" if cm_clip[i, j] > 25 else "#2D0C4E", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. CoOp Confusion Matrix
    im2 = axes[2].imshow(cm_coop, cmap="Blues", vmin=0, vmax=50)
    axes[2].set_title("Learnable CoOp Confusion Matrix\n(Accuracy: 77.0%, FP Reduced to 12)", fontsize=12, fontweight="bold", pad=10)
    axes[2].set_xticks([0, 1]); axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Non-Glacier\n(Bedrock)", "Glacier Ice"], fontweight="bold", fontsize=10)
    axes[2].set_yticklabels(["Non-Glacier\n(Bedrock)", "Glacier Ice"], fontweight="bold", fontsize=10)
    axes[2].set_xlabel("Predicted Class", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("Ground Truth Class", fontweight="bold", fontsize=11)

    cell_labels_coop = [
        [f"TN = 38\n(76.0%)", f"FP = 12\n(24.0%)"],
        [f"FN = 11\n(22.0%)", f"TP = 39\n(78.0%)"]
    ]
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, cell_labels_coop[i][j], ha="center", va="center",
                         color="white" if cm_coop[i, j] > 25 else "#002855", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_fig = FIG_DIR / "clip_vs_coop_comparison.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3-panel comparison figure to: {out_fig}")

if __name__ == "__main__":
    main()
