"""
Generates all figures for the final Results section of the paper.

Outputs:
    map_comparison_final.pdf/.png      — grouped bar chart, all 4 final configurations
    tuning_trajectory.pdf/.png         — RF-DETR Bayesian trial validation AP@50
    preprocessing_sensitivity.pdf/.png — AP@50 by preprocessing strategy, per model
    pr_curve_final.png                 — copied from Final Hyperparameter Results
    yolo_tuned_training_curves.pdf/.png  — YOLO26-L training curves (both conditions)
    rfdetr_tuned_training_curves.pdf/.png — RF-DETR 2XL training curves (both conditions)

Run from the repo root:
    python -m src.plots.generate_results_plots

Requires: matplotlib, seaborn, numpy
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# Paths
REPO_ROOT = "absolute/path/to/root"
OUT_DIR   = REPO_ROOT / "output" / "dir"
TUNE_DIR  = REPO_ROOT / "tune" / "dir"

YOLO_FS_CSV   = TUNE_DIR / "YOLO - Focus Stack Result"    / "results.csv"
YOLO_BP_CSV   = TUNE_DIR / "YOLO - Best Plane Result"     / "results.csv"
RFDETR_FS_LOG = TUNE_DIR / "RF-DETR - Focus Stack Result" / "log.txt"
RFDETR_BP_LOG = TUNE_DIR / "RF-DETR - Best Plane Result"  / "log.txt"

# Style
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":     150,
})

BLUE_DARK    = "#2c6fad"
BLUE_LIGHT   = "#7ab8e8"
ORANGE_DARK  = "#c0392b"
ORANGE_LIGHT = "#e8967a"


# Model Comparison Bar Chart
# NOTE: Make sure to edit values here!
def plot_model_comparison() -> None:
    """Two-panel grouped bar chart: AP@50 (top) and AP@50-95 (bottom), Y-axis 0-1."""
    configs = [
        ("YOLO26-L",    "Focus-Stacked", 0.826, 0.594),
        ("YOLO26-L",    "Best Plane",    0.860, 0.604),
        ("RF-DETR 2XL", "Focus-Stacked", 0.879, 0.642),
        ("RF-DETR 2XL", "Best Plane",    0.877, 0.637),
    ]
    x_labels = [f"{m}\n({p})" for m, p, *_ in configs]
    ap50   = [r[2] for r in configs]
    ap5095 = [r[3] for r in configs]
    colors = [BLUE_DARK, BLUE_LIGHT, ORANGE_DARK, ORANGE_LIGHT]
    x      = np.arange(len(configs))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.subplots_adjust(hspace=0.06)

    for ax, vals, metric in [
        (ax1, ap50,   "AP@50"),
        (ax2, ap5095, "AP@50–95"),
    ]:
        bars = ax.bar(x, vals, width=0.55, color=colors,
                      edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=10)

    legend_handles = [
        Patch(facecolor=c, label=f"{m} / {p}")
        for c, (m, p, *_) in zip(colors, configs)
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=10, columnspacing=1.2,
    )
    fig.suptitle(
        "Test-Set Detection Performance — All Final Configurations",
        fontsize=13, y=1.07,
    )

    _save(fig, "map_comparison_final")
    print("Saved map_comparison_final")


# Bayesian Tuning Trajectory
# NOTE: Make sure to edit values here!
def plot_tuning_trajectory() -> None:
    """Scatter+line plot of validation AP@50 across all RF-DETR Bayesian trials."""
    first_bp  = [0.8282, 0.8277, 0.8253, 0.8273, 0.8238, 0.8289,
                 0.8285, 0.8203, 0.8318, 0.8281, 0.8320, 0.8310]
    first_fs  = [0.8341, 0.8431, 0.8334, 0.8414, 0.8337, 0.8289,
                 0.8252, 0.8425, 0.8500, 0.8340, 0.8439, 0.8400]
    second_fs = [0.8408, 0.8435, 0.8436, 0.8459, 0.8411, 0.8288,
                 0.8505, 0.8446, 0.8418, 0.8445, 0.8431]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    def _panel(ax, series, colors, labels, markers, title, ylim):
        for vals, col, lbl, mk in zip(series, colors, labels, markers):
            xi = np.arange(1, len(vals) + 1)
            ax.plot(xi, vals, color=col, lw=1.4, alpha=0.65, zorder=2)
            ax.scatter(xi, vals, color=col, s=45, marker=mk, zorder=3, label=lbl)
            bi = int(np.argmax(vals))
            ax.scatter(
                [xi[bi]], [vals[bi]],
                s=230, marker="*", color=col, zorder=5,
                edgecolors="black", linewidths=0.7,
                label=f"Best (trial {bi + 1}, AP@50 = {vals[bi]:.4f})",
            )
        ax.set_xlabel("Trial Index", fontsize=11)
        ax.set_ylabel("Validation AP@50", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(*ylim)
        ax.set_xlim(0.2, len(max(series, key=len)) + 0.8)
        ax.set_xticks(np.arange(1, len(max(series, key=len)) + 1))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(axis="y", alpha=0.35)

    _panel(ax1, [first_bp], [BLUE_DARK], ["First Search Space"], ["o"],
           "Best-Focal-Plane Condition\n(12 Trials)", (0.817, 0.837))
    _panel(ax2, [first_fs, second_fs], [BLUE_DARK, ORANGE_DARK],
           ["First Search Space", "Second Search Space (Refined)"], ["o", "s"],
           "Focus-Stacked Condition\n(12 + 11 Trials)", (0.824, 0.857))

    fig.suptitle(
        "RF-DETR 2XL — Bayesian Hyperparameter Search: Validation AP@50 Across All 35 Trials",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "tuning_trajectory")
    print("Saved tuning_trajectory")


# YOLO26-L Training Curves
def _load_yolo_csv(path: Path) -> dict:
    """Load a YOLO results.csv and return a dict of metric lists keyed by name.

    Args:
        path: Path to the results.csv file produced by YOLO training.

    Returns:
        Dict with keys epochs, ap50, ap5095, prec, rec, tr_box, tr_cls,
        vl_box, vl_cls, each holding a list of float values.
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        "epochs": [int(r["epoch"])                  for r in rows],
        "ap50":   [float(r["metrics/mAP50(B)"])     for r in rows],
        "ap5095": [float(r["metrics/mAP50-95(B)"])  for r in rows],
        "prec":   [float(r["metrics/precision(B)"]) for r in rows],
        "rec":    [float(r["metrics/recall(B)"])    for r in rows],
        "tr_box": [float(r["train/box_loss"])       for r in rows],
        "tr_cls": [float(r["train/cls_loss"])       for r in rows],
        "vl_box": [float(r["val/box_loss"])         for r in rows],
        "vl_cls": [float(r["val/cls_loss"])         for r in rows],
    }


def plot_yolo_tuned_training_curves() -> None:
    """2x2 grid: validation metrics and losses for each YOLO26-L condition."""
    fs = _load_yolo_csv(YOLO_FS_CSV)
    bp = _load_yolo_csv(YOLO_BP_CSV)
    best_fs = fs["epochs"][int(np.argmax(fs["ap5095"]))]
    best_bp = bp["epochs"][int(np.argmax(bp["ap5095"]))]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.38, wspace=0.28)

    for row_idx, d, label, best_ep in [
        (0, fs, "Focus-Stacked", best_fs),
        (1, bp, "Best Plane",    best_bp),
    ]:
        ax_m, ax_l = axes[row_idx]

        ax_m.plot(d["epochs"], d["ap50"],   color=BLUE_DARK,      lw=1.6, label="AP@50")
        ax_m.plot(d["epochs"], d["ap5095"], color=ORANGE_DARK,    lw=1.6, label="AP@50-95")
        ax_m.plot(d["epochs"], d["prec"],   color="seagreen",     lw=1.0, ls="--",
                  alpha=0.85, label="Precision")
        ax_m.plot(d["epochs"], d["rec"],    color="mediumpurple", lw=1.0, ls="--",
                  alpha=0.85, label="Recall")
        ax_m.axvline(best_ep, color="gray", lw=1.2, ls=":",
                     label=f"Best ckpt (ep. {best_ep})")
        ax_m.axvspan(best_ep, max(d["epochs"]), alpha=0.06, color="red")
        ax_m.set_xlabel("Epoch")
        ax_m.set_ylabel("Metric")
        ax_m.set_title(f"YOLO26-L ({label}) — Validation Metrics")
        ax_m.set_ylim(0, 1)
        ax_m.legend(fontsize=8, loc="lower right")
        ax_m.grid(alpha=0.3)

        ax_l.plot(d["epochs"], d["tr_box"], color=BLUE_DARK,   lw=1.6,          label="Train Box")
        ax_l.plot(d["epochs"], d["vl_box"], color=BLUE_DARK,   lw=1.0, ls="--", label="Val Box",  alpha=0.85)
        ax_l.plot(d["epochs"], d["tr_cls"], color=ORANGE_DARK, lw=1.6,          label="Train Cls")
        ax_l.plot(d["epochs"], d["vl_cls"], color=ORANGE_DARK, lw=1.0, ls="--", label="Val Cls",  alpha=0.85)
        ax_l.axvline(best_ep, color="gray", lw=1.2, ls=":",
                     label=f"Best ckpt (ep. {best_ep})")
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("Loss")
        ax_l.set_title(f"YOLO26-L ({label}) — Losses")
        ax_l.legend(fontsize=8, loc="upper right")
        ax_l.grid(alpha=0.3)

    fig.suptitle("YOLO26-L Tuned Training Dynamics", fontsize=13, y=1.01)
    _save(fig, "yolo_tuned_training_curves")
    print(f"Saved yolo_tuned_training_curves  (best FS ep: {best_fs}, best BP ep: {best_bp})")


# RF-DETR 2XL Training Curves
def _load_rfdetr_log(path: Path) -> dict:
    """Load an RF-DETR log.txt (one JSON object per line) and return metric lists.

    Args:
        path: Path to the log.txt file produced by RF-DETR training.

    Returns:
        Dict with keys epochs, train_loss, ema_loss, ema_ap5095, ema_ap50,
        each holding a list of values.
    """
    with open(path) as f:
        lines = f.read().strip().split("\n")
    data = [json.loads(l) for l in lines]
    return {
        "epochs":     [d["epoch"]                      for d in data],
        "train_loss": [d["train_loss"]                 for d in data],
        "ema_loss":   [d["ema_test_loss"]              for d in data],
        "ema_ap5095": [d["ema_test_coco_eval_bbox"][0] for d in data],
        "ema_ap50":   [d["ema_test_coco_eval_bbox"][1] for d in data],
    }


def plot_rfdetr_tuned_training_curves() -> None:
    """2x2 grid: EMA validation metrics and losses for each RF-DETR 2XL condition."""
    fs = _load_rfdetr_log(RFDETR_FS_LOG)
    bp = _load_rfdetr_log(RFDETR_BP_LOG)
    best_fs = fs["epochs"][int(np.argmax(fs["ema_ap5095"]))]
    best_bp = bp["epochs"][int(np.argmax(bp["ema_ap5095"]))]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.38, wspace=0.28)

    for row_idx, d, label, best_ep in [
        (0, fs, "Focus-Stacked", best_fs),
        (1, bp, "Best Plane",    best_bp),
    ]:
        ax_m, ax_l = axes[row_idx]

        ax_m.plot(d["epochs"], d["ema_ap50"],   color=BLUE_DARK,   lw=1.6, label="EMA AP@50")
        ax_m.plot(d["epochs"], d["ema_ap5095"], color=ORANGE_DARK, lw=1.6, label="EMA AP@50–95")
        ax_m.axvline(best_ep, color="gray", lw=1.2, ls=":",
                     label=f"Best ckpt (ep. {best_ep})")
        ax_m.axvspan(best_ep, max(d["epochs"]), alpha=0.06, color="red")
        ax_m.set_xlabel("Epoch")
        ax_m.set_ylabel("EMA Metric (Validation Split)")
        ax_m.set_title(f"RF-DETR 2XL ({label}) — EMA Metrics")
        ax_m.set_ylim(0.2, 1.0)
        ax_m.legend(fontsize=9, loc="lower right")
        ax_m.grid(alpha=0.3)

        ax_l.plot(d["epochs"], d["train_loss"], color=BLUE_DARK,   lw=1.6,          label="Train Loss")
        ax_l.plot(d["epochs"], d["ema_loss"],   color=ORANGE_DARK, lw=1.6, ls="--", label="EMA Val Loss")
        ax_l.axvline(best_ep, color="gray", lw=1.2, ls=":",
                     label=f"Best ckpt (ep. {best_ep})")
        ax_l.axvspan(best_ep, max(d["epochs"]), alpha=0.06, color="red")
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("Loss")
        ax_l.set_title(f"RF-DETR 2XL ({label}) — Losses")
        ax_l.legend(fontsize=9, loc="upper right")
        ax_l.grid(alpha=0.3)

    fig.suptitle("RF-DETR 2XL Tuned Training Dynamics", fontsize=13, y=1.01)
    _save(fig, "rfdetr_tuned_training_curves")
    print(f"Saved rfdetr_tuned_training_curves  (best FS ep: {best_fs}, best BP ep: {best_bp})")


def _save(fig: plt.Figure, stem: str) -> None:
    """Save a figure as both PDF and PNG under OUT_DIR, then close it.

    Args:
        fig: Matplotlib figure to save.
        stem: Base filename without extension.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


# Entry point
if __name__ == "__main__":
    plot_model_comparison()
    plot_tuning_trajectory()
    plot_preprocessing_sensitivity()
    plot_yolo_tuned_training_curves()
    plot_rfdetr_tuned_training_curves()
    print("\nAll figures saved to:", OUT_DIR)
