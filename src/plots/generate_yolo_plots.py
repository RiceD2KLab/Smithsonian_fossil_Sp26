import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.font_manager as fm
import seaborn as sns

"""
Generates plots for YOLO training metrics from a results.csv file.
Assumes the results.csv file has the relevant columns for training 
and validation losses, precision, recall, and mAP.
"""

# get script dir
script_dir = os.path.dirname(os.path.abspath(__file__))

# load custom font from fonts folder
fm.fontManager.addfont(os.path.join(script_dir, "fonts", "RobotoMono-Regular.ttf"))
fm.fontManager.addfont(os.path.join(script_dir, "fonts", "RobotoMono-Bold.ttf"))
plt.rcParams["font.family"] = "Roboto Mono"

# assumes results is in top level directory
csv_path = os.path.join(script_dir, "..", "..", "results.csv")

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
epochs = df["epoch"].values

# creates subdirectory for plots
SIGMA = 4
OUT_DIR = os.path.join(script_dir, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")

BLUE   = "#4c8cbf"
ORANGE = "#e07b39"

def smooth(col):
    s = pd.Series(df[col].values.astype(float)).interpolate(method="linear").ffill().bfill().values
    return gaussian_filter1d(s, sigma=SIGMA, mode="nearest")

def save(fig, fname):
    path = os.path.join(OUT_DIR, f"{fname}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {path}")

def single_plot(col, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.despine(ax=ax, top=True, right=True)
    ax.plot(epochs, smooth(col), color=ORANGE, linewidth=2.0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlim(epochs[0], epochs[-1])
    fig.tight_layout()
    save(fig, fname)

def combined_plot(series, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.despine(ax=ax, top=True, right=True)
    for col, label, color in series:
        ax.plot(epochs, smooth(col), color=color, linewidth=2.0, label=label)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, fname)

single_plot("train/box_loss",        "Train Box Loss",                 "Loss",      "train_box_loss")
single_plot("train/cls_loss",        "Train Classification Loss",      "Loss",      "train_cls_loss")
single_plot("train/dfl_loss",        "Train DFL Loss",                 "Loss",      "train_dfl_loss")
single_plot("val/box_loss",          "Validation Box Loss",            "Loss",      "val_box_loss")
single_plot("val/cls_loss",          "Validation Classification Loss", "Loss",      "val_cls_loss")
single_plot("val/dfl_loss",          "Validation DFL Loss",            "Loss",      "val_dfl_loss")
single_plot("metrics/precision(B)",  "Precision",                      "Precision", "precision")
single_plot("metrics/recall(B)",     "Recall",                         "Recall",    "recall")
single_plot("metrics/mAP50(B)",      "mAP@50",                         "mAP",       "mAP50")
single_plot("metrics/mAP50-95(B)",   "mAP@50-95",                      "mAP",       "mAP50_95")

combined_plot(
    [("train/box_loss", "Train", BLUE), ("val/box_loss", "Validation", ORANGE)],
    "Box Loss: Train vs. Validation", "Loss", "box_loss_combined"
)
combined_plot(
    [("train/cls_loss", "Train", BLUE), ("val/cls_loss", "Validation", ORANGE)],
    "Classification Loss: Train vs. Validation", "Loss", "cls_loss_combined"
)
combined_plot(
    [("train/dfl_loss", "Train", BLUE), ("val/dfl_loss", "Validation", ORANGE)],
    "DFL Loss: Train vs. Validation", "Loss", "dfl_loss_combined"
)
combined_plot(
    [("metrics/precision(B)", "Precision", BLUE), ("metrics/recall(B)", "Recall", ORANGE)],
    "Precision & Recall", "Score", "precision_recall_combined"
)
combined_plot(
    [("metrics/mAP50(B)", "mAP@50", BLUE), ("metrics/mAP50-95(B)", "mAP@50-95", ORANGE)],
    "mAP@50 & mAP@50-95", "mAP", "map_combined"
)

print(f"\nDone. Plots saved to {OUT_DIR}")