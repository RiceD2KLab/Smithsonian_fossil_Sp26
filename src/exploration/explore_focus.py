"""
Focal-plane sharpness exploration for palynomorph slide tiles.

Consumes the output of the preprocessing pipeline (``generate_tiles.py``):
    * ``.npy`` tile files with shape ``(H, W, C, Z)``
    * ``annotations.json`` mapping each tile path to its annotation list

For every annotated palynomorph crop and for every full tile, two
complementary focus metrics are computed across all Z (focal) planes:

1. **Variance of Laplacian (VoL)**
2. **Tenengrad**

Results are exported as:
    * ``focus_scores.csv`` — one row per (palynomorph x Z-plane) with both scores
    * A suite of publication-ready matplotlib figures (PNG, 300 dpi)

Usage
-----
    python -m src.exploration.explore_focus \
        --input  output/tiled_images/<slide_name> \
        --output output/exploration
"""

import argparse
import csv
import json
import os
import sys
import concurrent.futures
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from exploration.focus_metrics import tenengrad, variance_of_laplacian
from data.ndpi_reader import NDPIData
from data.ndpa_reader import NDPAData
from data.util import bounds_to_pixels
from data.generate_tiles import compute_tile_grid

# ── Matplotlib house style (academic) ─────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# ── Colours ───────────────────────────────────────────────────────────────
COLOR_VOL = "#2C73D2"       # variance of laplacian
COLOR_TEN = "#E85D04"       # tenengrad


# ===================================================================== #
#  Data loading                                                          #
# ===================================================================== #

def load_tiles_and_annotations(input_dir: str):
    """Return ``(tiles, annotations)`` from a preprocessing output dir.

    Parameters
    ----------
    input_dir : str
        Directory produced by ``generate_tiles.py``, containing a ``tiles/``
        sub-directory of ``.npy`` files and an ``annotations.json`` file.

    Returns
    -------
    tiles : list[dict]
        Each dict: ``{"path": str, "data": np.ndarray}``  (H, W, C, Z).
    annotations : dict
        Mapping from tile relative path → list of annotation dicts.
    """
    ann_path = os.path.join(input_dir, "annotations.json")
    with open(ann_path) as f:
        annotations = json.load(f)

    tiles = []
    tiles_dir = os.path.join(input_dir, "tiles")
    for fname in sorted(os.listdir(tiles_dir)):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(tiles_dir, fname)
        data = np.load(fpath)
        rel_path = os.path.join("tiles", fname)
        tiles.append({"path": rel_path, "data": data})

    return tiles, annotations


# ===================================================================== #
#  Focus scoring                                                         #
# ===================================================================== #

def score_region(image_rgb: np.ndarray) -> dict:
    """Compute both focus metrics for one RGB image.

    Parameters
    ----------
    image_rgb : np.ndarray
        (H, W, 3) uint8 array.

    Returns
    -------
    dict with keys ``"vol"`` and ``"tenengrad"``.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return {
        "vol":       variance_of_laplacian(gray),
        "tenengrad": tenengrad(gray, 3),
    }


def compute_per_roi_scores(tiles, annotations, filter_clipped=False):
    """Score every annotated palynomorph across all focal planes.

    Parameters
    ----------
    tiles : list[dict]
    annotations : dict
    filter_clipped : bool
        If True, skip any annotation whose bounding box touches
        the tile boundary (0 or tile size).

    Returns
    -------
    records : list[dict]
        One dict per (palynomorph, Z-plane) with keys:
        ``tile_path, ann_idx, label, z_index, vol, tenengrad``
    n_z : int
        Number of focal planes.
    """
    records = []
    for tile_info in tiles:
        rel = tile_info["path"]
        data = tile_info["data"]       # (H, W, C, Z)
        h_full, w_full = data.shape[:2]
        anns = annotations.get(rel, [])
        n_z = data.shape[3] if data.ndim == 4 else 1

        for ann_idx, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            label = ann.get("label", "unknown")

            # Optional clipping filter
            if filter_clipped:
                # If any edge touches image boundary, it is potentially cut off
                if (x <= 0 or y <= 0 or x + w >= w_full or y + h >= h_full):
                    continue

            for z in range(n_z):
                plane = data[:, :, :, z] if data.ndim == 4 else data
                crop = plane[y: y + h, x: x + w]
                if crop.size == 0:
                    continue
                records.append({
                    "tile_path": rel,
                    "ann_idx":   ann_idx,
                    "label":     label,
                    "z_index":   z,
                    "vol":       scores["vol"],
                    "tenengrad": scores["tenengrad"],
                    "bbox":      [x, y, w, h],
                })
    return records, n_z


def compute_dataset_wide_scores(tiles):
    """Score each full tile across all focal planes.

    Returns
    -------
    records : list[dict]
        One dict per (tile, Z-plane).
    n_z : int
    """
    records = []
    n_z = 0
    for tile_info in tiles:
        data = tile_info["data"]
        n_z = data.shape[3] if data.ndim == 4 else 1
        for z in range(n_z):
            plane = data[:, :, :, z] if data.ndim == 4 else data
            scores = score_region(plane)
            records.append({
                "tile_path": tile_info["path"],
                "z_index":   z,
                "vol":       scores["vol"],
                "tenengrad": scores["tenengrad"],
            })
    return records, n_z


# ===================================================================== #
#  In-Memory Raw Processing Pipeline                                     #
# ===================================================================== #

def process_raw_dataset(raw_dir: str, filter_clipped: bool = False,
                        magnification: float = 40.0, tile_size: int = 1024,
                        overlap: float = 0.0):
    """
    Process raw NDPI/NDPA files directly in memory without saving tiles to disk.

    Extracts one tile at a time, computes global and per-ROI focus scores,
    and then drops the tile from memory.

    Returns
    -------
    roi_records, ds_records, n_z
    """
    roi_records = []
    ds_records = []
    n_z_global = 0

    files = [f for f in os.listdir(raw_dir) if f.endswith(".ndpi")]
    
    for file in files:
        ndpi_path = os.path.join(raw_dir, file)
        ndpa_path = ndpi_path + ".ndpa"
        if not os.path.exists(ndpa_path):
            print(f"  [Skip] No .ndpa found for {file}")
            continue

        print(f"  Processing raw slide: {file}")
        ndpi_data = NDPIData(ndpi_path)
        ndpa_data = NDPAData(ndpa_path)
        slide_name = os.path.splitext(file)[0]

        # 1. Generate the theoretical tile grid
        print("Step 1: Generate the theoretical tile grid")
        tiles = []
        for region in ndpa_data.rois:
            bounds = bounds_to_pixels(region.bounds, magnification, ndpi_data.metadata)
            tiles.extend(compute_tile_grid(*bounds, tile_size=tile_size, overlap=overlap))

        # 2. Extract global annotations in pixel coords
        print("Step 2: Extract global annotations in pixel coords")
        global_anns = []
        for ann in ndpa_data.palynomorphs:
            bbox = bounds_to_pixels(ann.bounds, magnification, ndpi_data.metadata)
            global_anns.append({
                "bbox": bbox,
                "label": ann.label
            })

        # 3. For each tile, find which annotations intersect it
        tile_annotations = []
        for tile in tiles:
            x0, y0, w, h = tile
            x1, y1 = x0 + w, y0 + h
            anns_in_tile = []
            for i, ann in enumerate(global_anns):
                ax0, ay0, aw, ah = ann["bbox"]
                ax1, ay1 = ax0 + aw, ay0 + ah
                if not (ax1 <= x0 or ax0 >= x1 or ay1 <= y0 or ay0 >= y1):
                    anns_in_tile.append(i)
            tile_annotations.append((tile, anns_in_tile))

        # 4. Extract tile from NDPI, score it, and discard
        def process_tile(tile_data):
            tile, anns_in_tile = tile_data
            local_ds_recs = []
            local_roi_recs = []
            local_n_z = 0
            image = ndpi_data.get_tile(*tile, magnification=magnification)
            # image shape is (H, W, C, Z)
            n_z_local = image.shape[3] if image.ndim == 4 else 1
            local_n_z = max(local_n_z, n_z_local)
            tile_path_id = f"{slide_name}/tile_{tile[0]}_{tile[1]}_{tile[2]}_{tile[3]}.npy"

            # 4a. Dataset-wide scores
            for z in range(n_z_local):
                plane = image[:, :, :, z] if image.ndim == 4 else image
                scores = score_region(plane)
                local_ds_recs.append({
                    "tile_path": tile_path_id,
                    "z_index":   z,
                    "vol":       scores["vol"],
                    "tenengrad": scores["tenengrad"],
                })

            # 4b. Per-ROI scores
            x0, y0, w, h = tile
            for i in anns_in_tile:
                ann = global_anns[i]
                ax0, ay0, aw, ah = ann["bbox"]
                rel_x = ax0 - x0
                rel_y = ay0 - y0

                # Determine standard cropped bbox within tile
                crop_x = max(0, rel_x)
                crop_y = max(0, rel_y)
                crop_w = min(aw, w - crop_x, aw - max(0, -rel_x))
                crop_h = min(ah, h - crop_y, ah - max(0, -rel_y))

                if crop_w <= 0 or crop_h <= 0:
                    continue

                if filter_clipped:
                    if (crop_x <= 0 or crop_y <= 0 or 
                        crop_x + crop_w >= w or crop_y + crop_h >= h):
                        continue

                for z in range(n_z_local):
                    plane = image[:, :, :, z] if image.ndim == 4 else image
                    crop = plane[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
                    if crop.size == 0:
                        continue
                    scores = score_region(crop)
                    local_roi_recs.append({
                        "tile_path": tile_path_id,
                        "ann_idx":   i,
                        "label":     ann["label"],
                        "z_index":   z,
                        "vol":       scores["vol"],
                        "tenengrad": scores["tenengrad"],
                        "bbox":      [crop_x, crop_y, crop_w, crop_h],
                    })

            return local_ds_recs, local_roi_recs, local_n_z

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
            for res_ds, res_roi, res_n_z in tqdm(executor.map(process_tile, tile_annotations), 
                                                 total=len(tile_annotations), 
                                                 desc=f"  Scoring tiles", 
                                                 unit="tile"):
                ds_records.extend(res_ds)
                roi_records.extend(res_roi)
                n_z_global = max(n_z_global, res_n_z)

    return roi_records, ds_records, n_z_global
#  Helper: build ROI identifiers                                        #
# ===================================================================== #

def _roi_key(rec):
    """Unique identifier for an annotated palynomorph."""
    return (rec["tile_path"], rec["ann_idx"])


# ===================================================================== #
#  Plotting                                                              #
# ===================================================================== #

def plot_best_z_histogram(roi_records, n_z, output_dir):
    """Plot 1 — histogram: how many palynomorphs peak at each Z index."""
    best_z = {"vol": [], "tenengrad": []}

    # Identify best Z per ROI for each metric
    from collections import defaultdict
    roi_scores = defaultdict(list)
    for rec in roi_records:
        roi_scores[_roi_key(rec)].append(rec)

    for key, recs in roi_scores.items():
        vol_best = max(recs, key=lambda r: r["vol"])["z_index"]
        ten_best = max(recs, key=lambda r: r["tenengrad"])["z_index"]
        best_z["vol"].append(vol_best)
        best_z["tenengrad"].append(ten_best)

    z_bins = np.arange(-0.5, n_z + 0.5, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # VoL histogram
    axes[0].hist(best_z["vol"], bins=z_bins, color=COLOR_VOL,
                 edgecolor="white", linewidth=0.8, alpha=0.85)
    axes[0].set_title("Variance of Laplacian\nBest Focal Plane per Palynomorph")
    axes[0].set_xlabel("Focal Plane Index (Z)")
    axes[0].set_ylabel("Number of Palynomorphs")
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Tenengrad histogram
    axes[1].hist(best_z["tenengrad"], bins=z_bins, color=COLOR_TEN,
                 edgecolor="white", linewidth=0.8, alpha=0.85)
    axes[1].set_title("Tenengrad Gradient\nBest Focal Plane per Palynomorph")
    axes[1].set_xlabel("Focal Plane Index (Z)")
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Distribution of Best Focal Planes Across Annotated Palynomorphs",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "01_best_focal_plane_histogram.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved to {path}")


def plot_mean_focus_per_z_roi(roi_records, n_z, output_dir):
    """Plot 2 — line plot: mean (± std) focus score per Z across all ROIs."""
    from collections import defaultdict

    z_vol  = defaultdict(list)
    z_ten  = defaultdict(list)
    for rec in roi_records:
        z_vol[rec["z_index"]].append(rec["vol"])
        z_ten[rec["z_index"]].append(rec["tenengrad"])

    zs = sorted(z_vol.keys())
    vol_mean = [np.mean(z_vol[z]) for z in zs]
    vol_std  = [np.std(z_vol[z])  for z in zs]
    ten_mean = [np.mean(z_ten[z]) for z in zs]
    ten_std  = [np.std(z_ten[z])  for z in zs]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.errorbar(zs, vol_mean, yerr=vol_std, fmt="o-", color=COLOR_VOL,
                 capsize=4, label="Variance of Laplacian", linewidth=1.8,
                 markersize=6)
    ax1.set_xlabel("Focal Plane Index (Z)")
    ax1.set_ylabel("VoL Score", color=COLOR_VOL)
    ax1.tick_params(axis="y", labelcolor=COLOR_VOL)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.errorbar(zs, ten_mean, yerr=ten_std, fmt="s--", color=COLOR_TEN,
                 capsize=4, label="Tenengrad", linewidth=1.8, markersize=6)
    ax2.set_ylabel("Tenengrad Score", color=COLOR_TEN)
    ax2.tick_params(axis="y", labelcolor=COLOR_TEN)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best",
               framealpha=0.9)

    ax1.set_title("Per-ROI Mean Focus Score Across Focal Planes\n"
                   "(mean ± 1 std over all annotated palynomorphs)")
    fig.tight_layout()
    path = os.path.join(output_dir, "02_per_roi_focus_vs_z.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved to {path}")


def plot_dataset_wide_focus(ds_records, n_z, output_dir):
    """Plot 3 — line plot: mean focus score per Z across all tiles."""
    from collections import defaultdict

    z_vol = defaultdict(list)
    z_ten = defaultdict(list)
    for rec in ds_records:
        z_vol[rec["z_index"]].append(rec["vol"])
        z_ten[rec["z_index"]].append(rec["tenengrad"])

    zs = sorted(z_vol.keys())
    vol_mean = [np.mean(z_vol[z]) for z in zs]
    ten_mean = [np.mean(z_ten[z]) for z in zs]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(zs, vol_mean, "o-", color=COLOR_VOL, label="Variance of Laplacian",
             linewidth=2, markersize=7)
    ax1.set_xlabel("Focal Plane Index (Z)")
    ax1.set_ylabel("VoL Score", color=COLOR_VOL)
    ax1.tick_params(axis="y", labelcolor=COLOR_VOL)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.plot(zs, ten_mean, "s--", color=COLOR_TEN, label="Tenengrad",
             linewidth=2, markersize=7)
    ax2.set_ylabel("Tenengrad Score", color=COLOR_TEN)
    ax2.tick_params(axis="y", labelcolor=COLOR_TEN)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best",
               framealpha=0.9)

    ax1.set_title("Dataset-Wide Mean Focus Score per Focal Plane\n"
                   "(averaged over all tiles)")
    fig.tight_layout()
    path = os.path.join(output_dir, "03_dataset_wide_focus_vs_z.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved to {path}")


def plot_metric_agreement_scatter(roi_records, output_dir):
    """Plot 4 — scatter: do VoL and Tenengrad agree on the best Z?"""
    from collections import defaultdict

    roi_scores = defaultdict(list)
    for rec in roi_records:
        roi_scores[_roi_key(rec)].append(rec)

    vol_best_z = []
    ten_best_z = []
    for key, recs in roi_scores.items():
        vol_best_z.append(max(recs, key=lambda r: r["vol"])["z_index"])
        ten_best_z.append(max(recs, key=lambda r: r["tenengrad"])["z_index"])

    fig, ax = plt.subplots(figsize=(6, 6))

    # Jitter for visibility when points overlap
    jitter = 0.15
    rng = np.random.default_rng(42)
    jx = rng.uniform(-jitter, jitter, len(vol_best_z))
    jy = rng.uniform(-jitter, jitter, len(ten_best_z))

    ax.scatter(np.array(vol_best_z) + jx,
               np.array(ten_best_z) + jy,
               alpha=0.6, s=50, edgecolors="white", linewidth=0.5,
               color="#6A4C93")

    # Perfect-agreement diagonal
    lims = [min(min(vol_best_z), min(ten_best_z)) - 0.5,
            max(max(vol_best_z), max(ten_best_z)) + 0.5]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="Perfect agreement")

    ax.set_xlabel("Best Focal Plane (Variance of Laplacian)")
    ax.set_ylabel("Best Focal Plane (Tenengrad)")
    ax.set_title("Agreement Between Focus Metrics\non Best Focal Plane per Palynomorph")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_aspect("equal", adjustable="box")

    # Compute agreement percentage
    agree = sum(1 for v, t in zip(vol_best_z, ten_best_z) if v == t)
    total = len(vol_best_z)
    ax.text(0.95, 0.05,
            f"Agreement: {agree}/{total} ({100 * agree / max(total, 1):.1f}%)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    path = os.path.join(output_dir, "04_metric_agreement_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved to {path}")


def plot_heatmap(roi_records, n_z, output_dir):
    """Plot 5 — heatmap: rows = palynomorphs, cols = Z, colour = score."""
    from collections import defaultdict, OrderedDict

    roi_scores = defaultdict(list)
    for rec in roi_records:
        roi_scores[_roi_key(rec)].append(rec)

    # Sort ROIs for deterministic display and build label list
    sorted_keys = sorted(roi_scores.keys())
    labels = []
    vol_matrix = []
    ten_matrix = []

    for key in sorted_keys:
        recs = sorted(roi_scores[key], key=lambda r: r["z_index"])
        label_str = recs[0]["label"]
        labels.append(f"{label_str} (tile …{os.path.basename(key[0])}, #{key[1]})")
        vol_row = [0.0] * n_z
        ten_row = [0.0] * n_z
        for rec in recs:
            vol_row[rec["z_index"]] = rec["vol"]
            ten_row[rec["z_index"]] = rec["tenengrad"]
        vol_matrix.append(vol_row)
        ten_matrix.append(ten_row)

    vol_matrix = np.array(vol_matrix)
    ten_matrix = np.array(ten_matrix)

    # Truncate label list if too long for readability
    max_rows_display = 50
    if len(labels) > max_rows_display:
        # Show first and last portions
        show_indices = list(range(max_rows_display // 2)) + \
                       list(range(len(labels) - max_rows_display // 2, len(labels)))
        labels = [labels[i] for i in show_indices]
        vol_matrix = vol_matrix[show_indices]
        ten_matrix = ten_matrix[show_indices]

    n_rows = len(labels)
    fig_height = max(5, 0.35 * n_rows + 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, fig_height))

    # VoL heatmap
    im0 = axes[0].imshow(vol_matrix, aspect="auto", cmap="viridis")
    axes[0].set_title("Variance of Laplacian\nper Palynomorph × Focal Plane")
    axes[0].set_xlabel("Focal Plane Index (Z)")
    axes[0].set_ylabel("Annotated Palynomorph")
    axes[0].set_yticks(range(n_rows))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_xticks(range(n_z))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="VoL Score")

    # Tenengrad heatmap
    im1 = axes[1].imshow(ten_matrix, aspect="auto", cmap="inferno")
    axes[1].set_title("Tenengrad Gradient\nper Palynomorph × Focal Plane")
    axes[1].set_xlabel("Focal Plane Index (Z)")
    axes[1].set_ylabel("Annotated Palynomorph")
    axes[1].set_yticks(range(n_rows))
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xticks(range(n_z))
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Tenengrad Score")

    fig.suptitle("Focus Score Heatmap — All Palynomorphs Across Focal Planes",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "05_focus_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved to {path}")


def plot_best_worst_examples(tiles, annotations, roi_records, output_dir,
                             n_examples=8, metric="vol", raw_dir=None, magnification=40.0, mode="spread"):
    """Plot 6 & 7 — side-by-side crops: sharpest Z vs worst Z for a subset of ROIs.

    For each selected palynomorph, two columns are shown:
      - Left:  crop at the focal plane with the highest focus score
      - Right: crop at the focal plane with the lowest focus score

    Parameters
    ----------
    tiles : list[dict]
        Loaded tiles (with 'path' and 'data' keys). If empty, reads from raw_dir.
    annotations : dict
        Tile-path -> annotation list mapping.
    roi_records : list[dict]
        Per-ROI focus score records.
    output_dir : str
        Where to save the PNG.
    n_examples : int
        How many palynomorphs to display
    metric : str
        Which metric to rank by ('vol' or 'tenengrad').
    raw_dir : str, optional
        Path to raw NDPI files if tiles are not loaded in memory.
    magnification : float, optional
        Magnification used for tile extraction.
    mode : str
        Selection mode: "spread" (top spread scores) or "random" (random subset).
    """
    import random
    from collections import defaultdict

    # Build a lookup: tile_path -> tile data (only used if tiles were pre-loaded)
    tile_lookup = {t["path"]: t["data"] for t in tiles}

    # Group records by ROI
    roi_scores = defaultdict(list)
    for rec in roi_records:
        roi_scores[_roi_key(rec)].append(rec)

    # For each ROI, find best and worst Z and the score spread
    roi_info = []
    for key, recs in roi_scores.items():
        best_rec = max(recs, key=lambda r: r[metric])
        worst_rec = min(recs, key=lambda r: r[metric])
        spread = best_rec[metric] - worst_rec[metric]
        roi_info.append({
            "key":       key,
            "best_rec":  best_rec,
            "worst_rec": worst_rec,
            "spread":    spread,
            "label":     best_rec["label"],
        })

    if mode == "random":
        random.seed(42)  # For reproducible random plots
        selected = random.sample(roi_info, min(n_examples, len(roi_info)))
    else:
        # Pick the N ROIs with the biggest spread (most visually distinct)
        roi_info.sort(key=lambda r: r["spread"], reverse=True)
        selected = roi_info[:n_examples]

    if not selected:
        return

    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D indexing

    metric_label = "VoL" if metric == "vol" else "Tenengrad"

    # Cache for NDPIData instances if reading from raw
    ndpi_cache = {}

    for i, info in enumerate(selected):
        tile_path = info["key"][0] # e.g. "slide_name/tile_x_y_w_h.npy"
        ann_idx   = info["key"][1]

        best_z  = info["best_rec"]["z_index"]
        worst_z = info["worst_rec"]["z_index"]

        if raw_dir:
            # Parse tile path: "slide_name/tile_x_y_w_h.npy"
            slide_name, tile_file_name = tile_path.split("/")
            parts = tile_file_name.replace(".npy", "").split("_")
            x0, y0 = int(parts[1]), int(parts[2])

            if slide_name not in ndpi_cache:
                ndpi_cache[slide_name] = NDPIData(os.path.join(raw_dir, slide_name + ".ndpi"))
            ndpi_data = ndpi_cache[slide_name]

            # Use bbox previously computed in the records
            # Since bbox is [crop_x, crop_y, crop_w, crop_h] relative to the tile
            bx, by, bw, bh = info["best_rec"]["bbox"]
            global_x = x0 + bx
            global_y = y0 + by
            
            # Get the single Z planes directly via lazy loading
            best_z_page = ndpi_data.get_page_index(magnification, ndpi_data.get_z_offsets()[best_z])
            worst_z_page = ndpi_data.get_page_index(magnification, ndpi_data.get_z_offsets()[worst_z])
            
            best_crop = ndpi_data.get_tile_from_page(best_z_page, global_x, global_y, bw, bh)
            worst_crop = ndpi_data.get_tile_from_page(worst_z_page, global_x, global_y, bw, bh)
            
            label_text = info['label']
            ylabel_str = f"{label_text}\n(#{ann_idx})"

        else:
            tile_data = tile_lookup[tile_path]  # (H, W, C, Z)
            bx, by, bw, bh = info["best_rec"]["bbox"]

            best_crop  = tile_data[by:by+bh, bx:bx+bw, :, best_z]
            worst_crop = tile_data[by:by+bh, bx:bx+bw, :, worst_z]
            ylabel_str = f"{info['label']}\n(#{ann_idx})"

        # Best Z
        axes[i, 0].imshow(best_crop)
        axes[i, 0].set_title(
            f"Best Z={best_z}  ({metric_label}={info['best_rec'][metric]:.0f})",
            fontsize=9)
        axes[i, 0].set_ylabel(ylabel_str, fontsize=8,
                               rotation=0, labelpad=45, va="center")

        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Worst Z
        axes[i, 1].imshow(worst_crop)
        axes[i, 1].set_title(
            f"Worst Z={worst_z}  ({metric_label}={info['worst_rec'][metric]:.0f})",
            fontsize=9)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    # Column headers
    axes[0, 0].text(0.5, 1.35, "Sharpest Focal Plane",
                     transform=axes[0, 0].transAxes, ha="center",
                     fontsize=11, fontweight="bold", color=COLOR_VOL)
    axes[0, 1].text(0.5, 1.35, "Worst Focal Plane",
                     transform=axes[0, 1].transAxes, ha="center",
                     fontsize=11, fontweight="bold", color="#B00020")

    fig.suptitle(f"Best vs Worst Focal Plane — {metric_label}\n"
                 f"({mode} selection, {n} examples)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    filename_prefix = "06" if mode == "spread" else "07"
    path = os.path.join(output_dir, f"{filename_prefix}_{mode}_examples_{metric}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved to {path}")


# ===================================================================== #
#  CSV export                                                            #
# ===================================================================== #

def export_csv(roi_records, ds_records, output_dir):
    """Write per-ROI and dataset-wide scores to CSV."""
    # Per-ROI CSV
    roi_csv = os.path.join(output_dir, "focus_scores_per_roi.csv")
    with open(roi_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tile_path", "ann_idx", "label",
                           "z_index", "vol", "tenengrad", "bbox"])
        writer.writeheader()
        writer.writerows(roi_records)
    print(f"Saved to {roi_csv}")

    # Dataset-wide CSV
    ds_csv = os.path.join(output_dir, "focus_scores_dataset_wide.csv")
    with open(ds_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tile_path", "z_index", "vol", "tenengrad"])
        writer.writeheader()
        writer.writerows(ds_records)
    print(f"Saved to {ds_csv}")


# ===================================================================== #
#  CLI entry point                                                       #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Focal-plane focus exploration for palynomorph tiles.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", 
        help="Path to a preprocessing output directory (must contain tiles/ and annotations.json)")
    group.add_argument(
        "--raw-dir",
        help="Path to directory containing raw .ndpi and .ndpa files. Skips disk storage.")
    
    parser.add_argument(
        "--output", default="output/exploration",
        help="Directory to write plots and CSV files (default: output/exploration)")
    parser.add_argument(
        "--filter-clipped", action="store_true",
        help="Ignore any annotation that touches a tile boundary (0 or tile size)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Load & Score ──────────────────────────────────────────────────
    if args.raw_dir:
        print(f"Running in-memory pipeline from raw NDPI files: {args.raw_dir}")
        if args.filter_clipped:
            print("  Filtering clipped annotations.")
        roi_records, ds_records, n_z = process_raw_dataset(
            args.raw_dir, filter_clipped=args.filter_clipped)
        print(f"  Generated {len(roi_records)} per-ROI records, {len(ds_records)} tile records.")
        print(f"  Found {n_z} maximum focal planes across the dataset.")
    else:
        print(f"Loading tiles and annotations from {args.input}")
        tiles, annotations = load_tiles_and_annotations(args.input)
        print(f"  Loaded {len(tiles)} tiles, {sum(len(v) for v in annotations.values())} annotations.")

        print("Computing per-ROI focus scores...")
        if args.filter_clipped:
            print("  Filtering clipped annotations.")
        roi_records, n_z = compute_per_roi_scores(tiles, annotations,
                                                  filter_clipped=args.filter_clipped)
        print(f"  Generated {len(roi_records)} records, {n_z} focal planes.")

        print("Computing dataset-wide focus scores...")
        ds_records, _ = compute_dataset_wide_scores(tiles)
        print(f"  Generated {len(ds_records)} records.")

    # ── Plot & export ─────────────────────────────────────────────────
    print("Generating plots and CSV files...")
    if roi_records:
        plot_best_z_histogram(roi_records, n_z, args.output)
        plot_mean_focus_per_z_roi(roi_records, n_z, args.output)
        plot_metric_agreement_scatter(roi_records, args.output)
        plot_heatmap(roi_records, n_z, args.output)
        
        # Plot top spread examples
        plot_best_worst_examples(tiles if args.input else [],
                                 annotations if args.input else {}, 
                                 roi_records, args.output, n_examples=8, metric="vol", raw_dir=args.raw_dir, mode="spread")
        plot_best_worst_examples(tiles if args.input else [],
                                 annotations if args.input else {},
                                 roi_records, args.output, n_examples=8, metric="tenengrad", raw_dir=args.raw_dir, mode="spread")
        
        # Plot random examples
        plot_best_worst_examples(tiles if args.input else [],
                                 annotations if args.input else {}, 
                                 roi_records, args.output, n_examples=8, metric="vol", raw_dir=args.raw_dir, mode="random")
        plot_best_worst_examples(tiles if args.input else [],
                                 annotations if args.input else {},
                                 roi_records, args.output, n_examples=8, metric="tenengrad", raw_dir=args.raw_dir, mode="random")
    else:
        print("No annotated ROIs found, skipping per-ROI plots.")

    if ds_records:
        plot_dataset_wide_focus(ds_records, n_z, args.output)
    else:
        print("No tiles found, skipping dataset-wide plot.")

    export_csv(roi_records, ds_records, args.output)
    print("Output available in output/exploration")

if __name__ == "__main__":
    main()
