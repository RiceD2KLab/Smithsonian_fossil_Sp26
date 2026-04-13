"""
Focal-plane sharpness exploration for palynomorph slide tiles.

Reads the H5 output of `generate_tiles.py`: one or more `.h5` files, each
containing tile groups with `data` (H, W, C, Z), `bboxes`, and `labels`.
For each annotated palynomorph crop and for each full tile, we compute two focus metrics
across all Z planes: Variance of Laplacian (VoL) and Tenengrad.
Results are written to CSV files and we plot some of the results.

Usage:
    python -m src.exploration.explore_focus --input output/tiles --output output/exploration
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Iterator

import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.preprocessing.focus_metrics import tenengrad, variance_of_laplacian, percentile_vol, percentile_tenengrad
from src.preprocessing.h5_utils import list_h5_paths, list_tile_jobs

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOR_VOL     = "#0077BB"
COLOR_TEN     = "#EE3377"
COLOR_PVOL    = "#009944"
COLOR_PTEN    = "#FF8C00"

# H5 data loading

def iter_tiles_from_h5(input_path: str) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns an iterator over (tile_path, data, bboxes, labels) for every tile in the H5 output.

    tile_path is "{image_stem}/tile_{x}_{y}". data shape is (H, W, C, Z).
    bboxes is (N, 4) float, labels is (N,) int.
    """
    for h5_path in list_h5_paths(input_path):
        image_stem = os.path.splitext(os.path.basename(h5_path))[0]
        with h5py.File(h5_path, "r") as h5f:
            for key in sorted(h5f.keys()):
                if not key.startswith("tile_"):
                    continue
                group = h5f[key]
                data = np.array(group["data"])
                bboxes = np.array(group["bboxes"])
                labels = np.array(group["labels"])
                if bboxes.size == 0:
                    bboxes = bboxes.reshape(0, 4)
                if labels.size == 0:
                    labels = np.array([], dtype=np.int32)
                tile_path = f"{image_stem}/{key}"
                yield tile_path, data, bboxes, labels


def load_single_tile_data(input_path: str, tile_path: str) -> np.ndarray | None:
    """
    Load the tile image array for a given tile_path from the H5 store.

    tile_path format: "{image_stem}/tile_{x}_{y}". Returns (H, W, C, Z) or None if not found.
    """
    parts = tile_path.split("/", 1)
    if len(parts) != 2:
        return None
    image_stem, group_name = parts
    if input_path.endswith(".h5"):
        h5_path = input_path
    else:
        h5_path = os.path.join(input_path, f"{image_stem}.h5")
    if not os.path.isfile(h5_path):
        return None
    with h5py.File(h5_path, "r") as h5f:
        if group_name not in h5f:
            return None
        return np.array(h5f[group_name]["data"])


def load_single_tile_data_with_stack(input_path: str, tile_path: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Load tile data and focus_stacked image for a given tile_path.
    Returns (data (H, W, C, Z), focus_stacked (H, W, C)) or (None, None).
    """
    parts = tile_path.split("/", 1)
    if len(parts) != 2:
        return None, None
    image_stem, group_name = parts
    h5_path = input_path if input_path.endswith(".h5") else os.path.join(input_path, f"{image_stem}.h5")
    if not os.path.isfile(h5_path):
        return None, None
    with h5py.File(h5_path, "r") as h5f:
        if group_name not in h5f:
            return None, None
        group = h5f[group_name]
        data = np.array(group["data"])
        focus_stacked = np.array(group["focus_stacked"])
        return data, focus_stacked

def plot_tile_metric_comparison(
    ds_records: list[dict[str, Any]],
    input_path: str,
    output_dir: str,
    n_examples: int = 4,
    mode: str = "random",
) -> None:
    """
    For a selection of tiles, show the best Z chosen by each metric side by side,
    plus the pre-computed focus-stacked image.
    """
    import random

    by_tile: dict[str, list[dict]] = defaultdict(list)
    for rec in ds_records:
        by_tile[rec["tile_path"]].append(rec)

    tile_paths = list(by_tile.keys())
    if mode == "random":
        random.seed(449)
        selected_paths = random.sample(tile_paths, min(n_examples, len(tile_paths)))
    else:
        selected_paths = tile_paths[:n_examples]

    metrics = [
        ("vol",        "VoL",        COLOR_VOL),
        ("tenengrad",  "Tenengrad",  COLOR_TEN),
        ("pvol",       "pVoL",       COLOR_PVOL),
        ("ptenengrad", "pTenengrad", COLOR_PTEN),
    ]

    n = len(selected_paths)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = np.atleast_2d(axes)

    for i, tile_path in enumerate(selected_paths):
        recs = by_tile[tile_path]
        tile_data, focus_stacked = load_single_tile_data_with_stack(input_path, tile_path)
        if tile_data is None:
            continue

        for j, (metric, label, color) in enumerate(metrics):
            best_z = max(recs, key=lambda r: r[metric])["z_index"]
            score  = max(recs, key=lambda r: r[metric])[metric]
            axes[i, j].imshow(tile_data[:, :, :, best_z])
            axes[i, j].set_title(f"{label}\nZ={best_z}  ({score:.0f})", fontsize=9, color=color)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

        axes[i, 4].imshow(focus_stacked)
        axes[i, 4].set_title("Focus Stack", fontsize=9, color="black")
        axes[i, 4].set_xticks([])
        axes[i, 4].set_yticks([])
        axes[i, 0].set_ylabel(os.path.basename(tile_path), fontsize=7, rotation=0, labelpad=60, va="center")

    fig.suptitle("Best Focal Plane per Metric vs Focus Stack — Full Tile", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "08_tile_metric_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# Focus scoring

def _score_grayscale(gray: np.ndarray) -> dict[str, float]:
    """Compute VoL, Tenengrad, percentile VoL, and percentile Tenengrad for a single grayscale image."""
    return {
        "vol": variance_of_laplacian(gray),
        "tenengrad": tenengrad(gray, 3),
        "pvol": percentile_vol(gray, 75.0),
        "ptenengrad": percentile_tenengrad(gray, 3, 75.0),
    }


def _score_rgb_plane(rgb: np.ndarray) -> dict[str, float]:
    """Convert RGB to grayscale and compute focus scores."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return _score_grayscale(gray)


def _process_tile_chunk(
    chunk: list[tuple[str, str, str]],
    filter_clipped: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """
    Process a list of tile jobs (h5_path, image_stem, group_name). Returns
    (roi_records, ds_records, n_z). Used as ProcessPoolExecutor worker.
    """
    roi_records: list[dict[str, Any]] = []
    ds_records: list[dict[str, Any]] = []
    n_z_max = 0

    # Group by h5_path to open each file once
    by_file: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for h5_path, image_stem, group_name in chunk:
        by_file[h5_path].append((image_stem, group_name))

    for h5_path, group_list in by_file.items():
        with h5py.File(h5_path, "r") as h5f:
            for image_stem, group_name in group_list:
                group = h5f[group_name]
                data = np.array(group["data"])
                bboxes = np.array(group["bboxes"])
                labels = np.array(group["labels"])
                if bboxes.size == 0:
                    bboxes = bboxes.reshape(0, 4)
                if labels.size == 0:
                    labels = np.array([], dtype=np.int32)
                tile_path = f"{image_stem}/{group_name}"

                height, width = data.shape[0], data.shape[1]
                n_z = data.shape[3] if data.ndim == 4 else 1
                n_z_max = max(n_z_max, n_z)

                gray_planes = [
                    cv2.cvtColor(
                        data[:, :, :, z] if data.ndim == 4 else data,
                        cv2.COLOR_RGB2GRAY,
                    )
                    for z in range(n_z)
                ]

                for z, gray in enumerate(gray_planes):
                    scores = _score_grayscale(gray)
                    ds_records.append({
                        "tile_path": tile_path,
                        "z_index": z,
                        "vol": scores["vol"],
                        "tenengrad": scores["tenengrad"],
                        "pvol": scores["pvol"],
                        "ptenengrad": scores["ptenengrad"],
                    })

                for ann_idx in range(len(labels)):
                    x, y, w, h = bboxes[ann_idx].tolist()
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    label = int(labels[ann_idx])
                    if filter_clipped:
                        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
                            continue
                    for z, gray in enumerate(gray_planes):
                        crop = gray[y : y + h, x : x + w]
                        if crop.size == 0:
                            continue
                        scores = _score_grayscale(crop)
                        roi_records.append({
                            "tile_path": tile_path,
                            "ann_idx": ann_idx,
                            "label": label,
                            "z_index": z,
                            "vol": scores["vol"],
                            "tenengrad": scores["tenengrad"],
                            "pvol": scores["pvol"],
                            "ptenengrad": scores["ptenengrad"],
                            "bbox": [x, y, w, h],
                        })

    return roi_records, ds_records, n_z_max


def compute_focus_scores_parallel(
    input_path: str,
    filter_clipped: bool = False,
    n_workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """
    Compute focus scores by processing tiles in parallel across multiple processes.

    Tiles are listed from H5, split into chunks, and each chunk is processed in
    a worker process. Returns (roi_records, ds_records, n_z).
    """
    jobs = list_tile_jobs(input_path)
    if not jobs:
        return [], [], 0

    n_workers = min(n_workers, len(jobs), os.cpu_count() or 1)
    n_workers = max(1, n_workers)
    chunk_size = (len(jobs) + n_workers - 1) // n_workers
    chunks = [
        jobs[i : i + chunk_size]
        for i in range(0, len(jobs), chunk_size)
    ]

    roi_records = []
    ds_records = []
    n_z_max = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_tile_chunk, chunk, filter_clipped): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            roi_part, ds_part, n_z = future.result()
            roi_records.extend(roi_part)
            ds_records.extend(ds_part)
            n_z_max = max(n_z_max, n_z)

    return roi_records, ds_records, n_z_max


def compute_focus_scores(
    tile_stream: Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    filter_clipped: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """
    Compute focus scores in a single pass over the tile stream (sequential).

    Returns (roi_records, ds_records, n_z). ROI records have tile_path, ann_idx,
    label, z_index, vol, tenengrad, pvol, ptenengrad, bbox. Dataset-wide records have tile_path,
    z_index, vol, tenengrad, pvol, ptenengrad.
    """
    roi_records: list[dict[str, Any]] = []
    ds_records: list[dict[str, Any]] = []
    n_z_max = 0

    for tile_path, data, bboxes, labels in tile_stream:
        height, width = data.shape[0], data.shape[1]
        n_z = data.shape[3] if data.ndim == 4 else 1
        n_z_max = max(n_z_max, n_z)

        gray_planes = [
            cv2.cvtColor(
                data[:, :, :, z] if data.ndim == 4 else data,
                cv2.COLOR_RGB2GRAY,
            )
            for z in range(n_z)
        ]

        for z, gray in enumerate(gray_planes):
            scores = _score_grayscale(gray)
            ds_records.append({
                "tile_path": tile_path,
                "z_index": z,
                "vol": scores["vol"],
                "tenengrad": scores["tenengrad"],
                "pvol": scores["pvol"],
                "ptenengrad": scores["ptenengrad"],
            })

        for ann_idx in range(len(labels)):
            x, y, w, h = bboxes[ann_idx].tolist()
            x, y, w, h = int(x), int(y), int(w), int(h)
            label = int(labels[ann_idx])

            if filter_clipped:
                if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
                    continue

            for z, gray in enumerate(gray_planes):
                crop = gray[y : y + h, x : x + w]
                if crop.size == 0:
                    continue
                scores = _score_grayscale(crop)
                roi_records.append({
                    "tile_path": tile_path,
                    "ann_idx": ann_idx,
                    "label": label,
                    "z_index": z,
                    "vol": scores["vol"],
                    "tenengrad": scores["tenengrad"],
                    "pvol": scores["pvol"],
                    "ptenengrad": scores["ptenengrad"],
                    "bbox": [x, y, w, h],
                })

    return roi_records, ds_records, n_z_max


def _roi_key(rec: dict[str, Any]) -> tuple[str, int]:
    """Unique key for an annotated ROI: (tile_path, ann_idx)."""
    return (rec["tile_path"], rec["ann_idx"])


# Plotting

def plot_best_z_histogram(
    roi_records: list[dict[str, Any]],
    n_z: int,
    output_dir: str,
) -> None:
    """Histogram: number of palynomorphs whose best focus is at each Z index."""
    roi_by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in roi_records:
        roi_by_key[_roi_key(rec)].append(rec)

    best_vol = [max(recs, key=lambda r: r["vol"])["z_index"] for recs in roi_by_key.values()]
    best_ten = [max(recs, key=lambda r: r["tenengrad"])["z_index"] for recs in roi_by_key.values()]
    best_pvol = [max(recs, key=lambda r: r["pvol"])["z_index"] for recs in roi_by_key.values()]
    best_pten = [max(recs, key=lambda r: r["ptenengrad"])["z_index"] for recs in roi_by_key.values()]

    z_bins = np.arange(-0.5, n_z + 0.5, 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

    axes[0, 0].hist(best_vol, bins=z_bins, color=COLOR_VOL, edgecolor="white", linewidth=0.8, alpha=0.85)
    axes[0, 0].set_title("Variance of Laplacian\nBest Focal Plane per Palynomorph")
    axes[0, 0].set_xlabel("Focal Plane Index (Z)")
    axes[0, 0].set_ylabel("Number of Palynomorphs")
    axes[0, 0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[0, 0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[0, 1].hist(best_ten, bins=z_bins, color=COLOR_TEN, edgecolor="white", linewidth=0.8, alpha=0.85)
    axes[0, 1].set_title("Tenengrad Gradient\nBest Focal Plane per Palynomorph")
    axes[0, 1].set_xlabel("Focal Plane Index (Z)")
    axes[0, 1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[1, 0].hist(best_pvol, bins=z_bins, color=COLOR_PVOL, edgecolor="white", linewidth=0.8, alpha=0.85)
    axes[1, 0].set_title("Percentile VoL (p=75)\nBest Focal Plane per Palynomorph")
    axes[1, 0].set_xlabel("Focal Plane Index (Z)")
    axes[1, 0].set_ylabel("Number of Palynomorphs")
    axes[1, 0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[1, 0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[1, 1].hist(best_pten, bins=z_bins, color=COLOR_PTEN, edgecolor="white", linewidth=0.8, alpha=0.85)
    axes[1, 1].set_title("Percentile Tenengrad (p=75)\nBest Focal Plane per Palynomorph")
    axes[1, 1].set_xlabel("Focal Plane Index (Z)")
    axes[1, 1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle(
        "Distribution of Best Focal Planes Across Annotated Palynomorphs",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "01_best_focal_plane_histogram.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_mean_focus_per_z_roi(
    roi_records: list[dict[str, Any]],
    n_z: int,
    output_dir: str,
) -> None:
    """Line plot: mean ± std focus score per Z across all ROIs."""
    z_vol: dict[int, list[float]] = defaultdict(list)
    z_ten: dict[int, list[float]] = defaultdict(list)
    z_pvol: dict[int, list[float]] = defaultdict(list)
    z_pten: dict[int, list[float]] = defaultdict(list)
    for rec in roi_records:
        z_vol[rec["z_index"]].append(rec["vol"])
        z_ten[rec["z_index"]].append(rec["tenengrad"])
        z_pvol[rec["z_index"]].append(rec["pvol"])
        z_pten[rec["z_index"]].append(rec["ptenengrad"])

    zs = sorted(z_vol.keys())
    vol_mean  = [np.mean(z_vol[z])  for z in zs]
    vol_std   = [np.std(z_vol[z])   for z in zs]
    ten_mean  = [np.mean(z_ten[z])  for z in zs]
    ten_std   = [np.std(z_ten[z])   for z in zs]
    pvol_mean = [np.mean(z_pvol[z]) for z in zs]
    pvol_std  = [np.std(z_pvol[z])  for z in zs]
    pten_mean = [np.mean(z_pten[z]) for z in zs]
    pten_std  = [np.std(z_pten[z])  for z in zs]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.errorbar(zs, vol_mean,  yerr=vol_std,  fmt="o-", color=COLOR_VOL,  capsize=4, label="Variance of Laplacian",      linewidth=1.8, markersize=6)
    ax1.errorbar(zs, pvol_mean, yerr=pvol_std, fmt="^-", color=COLOR_PVOL, capsize=4, label="Percentile VoL (p=75)",      linewidth=1.8, markersize=6)
    ax1.set_xlabel("Focal Plane Index (Z)")
    ax1.set_ylabel("VoL Score")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.errorbar(zs, ten_mean,  yerr=ten_std,  fmt="s-", color=COLOR_TEN,  capsize=4, label="Tenengrad",                  linewidth=1.8, markersize=6)
    ax2.errorbar(zs, pten_mean, yerr=pten_std, fmt="D-", color=COLOR_PTEN, capsize=4, label="Percentile Tenengrad (p=75)", linewidth=1.8, markersize=6)
    ax2.set_ylabel("Tenengrad Score")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.9)
    ax1.set_title(
        "Per-ROI Mean Focus Score Across Focal Planes\n"
        "(mean ± 1 std over all annotated palynomorphs)",
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "02_per_roi_focus_vs_z.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_dataset_wide_focus(
    ds_records: list[dict[str, Any]],
    n_z: int,
    output_dir: str,
) -> None:
    """Line plot: mean focus score per Z across all tiles."""
    z_vol: dict[int, list[float]] = defaultdict(list)
    z_ten: dict[int, list[float]] = defaultdict(list)
    z_pvol: dict[int, list[float]] = defaultdict(list)
    z_pten: dict[int, list[float]] = defaultdict(list)
    for rec in ds_records:
        z_vol[rec["z_index"]].append(rec["vol"])
        z_ten[rec["z_index"]].append(rec["tenengrad"])
        z_pvol[rec["z_index"]].append(rec["pvol"])
        z_pten[rec["z_index"]].append(rec["ptenengrad"])

    zs = sorted(z_vol.keys())
    vol_mean  = [np.mean(z_vol[z])  for z in zs]
    ten_mean  = [np.mean(z_ten[z])  for z in zs]
    pvol_mean = [np.mean(z_pvol[z]) for z in zs]
    pten_mean = [np.mean(z_pten[z]) for z in zs]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(zs, vol_mean,  "o-", color=COLOR_VOL,  label="Variance of Laplacian",      linewidth=2, markersize=7)
    ax1.plot(zs, pvol_mean, "^-", color=COLOR_PVOL, label="Percentile VoL (p=75)",      linewidth=2, markersize=7)
    ax1.set_xlabel("Focal Plane Index (Z)")
    ax1.set_ylabel("VoL Score")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.plot(zs, ten_mean,  "s-", color=COLOR_TEN,  label="Tenengrad",                  linewidth=2, markersize=7)
    ax2.plot(zs, pten_mean, "D-", color=COLOR_PTEN, label="Percentile Tenengrad (p=75)", linewidth=2, markersize=7)
    ax2.set_ylabel("Tenengrad Score")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.9)
    ax1.set_title("Dataset-Wide Mean Focus Score per Focal Plane\n(averaged over all tiles)")
    fig.tight_layout()
    path = os.path.join(output_dir, "03_dataset_wide_focus_vs_z.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_metric_agreement_scatter(
    roi_records: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Scatter: VoL best Z vs Tenengrad best Z per palynomorph."""
    roi_by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in roi_records:
        roi_by_key[_roi_key(rec)].append(rec)

    vol_best_z = [max(recs, key=lambda r: r["vol"])["z_index"] for recs in roi_by_key.values()]
    ten_best_z = [max(recs, key=lambda r: r["tenengrad"])["z_index"] for recs in roi_by_key.values()]

    fig, ax = plt.subplots(figsize=(6, 6))
    jitter = 0.15
    rng = np.random.default_rng(42)
    jx = rng.uniform(-jitter, jitter, len(vol_best_z))
    jy = rng.uniform(-jitter, jitter, len(ten_best_z))
    ax.scatter(
        np.array(vol_best_z) + jx,
        np.array(ten_best_z) + jy,
        alpha=0.6, s=50, edgecolors="white", linewidth=0.5, color="#6A4C93",
    )
    lims = [
        min(min(vol_best_z), min(ten_best_z)) - 0.5,
        max(max(vol_best_z), max(ten_best_z)) + 0.5,
    ]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="Perfect agreement")
    ax.set_xlabel("Best Focal Plane (Variance of Laplacian)")
    ax.set_ylabel("Best Focal Plane (Tenengrad)")
    ax.set_title("Agreement Between Focus Metrics on Best Focal Plane per Palynomorph")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_aspect("equal", adjustable="box")

    agree = sum(1 for v, t in zip(vol_best_z, ten_best_z) if v == t)
    total = len(vol_best_z)
    ax.text(
        0.95, 0.05,
        f"Agreement: {agree}/{total} ({100 * agree / max(total, 1):.1f}%)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "04_metric_agreement_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_heatmap(
    roi_records: list[dict[str, Any]],
    n_z: int,
    output_dir: str,
) -> None:
    """Heatmap: rows = palynomorphs, columns = Z, color = score."""
    roi_by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in roi_records:
        roi_by_key[_roi_key(rec)].append(rec)

    sorted_keys = sorted(roi_by_key.keys())
    row_labels = []
    vol_matrix = []
    ten_matrix = []
    pvol_matrix = []
    pten_matrix = []

    for key in sorted_keys:
        recs = sorted(roi_by_key[key], key=lambda r: r["z_index"])
        row_labels.append(f"label {recs[0]['label']} ({os.path.basename(key[0])}, #{key[1]})")
        vol_row = [0.0] * n_z
        ten_row = [0.0] * n_z
        pvol_row = [0.0] * n_z
        pten_row = [0.0] * n_z
        for rec in recs:
            vol_row[rec["z_index"]] = rec["vol"]
            ten_row[rec["z_index"]] = rec["tenengrad"]
            pvol_row[rec["z_index"]] = rec["pvol"]
            pten_row[rec["z_index"]] = rec["ptenengrad"]
        vol_matrix.append(vol_row)
        ten_matrix.append(ten_row)
        pvol_matrix.append(pvol_row)
        pten_matrix.append(pten_row)

    vol_matrix = np.array(vol_matrix)
    ten_matrix = np.array(ten_matrix)
    pvol_matrix = np.array(pvol_matrix)
    pten_matrix = np.array(pten_matrix)

    max_rows = 50
    if len(row_labels) > max_rows:
        show_indices = list(range(max_rows // 2)) + list(range(len(row_labels) - max_rows // 2, len(row_labels)))
        row_labels = [row_labels[i] for i in show_indices]
        vol_matrix = vol_matrix[show_indices]
        ten_matrix = ten_matrix[show_indices]
        pvol_matrix = pvol_matrix[show_indices]
        pten_matrix = pten_matrix[show_indices]

    n_rows = len(row_labels)
    fig_height = max(5, 0.35 * n_rows + 2)
    fig, axes = plt.subplots(2, 2, figsize=(18, fig_height))

    for ax, matrix, title, label in [
        (axes[0, 0], vol_matrix, "Variance of Laplacian\nper Palynomorph × Focal Plane", "VoL Score"),
        (axes[0, 1], ten_matrix, "Tenengrad Gradient\nper Palynomorph × Focal Plane", "Tenengrad Score"),
        (axes[1, 0], pvol_matrix, "Percentile VoL (p=75)\nper Palynomorph × Focal Plane", "pVoL Score"),
        (axes[1, 1], pten_matrix, "Percentile Tenengrad (p=75)\nper Palynomorph × Focal Plane", "pTenengrad Score"),
    ]:
        cmap = "viridis" if "VoL" in label else "inferno"
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Focal Plane Index (Z)")
        ax.set_ylabel("Annotated Palynomorph")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_xticks(range(n_z))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    fig.suptitle(
        "Focus Score Heatmap — All Palynomorphs Across Focal Planes",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "05_focus_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_best_worst_examples(
    roi_records: list[dict[str, Any]],
    input_path: str,
    output_dir: str,
    n_examples: int = 8,
    metric: str = "vol",
    mode: str = "spread",
) -> None:
    """
    Side-by-side crops: sharpest Z vs worst Z for a subset of ROIs.
    Tile data is loaded from H5 on demand.
    """
    import random

    roi_by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in roi_records:
        roi_by_key[_roi_key(rec)].append(rec)

    roi_infos = []
    for key, recs in roi_by_key.items():
        best_rec = max(recs, key=lambda r: r[metric])
        worst_rec = min(recs, key=lambda r: r[metric])
        roi_infos.append({
            "key": key,
            "best_rec": best_rec,
            "worst_rec": worst_rec,
            "spread": best_rec[metric] - worst_rec[metric],
            "label": best_rec["label"],
        })

    if mode == "random":
        random.seed(67)
        selected = random.sample(roi_infos, min(n_examples, len(roi_infos)))
    else:
        roi_infos.sort(key=lambda r: r["spread"], reverse=True)
        selected = roi_infos[:n_examples]

    if not selected:
        return

    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.5 * n))
    if n == 1:
        axes = np.atleast_2d(axes)

    metric_label = {"vol": "VoL", "tenengrad": "Tenengrad", "pvol": "pVoL", "ptenengrad": "pTenengrad"}.get(metric, metric)

    for i, info in enumerate(selected):
        tile_path, ann_idx = info["key"]
        best_z = info["best_rec"]["z_index"]
        worst_z = info["worst_rec"]["z_index"]
        bx, by, bw, bh = info["best_rec"]["bbox"]

        tile_data = load_single_tile_data(input_path, tile_path)
        if tile_data is None:
            continue

        best_crop = tile_data[by : by + bh, bx : bx + bw, :, best_z]
        worst_crop = tile_data[by : by + bh, bx : bx + bw, :, worst_z]
        ylabel_str = f"{info['label']}\n(#{ann_idx})"

        axes[i, 0].imshow(best_crop)
        axes[i, 0].set_title(
            f"Best Z={best_z}  ({metric_label}={info['best_rec'][metric]:.0f})",
            fontsize=9,
        )
        axes[i, 0].set_ylabel(ylabel_str, fontsize=8, rotation=0, labelpad=45, va="center")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(worst_crop)
        axes[i, 1].set_title(
            f"Worst Z={worst_z}  ({metric_label}={info['worst_rec'][metric]:.0f})",
            fontsize=9,
        )
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    axes[0, 0].text(
        0.5, 1.35, "Sharpest Focal Plane",
        transform=axes[0, 0].transAxes, ha="center", fontsize=11, fontweight="bold", color=COLOR_VOL,
    )
    axes[0, 1].text(
        0.5, 1.35, "Worst Focal Plane",
        transform=axes[0, 1].transAxes, ha="center", fontsize=11, fontweight="bold", color="#B00020",
    )
    fig.suptitle(
        f"Best vs Worst Focal Plane — {metric_label}\n({mode} selection, {n} examples)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    prefix = "06" if mode == "spread" else "07"
    path = os.path.join(output_dir, f"{prefix}_{mode}_examples_{metric}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def export_csv(
    roi_records: list[dict[str, Any]],
    ds_records: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Write per-ROI and dataset-wide focus scores to CSV."""
    roi_csv = os.path.join(output_dir, "focus_scores_per_roi.csv")
    with open(roi_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tile_path", "ann_idx", "label", "z_index", "vol", "tenengrad", "pvol", "ptenengrad", "bbox"],
        )
        writer.writeheader()
        writer.writerows(roi_records)
    print(f"Saved {roi_csv}")

    ds_csv = os.path.join(output_dir, "focus_scores_dataset_wide.csv")
    with open(ds_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile_path", "z_index", "vol", "tenengrad", "pvol", "ptenengrad"])
        writer.writeheader()
        writer.writerows(ds_records)
    print(f"Saved {ds_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focal-plane focus exploration for palynomorph tiles (H5 input from generate_tiles).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing .h5 files, or path to a single .h5 file",
    )
    parser.add_argument(
        "--output",
        default="output/exploration",
        help="Directory for plots and CSV (default: output/exploration)",
    )
    parser.add_argument(
        "--filter-clipped",
        action="store_true",
        help="Ignore annotations that touch the tile boundary",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        metavar="N",
        help="Number of worker processes for scoring (0 = sequential, default)",
    )
    args = parser.parse_args()

    h5_paths = list_h5_paths(args.input)
    if not h5_paths:
        print("No H5 files found at", args.input)
        return

    os.makedirs(args.output, exist_ok=True)

    if args.workers > 1:
        n_workers = min(args.workers, os.cpu_count() or 1)
        print(f"Computing focus scores in parallel ({n_workers} workers)...")
        roi_records, ds_records, n_z = compute_focus_scores_parallel(
            args.input, filter_clipped=args.filter_clipped, n_workers=n_workers
        )
    else:
        tile_stream = iter_tiles_from_h5(args.input)
        roi_records, ds_records, n_z = compute_focus_scores(
            tile_stream, filter_clipped=args.filter_clipped
        )
    print(f"Per-ROI records: {len(roi_records)}, dataset-wide: {len(ds_records)}, n_z: {n_z}")

    export_csv(roi_records, ds_records, args.output)

    if roi_records:
        plot_best_z_histogram(roi_records, n_z, args.output)
        plot_mean_focus_per_z_roi(roi_records, n_z, args.output)
        plot_metric_agreement_scatter(roi_records, args.output)
        plot_heatmap(roi_records, n_z, args.output)
        for metric in ("vol", "tenengrad", "pvol", "ptenengrad"):
            plot_best_worst_examples(
                roi_records, args.input, args.output,
                n_examples=8, metric=metric, mode="spread",
            )
            plot_best_worst_examples(
                roi_records, args.input, args.output,
                n_examples=8, metric=metric, mode="random",
            )
    else:
        print("No annotated ROIs; skipping per-ROI plots.")

    if ds_records:
        plot_dataset_wide_focus(ds_records, n_z, args.output)
        plot_tile_metric_comparison(ds_records, args.input, args.output, n_examples=10, mode="random")
    else:
        print("No tiles; skipping dataset-wide plot.")

    print("Done. Output in", args.output)


if __name__ == "__main__":
    main()
