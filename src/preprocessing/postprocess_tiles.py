import argparse
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from src.preprocessing.focus_metrics import tenengrad, variance_of_laplacian
from src.preprocessing.h5_utils import list_h5_paths

"""
Compute post-tile image products directly into H5 tile groups.

Supported outputs:
1. focus_stacked (LoG-based per-pixel focus stacking)
2. focal plane rankings (LoG, VoL, Tenengrad)
3. maximum_intensity_projection (MIP)
"""

TENENGRAD_KSIZE = 3
LOG_KERNEL_SIZE = 5
OUTPUT_CHOICES = ["focus_stacked", "rankings", "mip"]

def laplacian_of_gaussian(gray: np.ndarray, k: int = LOG_KERNEL_SIZE) -> np.ndarray:
    """Compute Laplacian of Gaussian for a 2D image."""
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=k)

def _to_gray_stack(tile_zfirst: np.ndarray) -> np.ndarray:
    """Convert (z, h, w, c) tile data into a grayscale stack (z, h, w)."""
    z, h, w, _ = tile_zfirst.shape
    gray_stack = np.zeros((z, h, w), dtype=np.uint8)

    for zi in range(z):
        img = tile_zfirst[zi].astype(np.uint8) if tile_zfirst[zi].dtype != np.uint8 else tile_zfirst[zi]
        gray_stack[zi] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return gray_stack

def focus_stack(tile: np.ndarray, k: int = LOG_KERNEL_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Focus stack a tile of shape (h, w, c, z) to (h, w, c).

    Returns:
        stacked: Focus-stacked RGB image.
        avg_logs: Per-z average absolute LoG response used for LoG ranking.
    """
    tile_zfirst = np.moveaxis(tile, -1, 0)  # (z, h, w, c)
    z, h, w, c = tile_zfirst.shape
    gray_stack = _to_gray_stack(tile_zfirst)

    log_stack = np.array([laplacian_of_gaussian(gray_stack[zi], k) for zi in range(z)])
    log_abs = np.abs(log_stack)
    avg_logs = log_abs.mean(axis=(1, 2))

    best_z = np.argmax(log_abs, axis=0)  # (h, w)
    row_idx = np.arange(h)[:, None]
    col_idx = np.arange(w)[None, :]

    stacked = np.zeros((h, w, c), dtype=tile_zfirst.dtype)
    for ch in range(c):
        channel_stack = tile_zfirst[..., ch]  # (z, h, w)
        stacked[..., ch] = channel_stack[best_z, row_idx, col_idx]

    return stacked, avg_logs

def maximum_intensity_projection(tile: np.ndarray) -> np.ndarray:
    """Collapse a tile of shape (h, w, c, z) to (h, w, c) via max projection."""
    return np.max(tile, axis=tile.ndim - 1)

def _compute_metric_rankings(
    tile: np.ndarray,
    tenengrad_ksize: int = TENENGRAD_KSIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-plane rankings using VoL and Tenengrad metrics."""
    tile_zfirst = np.moveaxis(tile, -1, 0)  # (z, h, w, c)
    z, _, _, _ = tile_zfirst.shape
    gray_stack = _to_gray_stack(tile_zfirst)

    vol_scores = np.array([variance_of_laplacian(gray_stack[zi]) for zi in range(z)], dtype=np.float64)
    tenengrad_scores = np.array(
        [tenengrad(gray_stack[zi], ksize=tenengrad_ksize) for zi in range(z)],
        dtype=np.float64,
    )

    ranking_vol = np.argsort(vol_scores)[::-1]
    ranking_tenengrad = np.argsort(tenengrad_scores)[::-1]
    return ranking_vol, ranking_tenengrad

def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Replace an H5 dataset in-place if it already exists."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, compression="gzip")

def process_h5_file(
    h5_path: str,
    write_focus_stacked: bool = True,
    write_rankings: bool = True,
    write_mip: bool = False,
    log_kernel_size: int = LOG_KERNEL_SIZE,
    tenengrad_ksize: int = TENENGRAD_KSIZE,
) -> None:
    """Generate selected outputs for every tile group in one H5 file."""
    if not (write_focus_stacked or write_rankings or write_mip):
        raise ValueError("At least one output must be selected.")

    with h5py.File(h5_path, "r+") as h5f:
        print(f"Processing {os.path.basename(h5_path)} with {len(h5f.keys())} tiles...")

        for tile_name in tqdm(h5f.keys()):
            group = h5f[tile_name]

            if not isinstance(group, h5py.Group) or "data" not in group:
                continue
            data = group["data"][:]

            stacked = None
            avg_logs = None
            if write_focus_stacked or write_rankings:
                stacked, avg_logs = focus_stack(data, k=log_kernel_size)

            if write_focus_stacked and stacked is not None:
                _replace_dataset(group, "focus_stacked", stacked)

            if write_rankings and avg_logs is not None:
                ranking_log = np.argsort(avg_logs)[::-1]
                ranking_vol, ranking_tenengrad = _compute_metric_rankings(
                    data,
                    tenengrad_ksize=tenengrad_ksize,
                )

                # Keep legacy field for compatibility and write explicit metric fields.
                _replace_dataset(group, "focal_plane_ranking", ranking_log)
                _replace_dataset(group, "focal_plane_ranking_LoG", ranking_log)
                _replace_dataset(group, "focal_plane_ranking_VoL", ranking_vol)
                _replace_dataset(group, "focal_plane_ranking_tenengrad", ranking_tenengrad)

            if write_mip:
                mip = maximum_intensity_projection(data)
                _replace_dataset(group, "maximum_intensity_projection", mip)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate focus stacking, focal rankings, and/or MIP outputs for tile H5 files."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to a single .h5 file or a directory containing .h5 files.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        choices=OUTPUT_CHOICES,
        default=["focus_stacked", "rankings"],
        help=(
            "Outputs to generate. Choose one or more of: "
            "focus_stacked rankings mip. Default: focus_stacked rankings"
        ),
    )
    parser.add_argument(
        "--log_kernel_size",
        type=int,
        default=LOG_KERNEL_SIZE,
        help=f"Gaussian/Laplacian kernel size for LoG stacking (default: {LOG_KERNEL_SIZE}).",
    )
    parser.add_argument(
        "--tenengrad_ksize",
        type=int,
        choices=[1, 3, 5, 7],
        default=TENENGRAD_KSIZE,
        help=f"Sobel kernel size for Tenengrad ranking (default: {TENENGRAD_KSIZE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_paths = list_h5_paths(args.input_path)
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found for input path: {args.input_path}")

    selected = set(args.outputs)
    for h5_path in h5_paths:
        process_h5_file(
            h5_path=h5_path,
            write_focus_stacked="focus_stacked" in selected,
            write_rankings="rankings" in selected,
            write_mip="mip" in selected,
            log_kernel_size=args.log_kernel_size,
            tenengrad_ksize=args.tenengrad_ksize,
        )

if __name__ == "__main__":
    main()