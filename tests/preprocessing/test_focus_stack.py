"""
Test script for focus stacking on preprocessed H5 tiles.

Reads a sample H5 file produced by generate_tiles.py, runs focus stacking on a
subset of tiles, prints the focal-plane ranking for each tile, and saves
side-by-side comparison images (focus-stacked vs. most-in-focus slice) as PNGs.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.postprocess_tiles import focus_stack


def test_focus_stack_on_sample(h5_path: str, output_dir: str, max_tiles: int = 20) -> None:
    """Run focus stacking on up to `max_tiles` tiles from an H5 file and save comparison images.

    For each tile, prints the focal-plane ranking (best to worst) and saves a
    side-by-side PNG showing the focus-stacked result next to the single most-
    in-focus slice.

    Args:
        h5_path: Path to the H5 file produced by generate_tiles.py.
        output_dir: Directory where output PNG images are written; created if absent.
        max_tiles: Maximum number of tiles to process from the H5 file.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(h5_path, 'r') as h5f:
        for i, tile_name in enumerate(list(h5f.keys())[:max_tiles]):
            group = h5f[tile_name]
            if not isinstance(group, h5py.Group) or 'data' not in group.keys():
                continue
            data = group['data'][:]
            if not hasattr(data, 'ndim') or data.ndim != 4:
                continue
            stacked, avg_logs = focus_stack(data)
            ranking = np.argsort(avg_logs)[::-1]
            print(f"Tile: {tile_name}, Focal plane ranking (best to worst): {ranking}")

            # Save the stacked image (convert to uint8 if needed)
            img = stacked
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            # Save as PNG using matplotlib (channels last)
            plt.figure(figsize=(12, 6))
            # Show both focus stacked and most in-focus slice
            plt.subplot(1, 2, 1)
            plt.title('Focus Stacked')
            if img.shape[-1] == 1:
                plt.imshow(img[..., 0], cmap='gray')
            else:
                plt.imshow(img)
            plt.axis('off')

            # Most in-focus slice (top-ranked z)
            best_z = ranking[0]
            best_slice = data[..., best_z]
            if best_slice.dtype != np.uint8:
                best_slice = np.clip(best_slice, 0, 255).astype(np.uint8)
            plt.subplot(1, 2, 2)
            plt.title(f'Most In-Focus Slice (z={best_z})')
            if best_slice.shape[-1] == 1:
                plt.imshow(best_slice[..., 0], cmap='gray')
            else:
                plt.imshow(best_slice)
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{tile_name}_focus_stacked_and_best.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

if __name__ == "__main__":
    TEST_H5_PATH = "/path/to/test_file.h5"  # Update with actual test file path
    TEST_OUTPUT_DIR = "/path/to/output/dir"  # Update with desired output directory
    test_focus_stack_on_sample(TEST_H5_PATH, TEST_OUTPUT_DIR)
