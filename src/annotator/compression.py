"""
Tile compression helpers used by the annotator pipeline.
Can be used to collapse W x H x C x Z tiles into W x H x C 2D 
images for detection, using focus stacking or focal plane 
selection.
"""

from functools import partial
from typing import Callable

import numpy as np

from src.preprocessing.postprocess_tiles import focus_stack, tenengrad_ranking


def _normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Clamp an image to uint8 if it is not already stored in that format."""
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)

def focus_stack(tile_3d: np.ndarray, focus_stack_kernel_size: int) -> np.ndarray:
    """Collapse a 3D tile into a 2D RGB tile using focus stacking."""
    stacked = focus_stack(tile_3d, k=focus_stack_kernel_size)
    image = stacked[0] if isinstance(stacked, tuple) else stacked
    return _normalize_uint8(image)

def best_focal_plane(tile_3d: np.ndarray, tenengrad_ksize: int) -> np.ndarray:
    """Select the top Tenengrad-ranked focal plane from a 3D tile."""
    ranking = tenengrad_ranking(tile_3d, tenengrad_ksize=tenengrad_ksize)
    plane = tile_3d[:, :, :, int(ranking[0])]
    return _normalize_uint8(plane)

def get_compression_func(
    method: str,
    focus_stack_kernel_size: int,
    tenengrad_ksize: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the 3D-to-2D compression function for the selected method."""
    method_key = method.strip().lower()

    if method_key == "focus_stack":
        return partial(focus_stack, focus_stack_kernel_size=focus_stack_kernel_size)

    if method_key == "best_focal_plane":
        return partial(best_focal_plane, tenengrad_ksize=tenengrad_ksize)

    raise ValueError("Unsupported compression method. Choose 'focus_stack' or 'best_focal_plane'.")
