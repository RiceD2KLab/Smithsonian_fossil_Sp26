"""
Tile compression helpers used by the annotator pipeline.

The helpers in this module collapse tile stacks from `W x H x C x Z` to
`W x H x C` for detector input. The available methods are focus stacking and
best-focal-plane selection.
"""

from functools import partial
from typing import Callable

import numpy as np

from src.preprocessing.postprocess_tiles import focus_stack as _focus_stack_impl
from src.preprocessing.postprocess_tiles import tenengrad_ranking


def _normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Return a `uint8` image, clipping values if needed.

    Args:
        image: Image array to normalize.

    Returns:
        The input image if it is already `uint8`, otherwise a clipped `uint8`
        copy in the range `[0, 255]`.
    """
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def focus_stack(tile_3d: np.ndarray, focus_stack_kernel_size: int) -> np.ndarray:
    """Collapse a tile stack into a 2D RGB tile using focus stacking.

    Args:
        tile_3d: Input tile stack with shape `H x W x C x Z`.
        focus_stack_kernel_size: Kernel size used by the focus-stacking
            implementation.

    Returns:
        A 2D RGB tile with shape `H x W x C` in `uint8` format.
    """
    stacked = _focus_stack_impl(tile_3d, k=focus_stack_kernel_size)
    image = stacked[0] if isinstance(stacked, tuple) else stacked
    return _normalize_uint8(image)


def best_focal_plane(tile_3d: np.ndarray, tenengrad_ksize: int) -> np.ndarray:
    """
    Select the best focal plane from a tile stack using Tenengrad gradient.

    Args:
        tile_3d: Input tile stack with shape `H x W x C x Z`.
        tenengrad_ksize: Sobel kernel size used when ranking focal planes.

    Returns:
        The selected focal plane as a `uint8` RGB image with shape `H x W x C`.
    """
    ranking = tenengrad_ranking(tile_3d, tenengrad_ksize=tenengrad_ksize)
    plane = tile_3d[:, :, :, int(ranking[0])]
    return _normalize_uint8(plane)

def get_compression_func(
    method: str,
    focus_stack_kernel_size: int,
    tenengrad_ksize: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the tile-compression function for the selected method.

    Args:
        method: Compression method name. Supported values are
            `focus_stack` and `best_focal_plane`.
        focus_stack_kernel_size: Kernel size to use when the selected method is
            focus stacking.
        tenengrad_ksize: Sobel kernel size to use when the selected method is
            best-focal-plane selection.

    Returns:
        A callable that accepts a `H x W x C x Z` tile stack and returns a
        `H x W x C` RGB image.

    Raises:
        ValueError: If `method` is not one of the supported compression modes.
    """
    method_key = method.strip().lower()

    if method_key == "focus_stack":
        return partial(focus_stack, focus_stack_kernel_size=focus_stack_kernel_size)

    if method_key == "best_focal_plane":
        return partial(best_focal_plane, tenengrad_ksize=tenengrad_ksize)

    raise ValueError("Unsupported compression method. Choose 'focus_stack' or 'best_focal_plane'.")
