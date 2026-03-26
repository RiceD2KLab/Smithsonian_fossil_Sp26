"""
h5_utils.py — Shared utilities for discovering and indexing H5 tile files.

Dependencies:
    h5py

Typical usage:
    from h5_utils import list_h5_paths, list_tile_jobs

    paths = list_h5_paths("data/tiles")
    tiles = list_tile_jobs("data/tiles")  # [(h5_path, image_stem, group_name), ...]
"""

import os

import h5py


def list_h5_paths(h5_root: str) -> list[str]:
    """
    Return a sorted list of H5 file paths found at ``h5_root``.

    Accepts either a single ``.h5`` file or a directory containing ``.h5``
    files. Returns an empty list if ``h5_root`` matches neither.

    Args:
        h5_root: Path to a ``.h5`` file or a directory of ``.h5`` files.

    Returns:
        Sorted list of absolute paths to ``.h5`` files.
    """
    if os.path.isfile(h5_root) and h5_root.endswith(".h5"):
        return [h5_root]
    if os.path.isdir(h5_root):
        return sorted(
            os.path.join(h5_root, f)
            for f in os.listdir(h5_root)
            if f.endswith(".h5")
        )
    return []


def list_tile_jobs(h5_root: str) -> list[tuple[str, str, str]]:
    """
    Build an index of all tiles found across every H5 file in ``h5_root``.

    Scans each H5 file for groups whose key starts with ``"tile_"`` and
    records their location as a three-element tuple. The result is suitable
    for both sequential iteration (dataset loading) and parallel dispatch
    (focus scoring workers).

    Args:
        h5_root: Path to a ``.h5`` file or a directory of ``.h5`` files.

    Returns:
        List of ``(h5_path, image_stem, group_name)`` tuples, where:
            h5_path    — absolute path to the containing H5 file.
            image_stem — slide filename without extension, e.g. ``"slide01"``.
            group_name — H5 group key for the tile, e.g. ``"tile_3_7"``.
    """
    tiles: list[tuple[str, str, str]] = []
    for h5_path in list_h5_paths(h5_root):
        image_stem = os.path.splitext(os.path.basename(h5_path))[0]
        with h5py.File(h5_path, "r") as h5f:
            for key in sorted(h5f.keys()):
                if key.startswith("tile_"):
                    tiles.append((h5_path, image_stem, key))
    return tiles
