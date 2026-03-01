import math
import argparse
import sys
from typing import Tuple

from src.data.ndpa_reader import NDPAData
from src.data.ndpi_reader import NDPIData

"""
Calculates annotation density for an NDPI whole-slide image:

The tile count is computed from the bounding box of the ROI rectangle(s)
drawn in the NDPA file.
"""

TILE_SIZE_PX = 1024

# Coordinate conversion
def nm_to_px(
    nm_x: float,
    nm_y: float,
    ndpi: NDPIData,
) -> Tuple[float, float]:
    """
    Convert an (x, y) position in NDPA nanometre coordinates to level-0
    pixel coordinates of the NDPI slide.

    The NDPA coordinate system is centred on the slide centre, so we must
    subtract the Hamamatsu XOffset / YOffset before scaling.
    """
    meta = ndpi.metadata
    px_x = (nm_x - meta.x_offset_nm) * 0.001 / meta.mpp_x + meta.full_width  / 2
    px_y = (nm_y - meta.y_offset_nm) * 0.001 / meta.mpp_y + meta.full_height / 2
    return px_x, px_y


# Core density calculation
def compute_density(
    ndpa_path: str,
    ndpi_path: str,
    tile_size: int = TILE_SIZE_PX,
) -> dict:
    """
    Compute annotation density (annotations per tile) over the ROI bounding box.

    Parameters:
        ndpa_path : Path to the .ndpa annotation file.
        ndpi_path : Path to the companion .ndpi slide file.
        tile_size : Edge length of each square tile in pixels (default 1024).

    Returns:
        dict with result fields (see keys below).
    """
    ndpa = NDPAData(ndpa_path)
    ndpi = NDPIData(ndpi_path)

    if not ndpa.rois:
        raise ValueError("No ROI rectangles found in the NDPA file.")

    n_annotations = len(ndpa.palynomorphs)

    # Convert every ROI rectangle corner to pixel space and sum
    # up the total number of tiles across all ROIs

    n_tiles = 0

    for roi in ndpa.rois:
        nm_x, nm_y, nm_w, nm_h = roi.bounds.get_bounding_box()

        # Top-left and bottom-right corners
        x0_tmp, y0_tmp = nm_to_px(nm_x, nm_y, ndpi)
        x1_tmp, y1_tmp = nm_to_px(nm_x + nm_w, nm_y + nm_h, ndpi)

        # Normalise in case axes are flipped
        x0, x1 = min(x0_tmp, x1_tmp), max(x0_tmp, x1_tmp)
        y0, y1 = min(y0_tmp, y1_tmp), max(y0_tmp, y1_tmp)

        area_w = x1 - x0
        area_h = y1 - y0

        # Use ceiling so partial edge tiles are counted.
        tiles_x = math.ceil(area_w / tile_size)
        tiles_y = math.ceil(area_h / tile_size)
        n_tiles += (tiles_x * tiles_y)

    density = n_annotations / n_tiles if n_tiles > 0 else float("nan")

    return {
        "n_annotations": n_annotations,
        "n_rois":         len(ndpa.rois),
        "n_tiles":        n_tiles,
        "density":        density,
        "tile_size":      tile_size,
    }

# CLI
def main():
    parser = argparse.ArgumentParser(
        description="Compute palynomorph annotation density per tile."
    )
    parser.add_argument("--ndpi", required=True, help="Path to the .ndpi slide file.")
    parser.add_argument("--ndpa", required=True, help="Path to the .ndpa annotation file.")
    parser.add_argument(
        "--tile-size", type=int, default=TILE_SIZE_PX,
        help=f"Tile edge length in pixels (default: {TILE_SIZE_PX}).",
    )
    args = parser.parse_args()

    try:
        result = compute_density(
            ndpa_path=args.ndpa,
            ndpi_path=args.ndpi,
            tile_size=args.tile_size,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 54)
    print("  NDPI Annotation Density Report")
    print("=" * 54)
    print(f"  Palynomorph annotations  : {result['n_annotations']}")
    print(f"  ROI rectangles           : {result['n_rois']}")
    print(f"  Tile number                : {result['n_tiles']} tiles")
    print(f"  Density                  : {result['density']:.6f}  annotations / tile")
    print("=" * 54)


if __name__ == "__main__":
    main()