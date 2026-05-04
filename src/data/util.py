from typing import Tuple

from src.data.ndpa_reader import BoundingRegion
from src.data.ndpi_reader import NDPIMetadata

def bounds_to_pixels(
        bounds: BoundingRegion, 
        magnification: float,
        info: NDPIMetadata
        ) -> Tuple[int, int, int, int]:
    """
    Convert a bounding region in nanometer coordinates to pixel coordinates.

    Accounts for the nm-per-pixel scale at the base (40x) magnification, the
    scale factor down to the requested magnification, and the slide-centre
    offset that defines where pixel (0, 0) sits in physical space.

    Args:
        bounds: Bounding region whose get_bounding_box() returns (x, y, w, h) in nm.
        magnification: Target magnification for the output pixel coordinates.
        info: NDPI slide metadata containing conversion constants.

    Returns:
        Tuple (x, y, width, height) in pixels at the requested magnification.
    """
    nm_per_px_40x = info.mpp_x * 1000.0
    scale = info.objective_power / magnification
    nm_per_px = nm_per_px_40x * scale

    # Pixel (0,0) in nm-space:  offset - (half image width in nm)
    origin_x_nm = info.x_offset_nm - (info.full_width / 2.0) * nm_per_px_40x
    origin_y_nm = info.y_offset_nm - (info.full_height / 2.0) * nm_per_px_40x

    bbox = bounds.get_bounding_box()

    px_x = (bbox[0] - origin_x_nm) / nm_per_px
    px_y = (bbox[1] - origin_y_nm) / nm_per_px
    px_width = bbox[2] / nm_per_px
    px_height = bbox[3] / nm_per_px
    return int(round(px_x)), int(round(px_y)), int(round(px_width)), int(round(px_height))

def pixels_to_nm_bbox(
        x1_px: float,
        y1_px: float,
        x2_px: float,
        y2_px: float,
        magnification: float,
        info: NDPIMetadata,
        ) -> Tuple[float, float, float, float]:
    """
    Convert a bounding box in pixel coordinates to nanometer coordinates.

    The conversion uses the same NDPI slide metadata as `bounds_to_pixels` and
    accounts for:
      1. The nm-per-pixel scale at the base (40x) magnification
      2. The scale factor from 40x down to the requested magnification
      3. The slide-centre offset that defines where pixel (0,0) sits
       in physical space

    Args:
        x1_px: Left edge of the bounding box in pixels.
        y1_px: Top edge of the bounding box in pixels.
        x2_px: Right edge of the bounding box in pixels.
        y2_px: Bottom edge of the bounding box in pixels.
        magnification: Target magnification of the pixel coordinates.
        info: NDPI metadata describing the slide geometry.

    Returns:
        A tuple `(x_nm, y_nm, width_nm, height_nm)` in nanometer space.
    """
    nm_per_px_40x = info.mpp_x * 1000.0
    scale = info.objective_power / magnification
    nm_per_px = nm_per_px_40x * scale

    origin_x_nm = info.x_offset_nm - (info.full_width / 2.0) * nm_per_px_40x
    origin_y_nm = info.y_offset_nm - (info.full_height / 2.0) * nm_per_px_40x

    x_nm = x1_px * nm_per_px + origin_x_nm
    y_nm = y1_px * nm_per_px + origin_y_nm
    width_nm = max(x2_px - x1_px, 0.0) * nm_per_px
    height_nm = max(y2_px - y1_px, 0.0) * nm_per_px
    return x_nm, y_nm, width_nm, height_nm