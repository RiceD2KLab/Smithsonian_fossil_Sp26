from src.data.ndpa_reader import BoundingRegion
from src.data.ndpi_reader import NDPIMetadata
from typing import Tuple

def bounds_to_pixels(
        bounds: BoundingRegion, 
        magnification: float,
        info: NDPIMetadata
        ) -> Tuple[int, int, int, int]:
    """
    Convert a bounding region in nanometer coords to a
    bounding box in pixel coords at the target magnification.

    The conversion accounts for:
      1. The nm-per-pixel scale at the base (40x) magnification
      2. The scale factor from 40x down to the requested magnification
      3. The slide-centre offset that defines where pixel (0,0) sits
         in physical space
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