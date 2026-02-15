from dataclasses import dataclass, field
import numpy as np
import tifffile
import openslide
import zarr
import matplotlib.pyplot as plt

@dataclass
class FocalPlaneInfo:
    """
    Metadata for one TIFF page representing a single focal plane at one magnification.

    The NDPI file stores every (magnification x z-offset) combination as a
    separate TIFF page. 
    """
    z_offset_nm: int      # focal depth offset in nanometres (negative = below, positive = above)
    magnification: float  # e.g. 5.0, 10.0, 20.0, 40.0
    page_index: int       # index into the TIFF page array
    shape: tuple          # (height, width, channels) of this page

@dataclass
class NDPIData:
    """
    Top-level container for all NDPI data.

    Populated by parse_ndpi(). Holds the physical-to-pixel conversion 
    constants, the focal plane map used by get_page_index(), and the 
    NDPI file path.
    """
    ndpi_path: str
    mpp_x: float              # microns per pixel at max (40x) magnification
    mpp_y: float
    objective_power: float    # highest native magnification (typically 40)
    x_offset_nm: int          # slide-centre offset (Hamamatsu metadata)
    y_offset_nm: int
    full_width: int           # image width in pixels at 40x
    full_height: int          # image height in pixels at 40x
    focal_planes: list[FocalPlaneInfo] = field(default_factory=list)   # list[FocalPlaneInfo]

    def get_page_index(self, magnification: float, z_offset: int) -> int:
        """Given a magnification and z-offset, return the corresponding TIFF page index."""
        for plane in self.focal_planes:
            if plane.magnification == magnification and plane.z_offset_nm == z_offset:
                return plane.page_index
        raise ValueError(f"No page found for magnification {magnification}x and z-offset {z_offset}nm")

    def get_z_offsets(self) -> list[int]:
        """Return sorted unique z-offsets (nm) across all focal planes."""
        return sorted(set(fp.z_offset_nm for fp in self.focal_planes))
    
    def get_tile_from_page(self, page_index: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Extract a rectangular crop from one TIFF page of the NDPI file.

        For small pages (< 50M pixels, e.g. 1.25x-5x), reads the full page and
        slices.  For large pages (10x-40x), uses tifffile's zarr store for lazy
        region access — only the JPEG strips overlapping the crop are decoded,
        keeping memory usage proportional to the *tile* size, not the *page* size.

        Args:
            page_index: TIFF page index (from get_page_index()).
            x, y:       Top-left corner of the crop in page pixel coords.
            w, h:       Width and height of the crop in pixels.

        Returns:
            RGB uint8 array of shape (h, w, 3).
        """
        
        with tifffile.TiffFile(self.ndpi_path) as tif:
            page = tif.pages[page_index]
            page_h, page_w = page.shape[0], page.shape[1]

            # Clamp crop coordinates to page bounds
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(page_w, x + w)
            y1 = min(page_h, y + h)

            if x1 <= x0 or y1 <= y0:
                return np.zeros((h, w, 3), dtype=np.uint8)

            if page_h * page_w < 50_000_000:
                # Small page — safe to load entirely
                img = page.asarray()
                crop = img[y0:y1, x0:x1]
            else:
                # Large page — lazy-read via zarr to avoid multi-GB allocation
                store = page.aszarr()
                z = zarr.open(store, mode="r")
                crop = np.array(z[y0:y1, x0:x1])

            return crop if crop.ndim == 3 else np.stack([crop] * 3, axis=-1)

    def get_tile(self, x: int, y: int, w: int, h: int,
                 magnification: float = 20.) -> np.ndarray:
        """
        Extract a 3-D tile from the NDPI file at the specified magnification and coordinates.
        """
        
        z_offsets = self.get_z_offsets()
        n_focal_planes = len(z_offsets)
        tile = np.zeros((h, w, 3, n_focal_planes), dtype=np.uint8)

        for i, z in enumerate(z_offsets):
            page_index = self.get_page_index(magnification, z)
            if page_index is None:
                continue
            tile[:, :, :, i] = self.get_tile_from_page(page_index, x, y, w, h)

        return tile
    
def load_ndpi(ndpi_path: str) -> NDPIData:
    """
    Open an NDPI file, extract metadata, and build the focal-plane map.

    Uses OpenSlide for high-level properties (MPP, dimensions, offsets) and
    tifffile to iterate through raw TIFF pages and read Hamamatsu-specific tags:
      - Tag 65421 = magnification of this page
      - Tag 65424 = z-offset (focal depth) of this page
    """
    slide = openslide.OpenSlide(ndpi_path)
    props = slide.properties

    data = NDPIData(
        ndpi_path=ndpi_path,
        mpp_x=float(props.get("openslide.mpp-x", 0)),
        mpp_y=float(props.get("openslide.mpp-y", 0)),
        objective_power=float(props.get("openslide.objective-power", 40)),
        x_offset_nm=int(props.get("hamamatsu.XOffsetFromSlideCentre", 0)),
        y_offset_nm=int(props.get("hamamatsu.YOffsetFromSlideCentre", 0)),
        full_width=slide.dimensions[0],
        full_height=slide.dimensions[1],
    )
    slide.close()

    # Walk through every TIFF page and record the (magnification, z-offset) → page_index mapping
    with tifffile.TiffFile(ndpi_path) as tif:
        for i, page in enumerate(tif.pages):
            mag_tag = None
            z_tag = None
            for tag in page.tags.values():
                if tag.code == 65421:
                    mag_tag = tag.value
                if tag.code == 65424:
                    z_tag = tag.value
            if mag_tag is not None and z_tag is not None and mag_tag > 0:
                data.focal_planes.append(FocalPlaneInfo(
                    z_offset_nm=int(z_tag),
                    magnification=float(mag_tag),
                    page_index=i,
                    shape=page.shape,
                ))

    return data

if __name__ == "__main__":
    # Example usage: parse an NDPA file and print the annotations
    ndpi_path = "<file path>"
    ndpi = load_ndpi(ndpi_path)
    tile = ndpi.get_tile(x=0, y=0, w=1024, h=1024, magnification=1.25)
    plt.imsave('output.png', tile[:, :, :, 0])