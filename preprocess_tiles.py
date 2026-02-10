"""
Data Pre-processing Pipeline for Palynomorph Detection.

Extracts tiles from annotated regions of an NDPI z-stack slide and exports
them as either:
  - PNGs  (one image per focal plane per tile)
  - HDF5  (one .h5 file per tile with shape (Z, H, W, 3) — all focal planes
            stacked)

Overall pipeline flow:
  1. discover_slide()     — Load slide metadata + NDPA annotations
  2. select_regions()     — Pick which annotations to tile (rectangles or circles)
  3. annotation_to_pixel_bbox() — Convert each annotation from nm → pixel coords
  4. compute_tile_grid()  — Subdivide each region into a grid of tile positions
  5. export_tile_streaming() — For each tile, loop through z-planes one at a time:
       a. extract_roi_from_page() — Read the tile crop from each focal plane
       b. Write to HDF5 dataset slot or PNG file immediately (never hold all planes in RAM)
  6. Write manifest.csv   — One row per tile for easy dataset loading

Usage:
    python preprocess_tiles.py --ndpi slide.ndpi --info
    python preprocess_tiles.py --ndpi slide.ndpi --mag 20 --dry-run
    python preprocess_tiles.py --ndpi slide.ndpi --mag 20 --format h5
    python preprocess_tiles.py --ndpi slide.ndpi --mag 20 --mode circles --label paly --format png
    python preprocess_tiles.py --ndpi slide.ndpi --mag 20 --tile-size 512 --overlap 0.25 --format h5
"""

import argparse
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import cv2
import h5py
import numpy as np
import tifffile
import zarr


# ========================== DATA CLASSES ====================================
#
# These hold the structured metadata parsed from the NDPI + NDPA files.
# They are populated once by discover_slide() and then passed through the
# entire pipeline so every downstream function has access to slide geometry,
# focal plane layout, and annotation coordinates.
# ============================================================================

@dataclass
class Annotation:
    """A single annotation parsed from the NDPA XML file.

    Annotations mark regions of interest on the slide:
      - "circle"    — individual palynomorph / pollen / spore markers
      - "rectangle" — large bounding region covering an annotated area of interest

    The pipeline uses these to decide *where* to generate tiles.
    """
    id: int
    title: str           # label category, e.g. "paly", "pol", "spo"
    ann_type: str        # "circle" or "rectangle"
    color: str           # hex color from NDP.view
    center_x_nm: int     # centre x in nanometres (NDPI physical space)
    center_y_nm: int     # centre y in nanometres
    radius_nm: Optional[int] = None     # circle radius (nm), None for rectangles
    points_nm: Optional[list] = None    # corner points (nm), None for circles

    @property
    def bbox_nm(self) -> Optional[tuple[int, int, int, int]]:
        """Compute the axis-aligned bounding box (x_min, y_min, x_max, y_max) in nm.

        Used by annotation_to_pixel_bbox() to convert this annotation into a
        pixel-space region that can be subdivided into tiles.
        """
        if self.ann_type == "circle" and self.radius_nm:
            r = self.radius_nm
            return (
                self.center_x_nm - r,
                self.center_y_nm - r,
                self.center_x_nm + r,
                self.center_y_nm + r,
            )
        elif self.points_nm:
            xs = [p[0] for p in self.points_nm]
            ys = [p[1] for p in self.points_nm]
            return (min(xs), min(ys), max(xs), max(ys))
        return None


@dataclass
class FocalPlaneInfo:
    """Metadata for one TIFF page representing a single focal plane at one magnification.

    The NDPI file stores every (magnification x z-offset) combination as a
    separate TIFF page.  The pipeline uses page_index to read the correct
    page when extracting a tile crop via extract_roi_from_page().
    """
    z_offset_nm: int      # focal depth offset in nanometres (negative = below, positive = above)
    magnification: float  # e.g. 5.0, 10.0, 20.0, 40.0
    page_index: int       # index into the TIFF page array
    shape: tuple          # (height, width, channels) of this page


@dataclass
class SlideInfo:
    """Top-level container for all slide metadata.

    Populated once by discover_slide() and threaded through the entire
    pipeline.  Holds the physical-to-pixel conversion constants needed by
    nm_to_pixel(), the focal plane map used by get_page_index(), and the
    parsed annotations used by select_regions().
    """
    ndpi_path: str
    mpp_x: float              # microns per pixel at max (40x) magnification
    mpp_y: float
    objective_power: float    # highest native magnification (typically 40)
    x_offset_nm: int          # slide-centre offset (Hamamatsu metadata)
    y_offset_nm: int
    full_width: int           # image width in pixels at 40x
    full_height: int          # image height in pixels at 40x
    focal_planes: list = field(default_factory=list)   # list[FocalPlaneInfo]
    annotations: list = field(default_factory=list)     # list[Annotation]


# ========================== NDPA PARSING ====================================
#
# Step 1a: Parse the companion .ndpa XML file to extract annotations.
# Called by discover_slide() — the annotations end up in SlideInfo.annotations.
# ============================================================================

def parse_ndpa(ndpa_path: str) -> list[Annotation]:
    """Parse an NDPA XML file and return a list of Annotation objects.

    The NDPA file is a companion to the NDPI slide, containing user-drawn
    annotations (circles around palynomorphs, rectangles around regions of
    interest).  Each <ndpviewstate> element describes one annotation with
    its type, position, and label.

    Pipeline role: called by discover_slide() to populate
    SlideInfo.annotations, which select_regions() later filters.
    """
    tree = ET.parse(ndpa_path)
    root = tree.getroot()
    annotations = []

    for vs in root.findall("ndpviewstate"):
        ann_id = int(vs.get("id", 0))
        title = vs.findtext("title", "").strip()
        ann_elem = vs.find("annotation")
        if ann_elem is None:
            continue

        ann_type = ann_elem.get("type", "")
        color = ann_elem.get("color", "#000000")

        if ann_type == "circle":
            cx = int(ann_elem.findtext("x", "0"))
            cy = int(ann_elem.findtext("y", "0"))
            radius = int(ann_elem.findtext("radius", "0"))
            annotations.append(Annotation(
                id=ann_id, title=title, ann_type="circle", color=color,
                center_x_nm=cx, center_y_nm=cy, radius_nm=radius,
            ))
        elif ann_type == "freehand":
            # NDP.view stores rectangles as "freehand" with 4 corner points
            points = []
            for pt in ann_elem.findall(".//point"):
                px = int(pt.findtext("x", "0"))
                py = int(pt.findtext("y", "0"))
                points.append((px, py))
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                annotations.append(Annotation(
                    id=ann_id, title=title, ann_type="rectangle", color=color,
                    center_x_nm=cx, center_y_nm=cy, points_nm=points,
                ))

    return annotations


# ========================== SLIDE DISCOVERY =================================
#
# Step 1b: Open the NDPI, read slide-level metadata via OpenSlide, and
# enumerate every TIFF page to build the focal-plane map.
# (TIFF is the raw image data, so each page is a different focal plane at a
# different magnification.)
# This is the entry point — everything else depends on the SlideInfo it returns.
# ============================================================================

def discover_slide(ndpi_path: str) -> SlideInfo:
    """Open an NDPI file, extract metadata, and build the focal-plane map.

    Uses OpenSlide for high-level properties (MPP, dimensions, offsets) and
    tifffile to iterate through raw TIFF pages and read Hamamatsu-specific tags:
      - Tag 65421 = magnification of this page
      - Tag 65424 = z-offset (focal depth) of this page

    Also calls parse_ndpa() if a .ndpa companion file exists.

    Pipeline role: this is the first function called.  Its output (SlideInfo)
    is the single source of truth passed to every other function.
    """
    import openslide
    slide = openslide.OpenSlide(ndpi_path)
    props = slide.properties

    info = SlideInfo(
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
                info.focal_planes.append(FocalPlaneInfo(
                    z_offset_nm=int(z_tag),
                    magnification=float(mag_tag),
                    page_index=i,
                    shape=page.shape,
                ))

    # Load annotations from companion .ndpa file
    ndpa_path = ndpi_path + ".ndpa"
    if os.path.exists(ndpa_path):
        info.annotations = parse_ndpa(ndpa_path)

    return info


def get_z_offsets(info: SlideInfo) -> list[int]:
    """Return sorted unique z-offsets (nm) across all focal planes.

    Pipeline role: used by export_tile_streaming() to know how many planes
    to iterate and what z-values to record in the HDF5 / PNG filenames.
    """
    return sorted(set(fp.z_offset_nm for fp in info.focal_planes))


def get_magnifications(info: SlideInfo) -> list[float]:
    """Return sorted unique magnification levels available in the slide.

    Pipeline role: used by main() to validate the --mag argument.
    """
    return sorted(set(fp.magnification for fp in info.focal_planes))


def get_page_index(info: SlideInfo, z_offset_nm: int, magnification: float) -> Optional[int]:
    """Look up the TIFF page index for a specific (z-offset, magnification) pair.

    Pipeline role: called inside export_tile_streaming() for each z-plane
    to find which TIFF page to pass to extract_roi_from_page().
    """
    for fp in info.focal_planes:
        if fp.z_offset_nm == z_offset_nm and fp.magnification == magnification:
            return fp.page_index
    return None


# ========================== COORDINATE CONVERSION ===========================
#
# NDPA annotations are in nanometre coordinates relative to the slide centre.
# TIFF pages use pixel coordinates starting at (0, 0) top-left.
# These functions bridge the two coordinate systems so that annotation
# bounding boxes can be converted to pixel regions for cropping.
# ============================================================================

def nm_to_pixel(x_nm: int, y_nm: int, info: SlideInfo, magnification: float) -> tuple[int, int]:
    """Convert a point from NDPA nanometre coords to pixel coords at the target magnification.

    The conversion accounts for:
      1. The nm-per-pixel scale at the base (40x) magnification
      2. The scale factor from 40x down to the requested magnification
      3. The slide-centre offset that defines where pixel (0,0) sits
         in physical space

    Pipeline role: called by annotation_to_pixel_bbox() to position each
    annotation's bounding box in the pixel grid of the target magnification.
    """
    nm_per_px_40x = info.mpp_x * 1000.0
    scale = info.objective_power / magnification
    nm_per_px = nm_per_px_40x * scale

    # Pixel (0,0) in nm-space:  offset - (half image width in nm)
    origin_x_nm = info.x_offset_nm - (info.full_width / 2.0) * nm_per_px_40x
    origin_y_nm = info.y_offset_nm - (info.full_height / 2.0) * nm_per_px_40x

    px_x = (x_nm - origin_x_nm) / nm_per_px
    px_y = (y_nm - origin_y_nm) / nm_per_px
    return int(round(px_x)), int(round(px_y))


def nm_size_to_pixels(size_nm: int, info: SlideInfo, magnification: float) -> int:
    """Convert a distance in nanometres to a pixel count at the target magnification.

    Pipeline role: called by annotation_to_pixel_bbox() to convert annotation
    width/height from nm to pixels.
    """
    nm_per_px_40x = info.mpp_x * 1000.0
    scale = info.objective_power / magnification
    nm_per_px = nm_per_px_40x * scale
    return max(1, int(round(size_nm / nm_per_px)))


# ========================== REGION EXTRACTION ===============================
#
# The lowest-level I/O function.  Reads a rectangular crop from a single
# TIFF page.  Uses zarr-backed lazy reading for large pages (20x, 40x) to
# avoid loading the entire multi-GB page into RAM.
# Called once per (tile × z-plane) inside export_tile_streaming().
# ============================================================================

def extract_roi_from_page(ndpi_path: str, page_index: int, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Extract a rectangular crop from one TIFF page of the NDPI file.

    For small pages (< 50M pixels, e.g. 1.25x-5x), reads the full page and
    slices.  For large pages (10x-40x), uses tifffile's zarr store for lazy
    region access — only the JPEG strips overlapping the crop are decoded,
    keeping memory usage proportional to the *tile* size, not the *page* size.

    Args:
        ndpi_path:  Path to the .ndpi file.
        page_index: TIFF page index (from get_page_index()).
        x, y:       Top-left corner of the crop in page pixel coords.
        w, h:       Width and height of the crop in pixels.

    Returns:
        RGB uint8 array of shape (h, w, 3).

    Pipeline role: the core I/O primitive.  export_tile_streaming() calls
    this once per z-plane per tile to get each focal plane's image data.
    """
    with tifffile.TiffFile(ndpi_path) as tif:
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


# ========================== INFO PRINTING ===================================
#
# Quick diagnostic output.  Used by --info and also printed at the start of
# every pipeline run so the user can verify the slide was loaded correctly.
# ============================================================================

def print_info(info: SlideInfo):
    """Print a human-readable summary of the slide's properties and annotations.

    Pipeline role: called at the start of every run (and exclusively on --info)
    to give the user a quick overview of dimensions, magnifications, z-range,
    and annotation counts before any extraction begins.
    """
    print(f"\n{'='*70}")
    print(f"Slide: {os.path.basename(info.ndpi_path)}")
    print(f"{'='*70}")
    print(f"  Dimensions (40x): {info.full_width} x {info.full_height} px")
    print(f"  MPP: {info.mpp_x:.4f} µm/px")
    print(f"  Objective: {info.objective_power}x")
    print(f"  Offset from centre: ({info.x_offset_nm}, {info.y_offset_nm}) nm")

    z_offsets = get_z_offsets(info)
    mags = get_magnifications(info)
    print(f"  Focal planes: {len(z_offsets)}")
    print(f"    Z range: {z_offsets[0]/1000:+.0f} to {z_offsets[-1]/1000:+.0f} µm "
          f"(step = {(z_offsets[1]-z_offsets[0])/1000:.0f} µm)")
    print(f"  Magnification levels: {mags}")

    if info.annotations:
        print(f"  Annotations: {len(info.annotations)}")
        titles = {}
        for a in info.annotations:
            titles[a.title] = titles.get(a.title, 0) + 1
        for t, c in sorted(titles.items(), key=lambda x: -x[1]):
            print(f"    '{t}': {c}")
    print()


# ========================== TILE COMPUTATION ================================
#
# These functions convert annotation bounding boxes into concrete lists of
# (x, y, w, h) tile positions.  This is the spatial planning step — no pixel
# data is read here, just coordinate math.
# ============================================================================

def compute_tile_grid(
    start_x: int,
    start_y: int,
    region_w: int,
    region_h: int,
    tile_size: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    """Subdivide a rectangular region into a grid of fixed-size tile positions.

    Tiles are laid out left-to-right, top-to-bottom, advancing by
    `tile_size * (1 - overlap)` pixels each step.  Tiles at the edges may
    extend beyond the region boundary (they will be zero-padded during
    extraction by _ensure_plane_shape()).

    Args:
        start_x, start_y: Top-left corner of the region in pixels.
        region_w, region_h: Size of the region in pixels.
        tile_size: Width and height of each tile in pixels.
        overlap: Fraction of overlap between adjacent tiles (0.0 = none).

    Returns:
        List of (x, y, w, h) tuples — one per tile.

    Pipeline role: called by run_pipeline() for each region returned by
    select_regions().  The resulting tile list is what export_tile_streaming()
    iterates over.
    """
    end_x = start_x + region_w
    end_y = start_y + region_h

    # How far to advance when placing the next tile
    increment = max(1, int(tile_size * (1.0 - overlap)))
    tiles = []

    curr_y = start_y
    while curr_y < end_y:
        curr_x = start_x
        while curr_x < end_x:
            tiles.append((curr_x, curr_y, tile_size, tile_size))
            curr_x += increment
        curr_y += increment

    return tiles


def annotation_to_pixel_bbox(
    annotation: Annotation,
    info: SlideInfo,
    magnification: float,
    padding: float = 1.0,
) -> tuple[int, int, int, int]:
    """Convert an annotation's bounding box from nm to pixel coords.

    Applies optional padding (multiplier on width/height) and returns the
    top-left corner + dimensions in the pixel space of the target
    magnification level.

    Args:
        annotation: The Annotation to convert.
        info: SlideInfo with the conversion constants.
        magnification: Target magnification (e.g. 20.0).
        padding: Multiplier to enlarge the bbox (1.0 = exact fit).

    Returns:
        (x, y, w, h) in pixels at the target magnification.

    Pipeline role: called by select_regions() to turn each annotation's
    nanometre bbox into the pixel region that compute_tile_grid() subdivides.
    """
    bbox = annotation.bbox_nm
    if bbox is None:
        raise ValueError(f"Annotation {annotation.id} has no bounding box")

    x_min_nm, y_min_nm, x_max_nm, y_max_nm = bbox
    w_nm = int((x_max_nm - x_min_nm) * padding)
    h_nm = int((y_max_nm - y_min_nm) * padding)
    cx_nm = (x_min_nm + x_max_nm) // 2
    cy_nm = (y_min_nm + y_max_nm) // 2

    cx_px, cy_px = nm_to_pixel(cx_nm, cy_nm, info, magnification)
    w_px = nm_size_to_pixels(w_nm, info, magnification)
    h_px = nm_size_to_pixels(h_nm, info, magnification)

    return cx_px - w_px // 2, cy_px - h_px // 2, w_px, h_px


# ========================== STREAMING TILE EXPORT ===========================
#
# The core export loop.  For each tile position, iterates through every
# z-plane one at a time: extract the crop, write it to disk, free the memory.
# Peak RAM usage = one tile plane (~3 MB at 1024×1024) regardless of how many
# z-planes exist.
# ============================================================================

def _ensure_plane_shape(img: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    """Zero-pad an extracted crop if it is smaller than the expected tile size.

    This happens when a tile at the edge of a region extends beyond the
    TIFF page boundary.  The result is always exactly (tile_h, tile_w, 3).

    Pipeline role: called inside export_tile_streaming() after each
    extract_roi_from_page() call to guarantee uniform tile dimensions.
    """
    if img.shape[0] != tile_h or img.shape[1] != tile_w:
        padded = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        padded[:min(img.shape[0], tile_h), :min(img.shape[1], tile_w)] = \
            img[:min(img.shape[0], tile_h), :min(img.shape[1], tile_w)]
        return padded
    return img


def _draw_annotations(img: np.ndarray, annotations: list[Annotation], info: SlideInfo,
                      magnification: float, tile_x: int, tile_y: int) -> np.ndarray:
    """Draw all annotation circles on an image in tile-relative coordinates.

    For each annotation with a radius, converts its nm-space centre and radius
    to pixel coordinates relative to the tile's top-left corner, then draws a
    circle outline.  Circles that fall outside the tile are clipped by OpenCV.

    Pipeline role: called inside export_tile_streaming() when --annotate is on.
    """
    out = img.copy()
    for ann in annotations:
        if ann.radius_nm is None:
            continue

        # Annotation centre → absolute pixel coords → tile-relative coords
        cx_px, cy_px = nm_to_pixel(ann.center_x_nm, ann.center_y_nm,
                                   info, magnification)
        rel_cx = cx_px - tile_x
        rel_cy = cy_px - tile_y
        radius_px = nm_size_to_pixels(ann.radius_nm, info, magnification)

        # Skip circles whose centre is far outside this tile (optimisation)
        if (rel_cx + radius_px < 0 or rel_cx - radius_px > img.shape[1] or
                rel_cy + radius_px < 0 or rel_cy - radius_px > img.shape[0]):
            continue

        # Parse annotation colour (hex string like "#ff0000" or "#ff0000ff")
        color_hex = ann.color.lstrip("#")
        if len(color_hex) >= 6:
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
        else:
            r, g, b = 0, 255, 0  # fallback green

        cv2.circle(out, (rel_cx, rel_cy), radius_px, (r, g, b), thickness=2)
    return out


def export_tile_streaming(
    info: SlideInfo,
    tile_x: int, tile_y: int,
    tile_w: int, tile_h: int,
    magnification: float,
    fmt: str,
    output_path: str,
    annotations: Optional[list[Annotation]] = None,
) -> int:
    """Extract and export one tile across all z-planes, streaming to disk.

    Processes z-planes one at a time so that at most one plane (~3 MB at
    1024×1024) is in memory at any moment.

    HDF5 mode:
      - Pre-allocates a gzip-compressed dataset of shape (Z, H, W, 3) on disk
      - Writes each plane into its dataset slot, then frees the array
      - Also stores z_offsets and tile metadata as HDF5 attributes

    PNG mode:
      - Creates a subdirectory for this tile
      - Writes each plane as a separate PNG named by z-offset (e.g. z+02000nm.png)

    If annotations are provided, their circles are drawn on every plane.

    Args:
        info: SlideInfo with slide metadata and focal plane map.
        tile_x, tile_y: Top-left pixel position of this tile.
        tile_w, tile_h: Tile dimensions in pixels.
        magnification: Magnification level to extract from.
        fmt: "h5" or "png".
        output_path: File path (.h5) or directory path (png).
        annotations: Optional list of Annotations to draw on each plane.

    Returns:
        Number of z-planes written.

    Pipeline role: this is the innermost function in the main tile loop of
    run_pipeline().  It orchestrates the per-plane calls to
    extract_roi_from_page() and writes the output.
    """
    z_offsets = get_z_offsets(info)

    if fmt == "h5":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as f:
            # Pre-allocate on-disk dataset — chunked by plane for streaming writes
            ds = f.create_dataset(
                "images",
                shape=(len(z_offsets), tile_h, tile_w, 3),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
                chunks=(1, tile_h, tile_w, 3),
            )
            f.create_dataset("z_offsets", data=np.array(z_offsets, dtype=np.int64))
            f.attrs["magnification"] = magnification
            f.attrs["tile_x"] = tile_x
            f.attrs["tile_y"] = tile_y
            f.attrs["tile_w"] = tile_w
            f.attrs["tile_h"] = tile_h
            f.attrs["num_z_planes"] = len(z_offsets)

            for i, z in enumerate(z_offsets):
                page_idx = get_page_index(info, z, magnification)
                if page_idx is None:
                    continue
                img = extract_roi_from_page(info.ndpi_path, page_idx, tile_x, tile_y, tile_w, tile_h)
                img = _ensure_plane_shape(img, tile_h, tile_w)
                if annotations:
                    img = _draw_annotations(img, annotations, info, magnification, tile_x, tile_y)
                ds[i] = img
                del img

    elif fmt == "png":
        os.makedirs(output_path, exist_ok=True)
        for z in z_offsets:
            page_idx = get_page_index(info, z, magnification)
            if page_idx is None:
                continue
            img = extract_roi_from_page(info.ndpi_path, page_idx, tile_x, tile_y, tile_w, tile_h)
            img = _ensure_plane_shape(img, tile_h, tile_w)
            if annotations:
                img = _draw_annotations(img, annotations, info, magnification, tile_x, tile_y)
            cv2.imwrite(os.path.join(output_path, f"z{z:+06d}nm.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            del img

    return len(z_offsets)


# ========================== PIPELINE ========================================
#
# The top-level orchestration functions.  select_regions() decides *what* to
# tile, run_pipeline() decides *how* and drives the main loop.
# ============================================================================

def select_regions(
    info: SlideInfo,
    mode: str,
    magnification: float,
    label: Optional[str],
    annotation_id: Optional[int],
    padding: float,
) -> list[tuple[str, int, int, int, int, Optional[Annotation]]]:
    """Choose which annotations to tile based on --mode, --label, and --annotation-id.

    "rectangle" mode: selects the large rectangular region annotation(s).
    "circles" mode:   selects individual circle annotations, optionally
                      filtered by title label or specific annotation ID.

    Each selected annotation is converted from nm to pixel coords via
    annotation_to_pixel_bbox().

    Returns:
        List of (region_label, x_px, y_px, w_px, h_px, annotation) tuples.
        The annotation is included so --annotate can draw it on exported tiles.

    Pipeline role: called at the start of run_pipeline() to determine
    the set of regions that compute_tile_grid() will subdivide.
    """
    regions = []

    if mode == "rectangle":
        rects = [a for a in info.annotations if a.ann_type == "rectangle"]
        if not rects:
            print("Error: no rectangle annotations found.")
            sys.exit(1)
        for rect in rects:
            label_str = f"rect{rect.id}_{rect.title or 'region'}"
            x, y, w, h = annotation_to_pixel_bbox(rect, info, magnification, padding=padding)
            regions.append((label_str, x, y, w, h, None))
            print(f"  Region: {label_str} — {w}×{h} px at ({x}, {y})")

    elif mode == "circles":
        circles = [a for a in info.annotations if a.ann_type == "circle"]
        if annotation_id is not None:
            circles = [a for a in circles if a.id == annotation_id]
        if label:
            circles = [a for a in circles if a.title.lower() == label.lower()]
        if not circles:
            print("Error: no matching circle annotations found.")
            sys.exit(1)
        for ann in circles:
            label_str = f"ann{ann.id}_{ann.title}"
            x, y, w, h = annotation_to_pixel_bbox(ann, info, magnification, padding=padding)
            regions.append((label_str, x, y, w, h, ann))
        print(f"  Selected {len(regions)} circle annotation(s)")

    return regions


def run_pipeline(
    info: SlideInfo,
    magnification: float,
    mode: str,
    label: Optional[str],
    annotation_id: Optional[int],
    tile_size: int,
    overlap: float,
    fmt: str,
    output_dir: str,
    padding: float,
    dry_run: bool,
    annotate: bool = False,
):
    """Top-level pipeline driver: region selection → tile grid → streaming export → manifest.

    Steps:
      1. select_regions()        — pick annotations to tile
      2. compute_tile_grid()     — subdivide each region into tiles
      3. export_tile_streaming() — extract + write each tile (one z-plane at a time)
      4. Write manifest.csv      — one row per tile for dataset loading
    """
    print(f"\n{'='*70}")
    print(f"Pre-processing Pipeline")
    print(f"{'='*70}")
    print(f"  Mode:          {mode}")
    print(f"  Magnification: {magnification}x")
    print(f"  Tile size:     {tile_size}×{tile_size} px")
    print(f"  Overlap:       {overlap*100:.0f}%")
    print(f"  Format:        {fmt}")
    print(f"  Output:        {output_dir}")
    print(f"  Padding:       {padding}")
    print()

    # Step 1: decide which annotations to tile
    regions = select_regions(info, mode, magnification, label, annotation_id, padding)

    z_offsets = get_z_offsets(info)
    num_z = len(z_offsets)
    print(f"\n  Focal planes:  {num_z} (z range: "
          f"{z_offsets[0]/1000:+.0f} to {z_offsets[-1]/1000:+.0f} µm)")

    # Step 2: compute tile positions for each region
    all_tiles = []  # (region_label, x, y, w, h, annotation_or_None)
    for region_label, rx, ry, rw, rh, ann in regions:
        if mode == "circles":
            # Center the tile grid on the annotation so the palynomorph is in
            # the middle of the tile
            cx, cy = rx + rw // 2, ry + rh // 2
            grid_w = max(rw, tile_size)
            grid_h = max(rh, tile_size)
            rx = cx - grid_w // 2
            ry = cy - grid_h // 2
            rw, rh = grid_w, grid_h
        tiles = compute_tile_grid(rx, ry, rw, rh, tile_size, overlap)
        for t in tiles:
            all_tiles.append((region_label, *t, ann))
        print(f"  {region_label}: {rw}×{rh} px → {len(tiles)} tiles")

    total_tiles = len(all_tiles)
    print(f"\n  Total tiles: {total_tiles}")
    print(f"  Memory per z-plane: {tile_size * tile_size * 3 / 1e6:.1f} MB  "
          f"(only 1 plane in memory at a time)")

    if dry_run:
        print("\n  [DRY RUN] No files written.")
        return

    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []
    start_time = time.time()

    # Step 3: extract and export each tile
    for i, (region_label, tx, ty, tw, th, ann) in enumerate(all_tiles):
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (total_tiles - i - 1) / rate if rate > 0 else 0
        print(f"\r  [{i+1}/{total_tiles}] {(i+1)/total_tiles*100:5.1f}%  "
              f"ETA {eta:.0f}s  tile ({tx},{ty}) — {region_label}",
              end="", flush=True)

        safe_label = region_label.replace("/", "_").replace(" ", "_")
        tile_name = f"tile_{tx:05d}_{ty:05d}"

        if fmt == "h5":
            rel_path = os.path.join(safe_label, f"{tile_name}.h5")
        else:
            rel_path = os.path.join(safe_label, tile_name)
        out_path = os.path.join(output_dir, rel_path)

        # Build annotation list for this tile when --annotate is on
        if annotate:
            if ann is not None:
                draw_anns = [ann]
            else:
                # Rectangle mode: draw all circle annotations that may overlap this tile
                draw_anns = [a for a in info.annotations if a.radius_nm is not None]
        else:
            draw_anns = None
        num_written = export_tile_streaming(info, tx, ty, tw, th, magnification, fmt, out_path, draw_anns)

        manifest_rows.append({
            "annotation": region_label,
            "tile_file": rel_path,
            "tile_x": tx, "tile_y": ty,
            "tile_w": tw, "tile_h": th,
            "num_z_planes": num_written,
            "magnification": magnification,
            "format": fmt,
        })

    print(f"\n\n  Done! {total_tiles} tiles exported in {time.time() - start_time:.1f}s")

    # Step 4: write manifest for downstream dataset loading
    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"  Manifest saved to {manifest_path}")


# ========================== CLI =============================================

def main():
    parser = argparse.ArgumentParser(
        description="Tile NDPI z-stack annotated regions for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ndpi", required=True, help="Path to the NDPI file")
    parser.add_argument("--info", action="store_true", help="Print slide info and exit")
    parser.add_argument("--mag", type=float, default=20.0, help="Magnification level (default: 20.0)")
    parser.add_argument("--mode", choices=["rectangle", "circles"], default="rectangle",
                        help="'rectangle' tiles the large rect region, 'circles' tiles individual circles (default: rectangle)")
    parser.add_argument("--label", type=str, default=None, help="Filter annotations by title (e.g. 'paly')")
    parser.add_argument("--annotation-id", type=int, default=None, help="Tile a specific annotation by NDPA ID")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size in pixels (default: 1024)")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction 0.0–1.0 (default: 0.0)")
    parser.add_argument("--format", choices=["h5", "png"], default="h5", dest="fmt",
                        help="Export format: 'h5' (HDF5 3D stack) or 'png' (default: h5)")
    parser.add_argument("--output", type=str, default="tiles_output", help="Output directory (default: tiles_output)")
    parser.add_argument("--padding", type=float, default=1.0, help="Padding multiplier around bbox (default: 1.0)")
    parser.add_argument("--annotate", action="store_true",
                        help="Draw annotation circle overlay on exported tiles")
    parser.add_argument("--dry-run", action="store_true", help="Report tile layout without extracting")

    args = parser.parse_args()

    print(f"Loading slide: {args.ndpi}")
    info = discover_slide(args.ndpi)

    if args.info:
        print_info(info)
        return

    available_mags = get_magnifications(info)
    if args.mag not in available_mags:
        print(f"Error: magnification {args.mag}x not available. Choose from: {available_mags}")
        sys.exit(1)

    print_info(info)

    run_pipeline(
        info=info, magnification=args.mag, mode=args.mode,
        label=args.label, annotation_id=args.annotation_id,
        tile_size=args.tile_size, overlap=args.overlap,
        fmt=args.fmt, output_dir=args.output,
        padding=args.padding, dry_run=args.dry_run,
        annotate=args.annotate,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
