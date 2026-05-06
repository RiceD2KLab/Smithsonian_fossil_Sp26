"""
Generate H5 tile datasets from NDPI whole-slide images and NDPA annotations.

Loads NDPI/NDPA file pairs, subdivides each annotated ROI into a grid of
fixed-size tiles, and writes the results (image data + bounding boxes +
labels) to per-slide H5 files. The main entry point iterates over all NDPI
files in an input directory and produces one H5 file per slide.
"""

import argparse
from src.data.ndpa_reader import NDPAData
from src.data.ndpi_reader import NDPIData
from src.data.util import bounds_to_pixels
import os
import numpy as np
import json
import h5py
from tqdm import tqdm
import pandas as pd

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
    `tile_size * (1 - overlap)` pixels each step. Tiles that would 
    extend beyond the region boundary are shifted back so they fit 
    exactly within the region.

    Args:
        start_x, start_y: Top-left corner of the region in pixels.
        region_w, region_h: Size of the region in pixels.
        tile_size: Width and height of each tile in pixels.
        overlap: Fraction of overlap between adjacent tiles (0.0 = none).

    Returns:
        List of (x, y, w, h) tuples — one per tile.
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
            tiles.append((
                min(curr_x + tile_size, end_x) - tile_size, # don't let tile extend beyond region boundary
                min(curr_y+ tile_size, end_y) - tile_size, 
                tile_size, 
                tile_size))
            curr_x += increment
        curr_y += increment

    return tiles

def generate_tiles(
        ndpi_data: NDPIData,
        ndpa_data: NDPAData,
        magnification: float,
        tile_size: int,
        overlap: float,
        output_dir: str,
        label_map: dict,
    ):
    """
    Generate tiles from a single NDPI file covering all ROIs in
    the associated NDPA file, and save them to disk along with 
    their annotations.

    Args:
        ndpi_data: Loaded NDPIData object for the image.
        ndpa_data: Loaded NDPAData object for the annotations.
        magnification: Magnification level to extract tiles at (e.g. 20).
        tile_size: Size of each tile in pixels (e.g. 1024).
        overlap: Fraction of tile overlap (e.g. 0.0 for no overlap).
        output_dir: Directory to save the generated H5 file.
        label_map: Mapping from annotation category names to integer labels.

    Returns:
        None
    """

    # Generate tile grid covering all ROIs
    tiles = []
    rois = []
    for region in ndpa_data.rois:
        bounds = bounds_to_pixels(region.bounds, magnification, ndpi_data.metadata)
        tiles.extend(compute_tile_grid(*bounds, tile_size=tile_size, overlap=overlap))
        rois.append(bounds)

    print(f"Generated {len(tiles)} tiles.")

    # Convert annotations to pixels and collect labels
    annotations = []  # List of dicts: {bbox, label}
    for ann in ndpa_data.palynomorphs:
        bbox = bounds_to_pixels(ann.bounds, magnification, ndpi_data.metadata)
        annotations.append({
            "bbox": bbox,
            "label": label_map.get(ann.label.lower(), -1)
        })

        if annotations[-1]["label"] == -1:
            print(f"Warning: Unrecognized label '{ann.label}' in annotation, assigned label -1.")

    # For each tile, find annotations that intersect it
    tile_annotations = []  # List of (tile, [annotation_indices])
    for tile in tiles:
        x0, y0, w, h = tile
        x1, y1 = x0 + w, y0 + h
        anns_in_tile = []
        for i, ann in enumerate(annotations):
            ax0, ay0, aw, ah = ann["bbox"]
            ax1, ay1 = ax0 + aw, ay0 + ah
            # Check for intersection
            if not (ax1 <= x0 or ax0 >= x1 or ay1 <= y0 or ay0 >= y1):
                anns_in_tile.append(i)
        tile_annotations.append((tile, anns_in_tile))
    
    # Create H5 file for this image
    image_name = os.path.splitext(os.path.basename(ndpi_data.ndpi_path))[0]
    h5_path = os.path.join(output_dir, f"{image_name}.h5")
    with h5py.File(h5_path, "w") as h5f:
        
        # For each tile, extract the image data and save it along with the annotations
        for tile_idx, (tile, anns_in_tile) in enumerate(tqdm(tile_annotations)):
            x0, y0, w, h = tile
            bboxes = []
            uncropped_bboxes = []
            labels = []

            # Standardize annotation coordinates to tile and crop to tile boundaries
            for i in anns_in_tile:
                ann = annotations[i]
                ax0, ay0, aw, ah = ann["bbox"]
                rel_x = ax0 - x0
                rel_y = ay0 - y0
                crop_x = max(0, rel_x)
                crop_y = max(0, rel_y)
                crop_w = min(aw, w - crop_x, aw - max(0, -rel_x))
                crop_h = min(ah, h - crop_y, ah - max(0, -rel_y))

                if crop_w > 0 and crop_h > 0:
                    bboxes.append([crop_x, crop_y, crop_w, crop_h])
                    uncropped_bboxes.append([rel_x, rel_y, aw, ah])
                    labels.append(ann["label"])

            # Save bboxes and labels as numpy arrays, or empty arrays if no annotations
            bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32)
            uncropped_bboxes = (
                np.array(uncropped_bboxes, dtype=np.float32)
                if uncropped_bboxes
                else np.zeros((0, 4), dtype=np.float32)
            )
            labels = np.array(labels, dtype=np.int32) if labels else np.array([], dtype=np.int32)

            # Create a group for this tile and save the image and annotations
            group = h5f.create_group(f"tile_{x0}_{y0}")
            image = ndpi_data.get_tile(*tile, magnification=magnification)
            group.create_dataset("data", data=image, compression="gzip")
            group.create_dataset("bboxes", data=bboxes, compression="gzip")
            group.create_dataset("uncropped_bboxes", data=uncropped_bboxes, compression="gzip")
            group.create_dataset("labels", data=labels, compression="gzip")
            group.attrs["x"] = x0
            group.attrs["y"] = y0
            group.attrs["w"] = w
            group.attrs["h"] = h
            group.attrs["num_annotations"] = len(labels)

        # Save image-level attributes
        h5f.attrs["source_image"] = ndpi_data.ndpi_path
        h5f.attrs["num_tiles"] = len(tile_annotations)
        h5f.attrs["num_focal_planes"] = len(ndpi_data.get_z_offsets())
        h5f.attrs["image_width"] = ndpi_data.metadata.full_width
        h5f.attrs["image_height"] = ndpi_data.metadata.full_height
        h5f.attrs["tile_size"] = tile_size
        h5f.attrs["overlap"] = overlap
        h5f.attrs["magnification"] = magnification
        h5f.attrs["rois"] = np.array(rois, dtype=np.float32) if rois else np.zeros((0, 4), dtype=np.float32)


def build_label_maps(annotation_map_path: str) -> tuple[dict, dict]:
    """Build specimen-to-index and category-to-index label maps from CSV."""
    annotation_map = pd.read_csv(annotation_map_path)
    specimen_to_category = {
        str(row["Specimen_name"]).lower(): str(row["Category"])
        for _, row in annotation_map.iterrows()
    }
    categories = sorted(set(specimen_to_category.values()))
    category_to_index = {cat: idx for idx, cat in enumerate(categories)}
    specimen_to_index = {
        label: category_to_index[cat]
        for label, cat in specimen_to_category.items()
        if cat in category_to_index
    }
    return specimen_to_index, category_to_index


def generate_tiles_for_dir(
    input_dir: str,
    output_dir: str,
    annotation_map_path: str,
    magnification: float,
    tile_size: int,
    overlap: float,
) -> dict:
    """Process NDPI/NDPA pairs in input_dir and write H5 tile files.

    Returns:
        Metadata dictionary written to output_dir/metadata.json.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("--overlap must be in the range [0, 1).")
    if tile_size <= 0:
        raise ValueError("--tile_size must be > 0.")

    specimen_to_index, category_to_index = build_label_maps(annotation_map_path)

    os.makedirs(output_dir, exist_ok=True)
    ndpi_files = sorted(file for file in os.listdir(input_dir) if file.endswith(".ndpi"))
    for i, file in enumerate(ndpi_files):
        ndpi_path = os.path.join(input_dir, file)
        ndpa_path = ndpi_path + ".ndpa"
        if os.path.exists(ndpa_path):
            print(f"Processing {file.strip('.ndpi')} ({i+1}/{len(ndpi_files)})")
            ndpi_data = NDPIData(ndpi_path)
            ndpa_data = NDPAData(ndpa_path)
            generate_tiles(
                ndpi_data,
                ndpa_data,
                magnification=magnification,
                tile_size=tile_size,
                overlap=overlap,
                output_dir=output_dir,
                label_map=specimen_to_index,
            )
        else:
            print(f"Warning: No corresponding .ndpa file found for {ndpi_path}, skipping.")

    metadata = {
        "input_dir": input_dir,
        "magnification": magnification,
        "tile_size": tile_size,
        "overlap": overlap,
        "label_map": category_to_index,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for tile generation."""
    parser = argparse.ArgumentParser(
        description="Generate H5 tile datasets from NDPI/NDPA pairs."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .ndpi files (and matching .ndpi.ndpa files).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where output .h5 files and metadata.json are written.",
    )
    parser.add_argument(
        "--annotation_map_path",
        type=str,
        required=True,
        help="Path to annotation category CSV used to map specimen labels to class ids.",
    )
    parser.add_argument(
        "--magnification",
        type=float,
        default=40.0,
        help="Magnification level used for tile extraction (default: 40).",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Tile size in pixels (default: 1024).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Tile overlap fraction in [0, 1) (default: 0.0).",
    )
    return parser.parse_args()


def main():
    """Process all NDPI/NDPA pairs in input_dir and write H5 tile files."""
    args = parse_args()

    generate_tiles_for_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        annotation_map_path=args.annotation_map_path,
        magnification=args.magnification,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

if __name__ == "__main__":
    main()