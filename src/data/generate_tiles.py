from ndpa_reader import NDPAData
from ndpi_reader import NDPIData
from util import bounds_to_pixels
import os
import numpy as np    
import json

"""
This file provides funtionality to load NDPI and NDPA files, 
generate tiles from the NDPI image based on the ROIs defined in the 
NDPA annotations, and save the tiles along with their corresponding
annotations to disk. The main function iterates through all NDPI 
files in a specified input directory, finds the corresponding NDPA file, 
and processes them.
"""

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
        ):
    """
    Generate tiles from a single NDPI file covering all ROIs in
    the associated NDPA file, and save them to disk along with 
    their annotations.
    """

    # Generate tile grid covering all ROIs
    tiles = []
    for region in ndpa_data.rois:
        bounds = bounds_to_pixels(region.bounds, magnification, ndpi_data.metadata)
        tiles.extend(compute_tile_grid(*bounds, tile_size=tile_size, overlap=overlap))

    print(f"Generated {len(tiles)} tiles.")

    # Convert annotations to pixels
    annotations = []
    for ann in ndpa_data.palynomorphs:
        annotations.append(bounds_to_pixels(ann.bounds, magnification, ndpi_data.metadata))

    # For each tile, find annotations that intersect it
    tile_annotations = []  # List of (tile, [annotation_indices])
    for tile in tiles:
        x0, y0, w, h = tile
        x1, y1 = x0 + w, y0 + h
        anns_in_tile = []
        for i, ann_bbox in enumerate(annotations):
            ax0, ay0, aw, ah = ann_bbox
            ax1, ay1 = ax0 + aw, ay0 + ah
            # Check for intersection
            if not (ax1 <= x0 or ax0 >= x1 or ay1 <= y0 or ay0 >= y1):
                anns_in_tile.append(i)
        tile_annotations.append((tile, anns_in_tile))
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    tiles_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    tile_to_annotations = {}

    # For each tile:
    for tile, anns_in_tile in tile_annotations:

        # Get the tile image from the NDPI file and save it to disk
        image = ndpi_data.get_tile(*tile, magnification=magnification)
        tile_filename = f"tile_{tile[0]}_{tile[1]}_{tile[2]}_{tile[3]}.npy"
        tile_path = os.path.join("tiles", tile_filename)
        np.save(os.path.join(tiles_dir, tile_filename), image)

        # Standardize annotation coordinates to tile and crop
        x0, y0, w, h = tile
        tile_ann_boxes = []
        for i in anns_in_tile:
            ax0, ay0, aw, ah = annotations[i]
            rel_x = ax0 - x0
            rel_y = ay0 - y0
            crop_x = max(0, rel_x)
            crop_y = max(0, rel_y)
            crop_w = min(aw, w - crop_x, aw - max(0, -rel_x))
            crop_h = min(ah, h - crop_y, ah - max(0, -rel_y))
            if crop_w > 0 and crop_h > 0:
                tile_ann_boxes.append([crop_x, crop_y, crop_w, crop_h])
        tile_to_annotations[tile_path] = tile_ann_boxes

    # Save the mapping to a JSON file in output_dir
    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(tile_to_annotations, f)

def main():
    """
    Main function to process all NDPI/NDPA files in the input directory.
    """
    INPUT_DIR = "<path_to_ndpi_files>"
    OUTPUT_DIR = "output/tiled_images"
    MAGNIFICATION = 40
    TILE_SIZE = 1024
    OVERLAP = 0.0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".ndpi"):
            ndpi_path = os.path.join(INPUT_DIR, file)
            ndpa_path = ndpi_path + ".ndpa"
            if os.path.exists(ndpa_path):
                print(f"Processing {file} with annotations from {file}.ndpa")
                ndpi_data = NDPIData(ndpi_path)
                ndpa_data = NDPAData(ndpa_path)
                output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(file)[0])
                generate_tiles(ndpi_data, ndpa_data, magnification=MAGNIFICATION, tile_size=TILE_SIZE, overlap=OVERLAP, output_dir=output_dir)
            else:
                print(f"Warning: No corresponding .ndpa file found for {ndpi_path}, skipping.")

    # Write global metadata
    metadata = {
        "input_dir": INPUT_DIR,
        "magnification": MAGNIFICATION,
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()