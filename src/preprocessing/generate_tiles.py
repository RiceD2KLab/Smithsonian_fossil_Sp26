from src.data.ndpa_reader import NDPAData
from src.data.ndpi_reader import NDPIData
from src.data.util import bounds_to_pixels
import os
import numpy as np    
import json
import h5py
from tqdm import tqdm
import pandas as pd

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
            image = ndpi_data.get_tile(*tile, magnification=magnification)
            x0, y0, w, h = tile
            bboxes = []
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
                    labels.append(ann["label"])

            # Save bboxes and labels as numpy arrays, or empty arrays if no annotations
            bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(labels, dtype=np.int32) if labels else np.array([], dtype=np.int32)

            # Create a group for this tile and save the image and annotations
            group = h5f.create_group(f"tile_{x0}_{y0}")
            group.create_dataset("data", data=image, compression="gzip")
            group.create_dataset("bboxes", data=bboxes, compression="gzip")
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

def main():
    """
    Main function to process all NDPI/NDPA files in the input directory.
    """

    # Data paths
    INPUT_DIR = "/path/to/ndpi/files"
    OUTPUT_DIR = "/path/to/output/tiles"
    ANNOTATION_MAP_PATH = "/path/to/annotation_categories.csv"
    
    # Tiling parameters
    MAGNIFICATION = 40
    TILE_SIZE = 1024
    OVERLAP = 0.0

    # The annotation map defines how specimens in the NDPA files are mapped to broader categories.
    annotation_map = pd.read_csv(ANNOTATION_MAP_PATH)
    specimen_to_category = {str(row['Specimen_name']).lower(): str(row['Category']) for _, row in annotation_map.iterrows()}
    
    # Create a mapping from each category to an integer label for modeling
    categories = sorted(set(specimen_to_category.values()))
    category_to_index = {cat: idx for idx, cat in enumerate(categories)}

    # Create merged mapping: label -> index
    specimen_to_index = {label: category_to_index[cat] for label, cat in specimen_to_category.items() if cat in category_to_index}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ndpi_files = [file for file in os.listdir(INPUT_DIR) if file.endswith(".ndpi")]
    for i, file in enumerate(ndpi_files):
        ndpi_path = os.path.join(INPUT_DIR, file)
        ndpa_path = ndpi_path + ".ndpa"
        if os.path.exists(ndpa_path):
            print(f"Processing {file.strip('.ndpi')} ({i+1}/{len(ndpi_files)})")
            ndpi_data = NDPIData(ndpi_path)
            ndpa_data = NDPAData(ndpa_path)
            generate_tiles(
                ndpi_data,
                ndpa_data,
                magnification=MAGNIFICATION,
                tile_size=TILE_SIZE,
                overlap=OVERLAP,
                output_dir=OUTPUT_DIR,
                label_map=specimen_to_index
            )
        else:
            print(f"Warning: No corresponding .ndpa file found for {ndpi_path}, skipping.")

    # Write global metadata
    metadata = {
        "input_dir": INPUT_DIR,
        "magnification": MAGNIFICATION,
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "label_map": category_to_index # Save the category-to-index mapping for reference
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()