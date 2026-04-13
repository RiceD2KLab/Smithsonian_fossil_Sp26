# Preprocessing

This directory contains scripts for converting NDPI/NDPA source data into H5 tile datasets and exporting those tiles for model training.

## Files

- `generate_tiles.py`
  - Builds tile H5 files from NDPI images and NDPA annotations.
  - Writes 3D image stacks (`data`) plus annotation datasets (`bboxes`, `uncropped_bboxes`, `labels`) for each tile.
- `postprocess_tiles.py`
  - Post-processes existing tiles in-place.
  - Can generate one or more outputs per tile:
    - 2D focus-stacked image computed using the Laplacian of Gaussian.
    - Rankings of the focal planes using various metrics including the Laplacian of Gaussian (LoG), Variance of Laplacian (VoL), and Tenengrad gradient.
    - A maximum intensity projection of the the tile.
- `split_data.py`
  - Creates train/val/test split JSON from H5 slide files.
- `export_coco.py`
  - Exports H5 tiles into COCO JSON + image files.
- `h5_utils.py`
  - Shared helpers for listing H5 files and tile groups.

## H5 Structure

Each H5 file corresponds to one source slide.

### File-level attributes

- `source_image`: Source NDPI path.
- `num_tiles`: Number of tile groups.
- `num_focal_planes`: Number of z planes in `data`.
- `image_width`, `image_height`: Full slide dimensions.
- `tile_size`: Tile edge length in pixels.
- `overlap`: Tile overlap fraction.
- `magnification`: Extraction magnification.
- `rois`: ROI bounds used for tile generation, shape `(N, 4)`.

### Tile groups

Each tile group is named:
- `tile_<x>_<y>`

Tile attributes written by `generate_tiles.py`:
- `x`, `y`: Tile top-left in slide pixels.
- `w`, `h`: Tile dimensions in pixels.
- `num_annotations`: Number of annotations retained for that tile.

Tile datasets written by `generate_tiles.py`:
- `data`: Image stack, shape `(H, W, C, Z)`.
- `bboxes`: Cropped tile-relative boxes, shape `(N, 4)` with `[x, y, w, h]`.
- `uncropped_bboxes`: Uncropped tile-relative boxes, shape `(N, 4)` with `[x, y, w, h]` before clipping.
- `labels`: Class ids aligned with `bboxes`, shape `(N,)`.

Optional tile datasets written by `postprocess_tiles.py`:
- `focus_stacked`: LoG focus-stacked image, shape `(H, W, C)`.
- `focal_plane_ranking`: Legacy LoG ranking, best-to-worst, shape `(Z,)`.
- `focal_plane_ranking_LoG`: LoG ranking, shape `(Z,)`.
- `focal_plane_ranking_VoL`: VoL ranking, shape `(Z,)`.
- `focal_plane_ranking_tenengrad`: Tenengrad ranking, shape `(Z,)`.
- `maximum_intensity_projection`: MIP image, shape `(H, W, C)`.

## Typical Pipeline

1. Generate H5 tiles.
2. Add post-processing outputs (focus stack, rankings, MIP).
3. Split slides into train/val/test.
4. Export to COCO.

## Commands

### Generate tiles

Run with explicit CLI arguments:

```bash
python -m src.preprocessing.generate_tiles \
  --input_dir /path/to/ndpi_dir \
  --output_dir /path/to/output_h5_dir \
  --annotation_map_path /path/to/annotation_categories.csv \
  --magnification 40 \
  --tile_size 1024 \
  --overlap 0.0
```

Required:
- `--input_dir`: Directory containing `.ndpi` files and matching `.ndpi.ndpa` annotation files.
- `--output_dir`: Destination directory for output `.h5` files and `metadata.json`.
- `--annotation_map_path`: CSV mapping specimen names to categories.

Optional:
- `--magnification` (default: `40`)
- `--tile_size` (default: `1024`)
- `--overlap` in `[0, 1)` (default: `0.0`)

### Generate focus stack / rankings / MIP

Run on one H5 file or a directory of H5 files:

```bash
python -m src.preprocessing.postprocess_tiles --input_path /path/to/h5_or_dir --outputs focus_stacked rankings
```

Examples:

```bash
# Focus stacked only
python -m src.preprocessing.postprocess_tiles --input_path /path/to/h5_or_dir --outputs focus_stacked

# Rankings only
python -m src.preprocessing.postprocess_tiles --input_path /path/to/h5_or_dir --outputs rankings

# MIP only
python -m src.preprocessing.postprocess_tiles --input_path /path/to/h5_or_dir --outputs mip

# All supported outputs
python -m src.preprocessing.postprocess_tiles --input_path /path/to/h5_or_dir --outputs focus_stacked rankings mip
```

Optional tuning:

```bash
python -m src.preprocessing.postprocess_tiles \
  --input_path /path/to/h5_or_dir \
  --outputs focus_stacked rankings \
  --log_kernel_size 5 \
  --tenengrad_ksize 3
```

### Split slides

```bash
python -m src.preprocessing.split_data --input_dir /path/to/h5_dir --output_dir /path/to/splits --seed 67
```

### Export COCO

```bash
python -m src.preprocessing.export_coco \
  --h5_root /path/to/h5_or_dir \
  --output_dir /path/to/coco_out \
  --mode best_plane \
  --ranking_metric LoG \
  --workers 4
```

Example for top-k planes:

```bash
python -m src.preprocessing.export_coco \
  --h5_root /path/to/h5_or_dir \
  --output_dir /path/to/coco_out \
  --mode top_k_planes \
  --top_k 5 \
  --ranking_metric tenengrad
```