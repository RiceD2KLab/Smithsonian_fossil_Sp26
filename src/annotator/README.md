# NDPI Annotator Pipeline

This module implements a full end-to-end annotation pipeline for NDPI slides.

Given an NDPI file, the pipeline:
1. Opens the NDPI using `NDPIData` from `src/data/ndpi_reader.py`.
2. Loads one model family: YOLO or RF-DETR from a fixed model-config registry.
3. Generates a full tile grid over the entire image bounds (width and height) at the requested magnification.
4. Compresses each tile from `W x H x C x Z` to `W x H x C` using a selected method.
5. Runs tile-level detection and converts tile outputs back to global coordinates and nanometer-space coordinates.
6. Merges detections from overlapping tiles using overlap-aware deduplication (neighbor-tile IoU suppression).
7. Writes final merged annotations to both CSV and NDPA files.

All output rows use class label `paly` by default.

## Files

- `src/annotator/config.py`
  - `AnnotatorConfig`: validated pipeline configuration.
- `src/annotator/compression.py`
  - `focus_stack_2d`, `best_focal_plane`, and `resolve_compress_2d`.
- `src/annotator/pipeline.py`
  - `NDPIAnnotator`: full orchestration class.
- `src/annotator/detectors.py`
  - `YOLOTileDetector` and `RFDETRTileDetector` adapters.
  - `build_detector` factory.
- `src/annotator/nms.py`
  - Tile-boundary deduplication utilities.
- `src/annotator/types.py`
  - `Detection` dataclass for normalized in-memory output.
- `src/annotator/run.py`
  - CLI entrypoint and argument parsing.
- `src/annotator/__main__.py`
  - Allows `python -m src.annotator ...`.
- `src/annotator/__init__.py`
  - Public exports.
- `src/data/ndpa_writer.py`
  - `NDPAWriter` for NDPA XML output.

## Pipeline Behavior

### 1) NDPI Loading

The annotator initializes `NDPIData(ndpi_path)` and reads metadata (`NDPIMetadata`) including:
- `mpp_x`, `mpp_y`
- `objective_power`
- `x_offset_nm`, `y_offset_nm`
- `full_width`, `full_height`

### 2) Model Loading

Model adapters are abstracted behind one interface:
- `TileDetector.predict(tile_rgb) -> (boxes_xyxy, scores)`

Supported model names:
- `yolo`
- `rfdetr`
- `rf-detr` (CLI alias)

Each model key maps to a constant config used throughout the pipeline.

Current model config map:
- `yolo`: `tile_size=1024`, NDPA color=`#ff0000` (red)
- `rfdetr`: `tile_size=1008`, NDPA color=`#00ff00` (green)

YOLO inference size is tied to tile size (no separate `yolo_imgsz` parameter).
RF-DETR uses the fixed `base` variant in this pipeline.

### 3) Full-Slide Tiling Across Entire Bounds

The tile grid is computed from metadata-scaled slide dimensions at requested magnification:
- `scale = magnification / objective_power`
- `width_at_mag = round(full_width * scale)`
- `height_at_mag = round(full_height * scale)`

Stride is derived from overlap:
- `stride = round(tile_size * (1 - overlap))`

The grid always includes trailing positions so the final tiles reach the rightmost and bottom-most edges.

### 4) Tile Compression (4D -> 2D)

Each tile is read as a 4D stack (`H x W x C x Z`) and compressed to 3D (`H x W x C`) using a resolved `compress_2d(tile_4d)` function selected by `compression_method`.

Supported methods:
- `focus_stack`: uses `src.preprocessing.postprocess_tiles.focus_stack(tile, k=focus_stack_kernel_size)`
- `best_focal_plane`: selects the top Tenengrad-ranked focal plane from `src.annotator.compression.best_focal_plane(tile, tenengrad_ksize=tenengrad_ksize)`

Both methods are normalized to return a `uint8` 3D tile.

### 5) Inference + Coordinate Conversion

For each focus-stacked tile:
- Run detector and collect `xyxy` tile-local boxes and confidence.
- Clip boxes to tile bounds.
- Shift tile-local boxes to global image pixel coordinates by tile origin `(tile_x, tile_y)`.
- Convert global pixel boxes into nanometer-space using NDPI metadata.

Nanometer conversion constants:
- `nm_per_px_40x = mpp_x * 1000`
- `nm_per_px = nm_per_px_40x * (objective_power / magnification)`
- `origin_x_nm = x_offset_nm - (full_width / 2) * nm_per_px_40x`
- `origin_y_nm = y_offset_nm - (full_height / 2) * nm_per_px_40x`

Bounding box conversion:
- `x_nm = x1_px * nm_per_px + origin_x_nm`
- `y_nm = y1_px * nm_per_px + origin_y_nm`
- `width_nm = (x2_px - x1_px) * nm_per_px`
- `height_nm = (y2_px - y1_px) * nm_per_px`

### 6) Cross-Tile Merge (Boundary-Pair End Pass)

Detections are first stored per tile in a tile-index map keyed as `(row, col)` where:
- `(0, 0)` is top-left
- `(0, 1)` is one tile to the right
- `(1, 0)` is one tile below

After all tiles are processed, deduplication runs only across overlapping tile pairs (tile boundaries), not globally across the whole slide.

Behavior:
- Candidate comparisons are limited to overlap-neighbor tile pairs (including diagonal overlap pairs when applicable).
- A fast rectangle-intersection check runs before IoU.
- Pairs with IoU above `nms_iou_threshold` are grouped as duplicate clusters.
- In each duplicate cluster, the highest-confidence detection is kept.

This handles multi-tile overlap points (3+ tiles) robustly because duplicates are resolved by connected components rather than isolated pair decisions.

### 7) CSV + NDPA Export

Outputs are written into `output_dir` using NDPI-based naming:
- CSV: `x.csv`
- NDPA: `x.ndpi.ndpa`

where the input NDPI filename is `x.ndpi`.

CSV columns:
- `class`
- `confidence`
- `x_nm`, `y_nm`, `width_nm`, `height_nm`
- `x1_px`, `y1_px`, `x2_px`, `y2_px`

`class` is set from `annotation_class` (default `paly`).

NDPA format:
- Root element: `annotations`
- One `ndpviewstate` per detection
- `title` = class label
- `details` contains a source note (for example: `source=model; model=yolo`)
- `coordformat` = `nanometers`
- `lens` = magnification
- `x`, `y` = bbox center in nanometers
- `z` = `0`
- Circle center/radius stored under `annotation/x`, `annotation/y`, and `annotation/radius`
- `annotation/@color` is model-specific (YOLO red, RF-DETR green)
- `details` stores per-annotation metadata in the form `source=<model>; confidence=<score>`

If the NDPA file already exists, new annotations are appended. If it does not exist, an empty NDPA file is created first. This allows running YOLO and RF-DETR sequentially into the same NDPA file.

## CLI Usage

Run via module:

```bash
python -m src.annotator \
  --ndpi_path /path/to/slide.ndpi \
  --output_dir /path/to/outputs \
  --model_name yolo \
  --checkpoint_path /path/to/yolo.pt \
  --overlap 0.10 \
  --magnification 20 \
  --compression_method focus_stack
```

RF-DETR example:

```bash
python -m src.annotator \
  --ndpi_path /path/to/slide.ndpi \
  --output_dir /path/to/outputs \
  --model_name rf-detr \
  --checkpoint_path /path/to/rfdetr.pt \
  --overlap 0.10 \
  --magnification 20 \
  --compression_method best_focal_plane
```

## Parameters

Required:
- `--ndpi_path`
- `--output_dir`
- `--model_name`
- `--checkpoint_path`
- `--overlap`
- `--magnification`

Common optional:
- `--confidence_threshold` (default `0.5`)
- `--nms_iou_threshold` (default `0.5`)
- `--compression_method` in `{focus_stack, best_focal_plane}`
- `--annotation_class` (default `paly`)
- `--focus_stack_kernel_size` (default `5`)
- `--tenengrad_ksize` (default `3`)

## Programmatic Usage

```python
from src.annotator import AnnotatorConfig, NDPIAnnotator

config = AnnotatorConfig(
    ndpi_path="/path/to/slide.ndpi",
    output_dir="/path/to/outputs",
    model_name="yolo",
    checkpoint_path="/path/to/model.pt",
    overlap=0.1,
    magnification=20,
)

annotator = NDPIAnnotator(config)
final_detections = annotator.run()
print(len(final_detections))
```

## Validation and Error Handling

The config validates:
- NDPI path exists.
- Checkpoint path exists.
- `overlap` in `[0.0, 1.0)`.
- `magnification > 0`.
- confidence and IoU thresholds in `[0.0, 1.0]`.

The pipeline also validates that requested magnification exists in NDPI focal plane metadata.

## Notes

- RF-DETR adapter attempts ndarray inference first; if not supported by backend, it falls back to temporary file inference.
- NMS is class-agnostic by design because this pipeline currently exports one semantic class (`paly`).
