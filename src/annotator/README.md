# NDPI Annotator Pipeline

This module implements a full end-to-end annotation pipeline for NDPI slides.

Given an NDPI file, the pipeline:
1. Opens the NDPI using `NDPIData` from `src/data/ndpi_reader.py`.
2. Loads one model family: YOLO or RF-DETR.
3. Generates a full tile grid over the entire image bounds (width and height) at the requested magnification.
4. Focus-stacks each tile from `W x H x C x Z` to `W x H x C` using `focus_stack` from `src/preprocessing/focus_stack.py`.
5. Runs tile-level detection and converts tile outputs back to global coordinates and nanometer-space coordinates.
6. Merges detections from overlapping tiles with non-max suppression (NMS).
7. Writes final merged annotations to both CSV and NDPA files.

All output rows use class label `paly` by default.

## Files Added

- `src/annotator/pipeline.py`
  - `AnnotatorConfig`: validated pipeline configuration.
  - `NDPIAnnotator`: full orchestration class.
- `src/annotator/models.py`
  - `YOLOTileDetector` and `RFDETRTileDetector` adapters.
  - `build_detector` factory.
- `src/annotator/nms.py`
  - IoU and class-agnostic NMS utilities.
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

Tile sizes are enforced by model family:
- YOLO: `1024`
- RF-DETR: `1008`

### 3) Full-Slide Tiling Across Entire Bounds

The tile grid is computed from metadata-scaled slide dimensions at requested magnification:
- `scale = magnification / objective_power`
- `width_at_mag = round(full_width * scale)`
- `height_at_mag = round(full_height * scale)`

Stride is derived from overlap:
- `stride = round(tile_size * (1 - overlap))`

The grid always includes trailing positions so the final tiles reach the rightmost and bottom-most edges.

### 4) Focus Stacking

Each tile is read as a 4D stack (`H x W x C x Z`) and focus-stacked via:
- `src.preprocessing.focus_stack.focus_stack(tile, k=focus_stack_kernel_size)`

The pipeline accepts either:
- return value as stacked image, or
- tuple return where stacked image is first element.

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

### 6) Cross-Tile Merge (NMS)

After all tiles are processed, detections are merged globally with score-sorted class-agnostic NMS:
- IoU threshold configured by `nms_iou_threshold`.

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
- `details` empty
- `coordformat` = `nanometers`
- `lens` = magnification
- `x`, `y` = bbox center in nanometers
- `z` = `0`
- Rectangle corners stored under `annotation/pointlist/point`

## CLI Usage

Run via module:

```bash
python -m src.annotator \
  --ndpi_path /path/to/slide.ndpi \
  --output_dir /path/to/outputs \
  --model_name yolo \
  --checkpoint_path /path/to/yolo.pt \
  --overlap 0.10 \
  --magnification 20
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
  --rfdetr_variant base
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
- `--confidence_threshold` (default `0.25`)
- `--nms_iou_threshold` (default `0.5`)
- `--annotation_class` (default `paly`)
- `--focus_stack_kernel_size` (default `5`)
- `--device` (optional)

Model-specific optional:
- YOLO: `--yolo_imgsz`
- RF-DETR: `--rfdetr_variant` in `{nano, small, base, large}`

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
