# Annotator Pipeline

This package implements the end-to-end NDPI annotation workflow.

It reads an NDPI slide, selects the requested magnification, compresses each tile stack into a 2D image, runs tile-level detection, merges duplicate detections from overlapping tiles, and writes CSV and NDPA outputs.

## Package Layout

The annotator code is split across these modules:

- `src/annotator/config.py`: configuration objects and validation
- `src/annotator/compression.py`: tile compression helpers
- `src/annotator/detectors.py`: YOLO and RF-DETR detector adapters
- `src/annotator/nms.py`: cross-tile deduplication logic
- `src/annotator/pipeline.py`: end-to-end orchestration
- `src/annotator/run.py`: CLI entrypoint
- `src/annotator/__main__.py`: `python -m src.annotator` entrypoint

The pipeline also depends on shared slide/data helpers in `src/data/`.

## Workflow

1. `NDPIData` loads slide metadata and focal-plane information from the NDPI file.
2. The pipeline validates the requested magnification against the slide focal-plane metadata.
3. A tile grid is built from the slide size at the target magnification.
4. Each `H x W x C x Z` tile stack is compressed to `H x W x C`.
5. The selected detector runs on the 2D tile and returns `xyxy` boxes with scores.
6. Tile-local boxes are shifted into global pixel space.
7. Global pixel boxes are converted into nanometer coordinates.
8. Detections from overlapping tiles are merged with boundary-aware deduplication.
9. Final detections are written to CSV and NDPA files.

## Supported Models

The current CLI accepts these model families:

- `yolo`
- `rfdetr`

Each model family has a fixed tile size and NDPA color in the detector registry.

## Compression Methods

The current compression modes are:

- `focus_stack`: focus-stack the `Z` dimension into a single RGB image
- `best_focal_plane`: choose the Tenengrad-ranked focal plane

Related CLI parameters:

- `--compression_method`
- `--focus_stack_kernel_size`
- `--tenengrad_ksize`

## Coordinate Conversion

The pipeline uses NDPI metadata to convert between pixel and nanometer coordinates.

For a box in pixel coordinates:

- `nm_per_px_40x = mpp_x * 1000`
- `nm_per_px = nm_per_px_40x * (objective_power / magnification)`
- `origin_x_nm = x_offset_nm - (full_width / 2) * nm_per_px_40x`
- `origin_y_nm = y_offset_nm - (full_height / 2) * nm_per_px_40x`

Pixel-to-nanometer conversion is handled by `pixels_to_nm_bbox(...)` in `src/data/util.py`.

## Deduplication

The pipeline performs overlap-aware deduplication after all tiles are processed.

It does not deduplicate globally across the whole slide. Instead, it compares detections only across overlapping neighboring tiles and keeps the highest-confidence member of each duplicate cluster.

## Outputs

The pipeline writes two files into `output_dir`:

- `<slide_name>.csv`
- `<slide_name>.ndpi.ndpa`

CSV columns:

- `class`
- `confidence`
- `x_nm`
- `y_nm`
- `width_nm`
- `height_nm`
- `x1_px`
- `y1_px`
- `x2_px`
- `y2_px`

The `class` column defaults to `paly` unless overridden with `--annotation_class`.

NDPA annotations are written as circles with per-annotation details in the form:

- `source=<model name>; confidence=<score>`

## CLI Usage

Run the package with:

```bash
python -m src.annotator \
  --ndpi_path /path/to/slide.ndpi \
  --output_dir /path/to/outputs \
  --model_name yolo \
  --checkpoint_path /path/to/checkpoint.pt \
  --overlap 0.10 \
  --magnification 20 \
  --compression_method focus_stack
```

RF-DETR example:

```bash
python -m src.annotator \
  --ndpi_path /path/to/slide.ndpi \
  --output_dir /path/to/outputs \
  --model_name rfdetr \
  --checkpoint_path /path/to/checkpoint.pt \
  --overlap 0.10 \
  --magnification 20 \
  --compression_method best_focal_plane
```

### Required arguments

- `--ndpi_path`
- `--output_dir`
- `--model_name`
- `--checkpoint_path`
- `--overlap`
- `--magnification`

### Optional arguments

- `--confidence_threshold` default `0.5`
- `--nms_iou_threshold` default `0.5`
- `--compression_method` default `focus_stack`
- `--focus_stack_kernel_size` default `5`
- `--tenengrad_ksize` default `3`
- `--annotation_class` default `paly`

## Programmatic Use

The package `__init__.py` is currently empty, so import from the concrete modules directly:

```python
from src.annotator.config import AnnotatorConfig
from src.annotator.pipeline import NDPIAnnotator

config = AnnotatorConfig(
    ndpi_path="/path/to/slide.ndpi",
    output_dir="/path/to/outputs",
    model_name="yolo",
    checkpoint_path="/path/to/checkpoint.pt",
    overlap=0.1,
    magnification=20,
)

annotator = NDPIAnnotator(config)
detections = annotator.run()
print(len(detections))
```

## Validation

The configuration validation currently checks:

- NDPI path exists
- checkpoint path exists
- `output_dir` is not empty
- `overlap` is in `[0.0, 1.0)`
- `magnification > 0`
- confidence and IoU thresholds are in `[0.0, 1.0]`
- compression method is one of `focus_stack` or `best_focal_plane`
- `focus_stack_kernel_size` is one of `1, 3, 5, 7`
- `tenengrad_ksize` is one of `1, 3, 5, 7`

## Notes

- The pipeline currently uses a shared `NDPIData` reader for slide metadata and tile extraction.
- RF-DETR is wired through the detector adapter in `src/annotator/detectors.py`.
- The README reflects the code as it exists now, not an idealized API surface.
