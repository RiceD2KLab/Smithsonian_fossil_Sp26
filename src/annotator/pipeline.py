import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from src.annotator.detectors import MODEL_CONFIGS, ModelConfig, TileDetector, build_detector
from src.annotator.nms import TileSpec, deduplicate_tile_boundaries
from src.data.ndpa_writer import NDPAWriter
from src.data.ndpi_reader import NDPIData
from src.preprocessing.postprocess_tiles import focus_stack

## TO REMOVE:
def best_focal_plane(tile_4d, k):
    return tile_4d[:, :, :, k // 2]  # Placeholder: just take the middle plane for now.

@dataclass
class AnnotatorConfig:
    """Configuration for end-to-end NDPI annotation."""

    ndpi_path: str
    output_dir: str
    model_name: str
    checkpoint_path: str
    overlap: float
    magnification: float
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.5
    compression_method: str = "focus_stack"
    focus_stack_kernel_size: int = 5
    annotation_class: str = "paly"

    @property
    def model_key(self) -> str:
        return self.model_name.lower().strip()

    @property
    def model_config(self) -> ModelConfig:
        return MODEL_CONFIGS[self.model_key]

    @property
    def tile_size(self) -> int:
        return self.model_config.tile_size

    def validate(self) -> None:
        if not os.path.isfile(self.ndpi_path):
            raise FileNotFoundError(f"NDPI file not found: {self.ndpi_path}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        _ = self.model_key
        if not self.output_dir:
            raise ValueError("output_dir must be a non-empty path.")
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError("overlap must be in [0.0, 1.0).")
        if self.magnification <= 0:
            raise ValueError("magnification must be > 0.")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0.0, 1.0].")
        if not (0.0 <= self.nms_iou_threshold <= 1.0):
            raise ValueError("nms_iou_threshold must be in [0.0, 1.0].")
        if self.compression_method not in {"focus_stack", "best_focal_plane"}:
            raise ValueError("compression_method must be one of: focus_stack, best_focal_plane.")

@dataclass
class Detection:
    """Single detection in both pixel and physical-coordinate spaces."""

    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float
    score: float
    x_nm: float
    y_nm: float
    width_nm: float
    height_nm: float

def resolve_compress_2d(
    method: str,
    kernel_size: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve a tile-compression method string to a callable compress_2d(tile_4d)."""
    method_key = method.strip().lower()
    if method_key == "focus_stack":
        def compress_2d(tile_4d: np.ndarray) -> np.ndarray:
            stacked = focus_stack(tile_4d, k=kernel_size)
            image = stacked[0] if isinstance(stacked, tuple) else stacked
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return image

        return compress_2d

    if method_key == "best_focal_plane":
        def compress_2d(tile_4d: np.ndarray) -> np.ndarray:
            plane = best_focal_plane(tile_4d, k=kernel_size)
            image = plane[0] if isinstance(plane, tuple) else plane
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return image

        return compress_2d

    raise ValueError("Unsupported compression method. Choose 'focus_stack' or 'best_focal_plane'.")

class NDPIAnnotator:
    """Pipeline that annotates a full NDPI slide and writes merged CSV and NDPA output."""

    def __init__(
        self,
        config: AnnotatorConfig,
        compress_2d_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.model_key = config.model_key
        self.model_config = config.model_config
        self.ndpi = NDPIData(config.ndpi_path)
        self.compress_2d = compress_2d_fn or resolve_compress_2d(
            method=config.compression_method,
            kernel_size=config.focus_stack_kernel_size,
        )
        self.detector: TileDetector = build_detector(
            model_name=self.model_key,
            checkpoint_path=config.checkpoint_path,
            confidence_threshold=config.confidence_threshold,
            tile_size=self.model_config.tile_size,
        )

        # Conversion factors for pixel to physical space based on NDPI metadata 
        # and config magnification.
        self._nm_per_px_40x = self.ndpi.metadata.mpp_x * 1000.0
        self._nm_per_px = self._nm_per_px_40x * (
            self.ndpi.metadata.objective_power / self.config.magnification
        )
        self._origin_x_nm = self.ndpi.metadata.x_offset_nm - (
            self.ndpi.metadata.full_width / 2.0
        ) * self._nm_per_px_40x
        self._origin_y_nm = self.ndpi.metadata.y_offset_nm - (
            self.ndpi.metadata.full_height / 2.0
        ) * self._nm_per_px_40x

        self._tile_stride = max(1, int(round(self.config.tile_size * (1.0 - self.config.overlap))))
        # Number of tile-index steps that can still overlap at current stride.
        self._overlap_span = max(0, (self.config.tile_size + self._tile_stride - 1) // self._tile_stride)
        self.csv_output_path: str | None = None
        self.ndpa_output_path: str | None = None

    def _image_size_at_magnification(self) -> tuple[int, int]:
        matched = any(
            abs(fp.magnification - self.config.magnification) < 1e-6
            for fp in self.ndpi.focal_planes
        )
        if not matched:
            available = sorted({fp.magnification for fp in self.ndpi.focal_planes})
            raise ValueError(
                f"Magnification {self.config.magnification}x is unavailable for this NDPI. "
                f"Available magnifications: {available}"
            )

        scale = self.config.magnification / self.ndpi.metadata.objective_power
        width = int(round(self.ndpi.metadata.full_width * scale))
        height = int(round(self.ndpi.metadata.full_height * scale))
        return width, height

    @staticmethod
    def _axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
        if length <= tile_size:
            return [0]

        positions = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _build_tile_grid(self) -> list[TileSpec]:
        image_w, image_h = self._image_size_at_magnification()
        tile_size = self.config.tile_size
        stride = self._tile_stride

        xs = self._axis_positions(image_w, tile_size, stride)
        ys = self._axis_positions(image_h, tile_size, stride)
        tiles: list[TileSpec] = []
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                tiles.append(TileSpec(x=x, y=y, w=tile_size, h=tile_size, ix=ix, iy=iy))
        return tiles

    def _to_nm_bbox(self, x1_px: float, y1_px: float, x2_px: float, y2_px: float) -> tuple[float, float, float, float]:
        x_nm = x1_px * self._nm_per_px + self._origin_x_nm
        y_nm = y1_px * self._nm_per_px + self._origin_y_nm
        width_nm = max(x2_px - x1_px, 0.0) * self._nm_per_px
        height_nm = max(y2_px - y1_px, 0.0) * self._nm_per_px
        return x_nm, y_nm, width_nm, height_nm

    def _write_csv(self, detections: list[Detection], output_csv_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        fieldnames = self._csv_fieldnames()

        with open(output_csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for det in detections:
                writer.writerow(
                    {
                        "class": self.config.annotation_class,
                        "confidence": float(det.score),
                        "x_nm": float(det.x_nm),
                        "y_nm": float(det.y_nm),
                        "width_nm": float(det.width_nm),
                        "height_nm": float(det.height_nm),
                        "x1_px": float(det.x1_px),
                        "y1_px": float(det.y1_px),
                        "x2_px": float(det.x2_px),
                        "y2_px": float(det.y2_px),
                    }
                )

    @staticmethod
    def _csv_fieldnames() -> list[str]:
        return [
            "class",
            "confidence",
            "x_nm",
            "y_nm",
            "width_nm",
            "height_nm",
            "x1_px",
            "y1_px",
            "x2_px",
            "y2_px",
        ]

    def _write_ndpa(self, detections: list[Detection], output_ndpa_path: str) -> None:
        writer = NDPAWriter(output_path=output_ndpa_path)
        for det in detections:
            details = f"source={self.model_key}; confidence={float(det.score):.6f}"
            writer.add_circle(
                label=self.config.annotation_class,
                lens=self.config.magnification,
                x_nm=det.x_nm,
                y_nm=det.y_nm,
                width_nm=det.width_nm,
                height_nm=det.height_nm,
                color=self.model_config.ndpa_color,
                details=details,
            )
        writer.save()

    def run(self) -> list[Detection]:
        """Execute the full NDPI annotation pipeline and return merged detections."""
        
        # Get name of image (minus extension)
        image_name = Path(self.config.ndpi_path).name.strip(".ndpi")
        
        # Get NDPA and CSV output paths ready
        output_dir = os.path.abspath(self.config.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{image_name}.csv")
        ndpa_path = os.path.join(output_dir, f"{image_name}.ndpa")
        
        # Build the tile grid
        tiles = self._build_tile_grid()
        
        # Key data structures for merging:
        tiles_by_key: dict[tuple[int, int], TileSpec] = {}
        detections_by_tile: dict[tuple[int, int], list[Detection]] = {}

        # Write checkpoint rows as tiles are processed; final CSV is rewritten after deduplication.
        with open(csv_path, "w", newline="") as checkpoint_file:
            
            checkpoint_writer = csv.DictWriter(checkpoint_file, fieldnames=self._csv_fieldnames())
            checkpoint_writer.writeheader()

            for tile in tqdm(tiles, desc="Annotating tiles", unit="tile"):
                
                # Store tile spec for merging later.
                key = (tile.iy, tile.ix)
                tiles_by_key[key] = tile

                # Read tile, compress to 2D, and run detector.
                x, y, w, h = tile.x, tile.y, tile.w, tile.h
                tile_3d = self.ndpi.get_tile(x=x, y=y, w=w, h=h, magnification=self.config.magnification)
                tile_2d = self.compress_2d(tile_3d)
                boxes, scores = self.detector.predict(tile_2d)
                
                # If no detections, record empty list and continue.
                if len(boxes) == 0:
                    detections_by_tile[key] = []
                    continue

                tile_candidates: list[Detection] = []
                for box, score in zip(boxes, scores):

                    # Convert box coordinates from tile-local to global pixel space.
                    gx1 = float(x + box[0])
                    gy1 = float(y + box[1])
                    gx2 = float(x + box[2])
                    gy2 = float(y + box[3])

                    # Convert global pixel coordinates to physical space.
                    x_nm, y_nm, width_nm, height_nm = self._to_nm_bbox(gx1, gy1, gx2, gy2)
                    
                    # Create Detection object and add to tile candidates.
                    tile_candidates.append(
                        Detection(
                            x1_px=gx1,
                            y1_px=gy1,
                            x2_px=gx2,
                            y2_px=gy2,
                            score=float(score),
                            x_nm=x_nm,
                            y_nm=y_nm,
                            width_nm=width_nm,
                            height_nm=height_nm,
                        )
                    )
                
                # Store candidates for this tile for later merging.
                detections_by_tile[key] = tile_candidates

                # Optionally write tile candidates to checkpoint CSV immediately for checkpointing; will be rewritten after deduplication.
                if tile_candidates:
                    checkpoint_writer.writerows(
                        [self._detection_to_csv_row(det) for det in tile_candidates]
                    )
                    checkpoint_file.flush()

        final_detections = deduplicate_tile_boundaries(
            tiles_by_key=tiles_by_key,
            detections_by_tile=detections_by_tile,
            iou_threshold=self.config.nms_iou_threshold,
            overlap_span=self._overlap_span,
        )

        # Replace checkpoint CSV with final deduplicated results.
        self._write_csv(final_detections, csv_path)
        self._write_ndpa(final_detections, ndpa_path)
        self.csv_output_path = csv_path
        self.ndpa_output_path = ndpa_path
        return final_detections
