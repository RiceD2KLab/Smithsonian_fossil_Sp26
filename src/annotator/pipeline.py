"""End-to-end NDPI annotation pipeline."""

import csv
import os
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from src.annotator.compression import get_compression_func
from src.annotator.config import AnnotatorConfig
from src.annotator.detectors import TileDetector, build_detector
from src.annotator.nms import TileSpec, deduplicate_tile_boundaries
from src.annotator.types import Detection
from src.data.ndpa_writer import NDPAWriter
from src.data.ndpi_reader import NDPIData

class NDPIAnnotator:
    """Pipeline that annotates a full NDPI slide and writes merged CSV and NDPA output."""

    def __init__(
        self,
        config: AnnotatorConfig,
    ) -> None:
        """Initialize the pipeline with configuration, metadata, and detector state."""
        config.validate()
        self.config = config
        self.model_key = config.model_key
        self.model_config = config.model_config
        self.ndpi = NDPIData(config.ndpi_path)
        self.compress_2d = get_compression_func(
            method=config.compression_method,
            focus_stack_kernel_size=config.focus_stack_kernel_size,
            tenengrad_ksize=config.tenengrad_ksize,
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

    def _resolve_output_paths(self) -> tuple[str, str]:
        """Return the CSV and NDPA paths derived from the NDPI filename."""
        ndpi_name = Path(self.config.ndpi_path).name
        ndpi_stem = ndpi_name[: -len(".ndpi")] if ndpi_name.lower().endswith(".ndpi") else Path(ndpi_name).stem

        output_dir = os.path.abspath(self.config.output_dir)
        csv_path = os.path.join(output_dir, f"{ndpi_stem}.csv")
        ndpa_path = os.path.join(output_dir, f"{ndpi_name}.ndpa")
        return csv_path, ndpa_path

    def _image_size_at_magnification(self) -> tuple[int, int]:
        """Return the target image size at the requested magnification."""
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
        """Return tile origins that cover one image axis."""
        if length <= tile_size:
            return [0]

        positions = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _build_tile_grid(self) -> list[TileSpec]:
        """Build the tile grid that covers the slide at the target magnification."""
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
        """Convert global pixel bbox coordinates to NDPI nanometer space."""
        x_nm = x1_px * self._nm_per_px + self._origin_x_nm
        y_nm = y1_px * self._nm_per_px + self._origin_y_nm
        width_nm = max(x2_px - x1_px, 0.0) * self._nm_per_px
        height_nm = max(y2_px - y1_px, 0.0) * self._nm_per_px
        return x_nm, y_nm, width_nm, height_nm

    def _write_csv(self, detections: list[Detection], output_csv_path: str) -> None:
        """Write detections to a CSV file using the annotator export schema."""
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
        """Return the CSV field order used by the annotator."""
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

    def _detection_to_csv_row(self, det: Detection) -> dict[str, float | str]:
        """Convert one detection into a CSV row dictionary."""
        return {
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

    def _write_ndpa(self, detections: list[Detection], output_ndpa_path: str) -> None:
        """Write detections to an NDPA file using circle annotations."""
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

        csv_path, ndpa_path = self._resolve_output_paths()
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        tiles = self._build_tile_grid()
        tiles_by_key: dict[tuple[int, int], TileSpec] = {}
        detections_by_tile: dict[tuple[int, int], list[Detection]] = {}

        # Write checkpoint rows as tiles are processed; final CSV is rewritten after deduplication.
        with open(csv_path, "w", newline="") as checkpoint_file:
            checkpoint_writer = csv.DictWriter(checkpoint_file, fieldnames=self._csv_fieldnames())
            checkpoint_writer.writeheader()

            for tile in tqdm(tiles, desc="Annotating tiles", unit="tile"):
                key = (tile.iy, tile.ix)
                tiles_by_key[key] = tile

                x, y, w, h = tile.x, tile.y, tile.w, tile.h
                tile_3d = self.ndpi.get_tile(x=x, y=y, w=w, h=h, magnification=self.config.magnification)
                tile_2d = self.compress_2d(tile_3d)
                boxes, scores = self.detector.predict(tile_2d)

                if len(boxes) == 0:
                    detections_by_tile[key] = []
                    continue

                tile_candidates: list[Detection] = []
                for box, score in zip(boxes, scores):
                    gx1 = float(x + box[0])
                    gy1 = float(y + box[1])
                    gx2 = float(x + box[2])
                    gy2 = float(y + box[3])

                    x_nm, y_nm, width_nm, height_nm = self._to_nm_bbox(gx1, gy1, gx2, gy2)
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

                detections_by_tile[key] = tile_candidates

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
