from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from src.annotator.models import TileDetector, build_detector
from src.annotator.nms import non_max_suppression
from src.annotator.types import Detection
from src.data.ndpa_writer import NDPAWriter
from src.data.ndpi_reader import NDPIData
from src.preprocessing.focus_stack import focus_stack


@dataclass
class AnnotatorConfig:
    """Configuration for end-to-end NDPI annotation."""

    ndpi_path: str
    output_dir: str
    model_name: str
    checkpoint_path: str
    overlap: float
    magnification: float
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.5
    rfdetr_variant: str = "base"
    yolo_imgsz: int | None = None
    focus_stack_kernel_size: int = 5
    annotation_class: str = "paly"
    device: str | None = None

    @property
    def tile_size(self) -> int:
        key = self.model_name.strip().lower().replace("-", "")
        if key == "yolo":
            return 1024
        if key == "rfdetr":
            return 1008
        raise ValueError("model_name must be either 'yolo' or 'rfdetr'.")

    def validate(self) -> None:
        if not os.path.isfile(self.ndpi_path):
            raise FileNotFoundError(f"NDPI file not found: {self.ndpi_path}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
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


class NDPIAnnotator:
    """Pipeline that annotates a full NDPI slide and writes merged CSV and NDPA output."""

    def __init__(
        self,
        config: AnnotatorConfig,
        focus_stack_fn: Callable[..., object] = focus_stack,
    ) -> None:
        config.validate()
        self.config = config
        self.ndpi = NDPIData(config.ndpi_path)
        self.focus_stack_fn = focus_stack_fn
        self.detector: TileDetector = build_detector(
            model_name=config.model_name,
            checkpoint_path=config.checkpoint_path,
            confidence_threshold=config.confidence_threshold,
            tile_size=config.tile_size,
            rfdetr_variant=config.rfdetr_variant,
            yolo_imgsz=config.yolo_imgsz,
            device=config.device,
        )

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

    def _build_tile_grid(self) -> list[tuple[int, int, int, int]]:
        image_w, image_h = self._image_size_at_magnification()
        tile_size = self.config.tile_size
        stride = max(1, int(round(tile_size * (1.0 - self.config.overlap))))

        xs = self._axis_positions(image_w, tile_size, stride)
        ys = self._axis_positions(image_h, tile_size, stride)
        return [(x, y, tile_size, tile_size) for y in ys for x in xs]

    def _to_nm_bbox(self, x1_px: float, y1_px: float, x2_px: float, y2_px: float) -> tuple[float, float, float, float]:
        x_nm = x1_px * self._nm_per_px + self._origin_x_nm
        y_nm = y1_px * self._nm_per_px + self._origin_y_nm
        width_nm = max(x2_px - x1_px, 0.0) * self._nm_per_px
        height_nm = max(y2_px - y1_px, 0.0) * self._nm_per_px
        return x_nm, y_nm, width_nm, height_nm

    @staticmethod
    def _clip_boxes_xyxy(
        boxes: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        clipped = boxes.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
        clipped[:, 2] = np.clip(clipped[:, 2], 0, width)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
        clipped[:, 3] = np.clip(clipped[:, 3], 0, height)
        valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
        return clipped[valid], valid

    def _focus_stack_tile(self, tile_4d: np.ndarray) -> np.ndarray:
        stacked = self.focus_stack_fn(tile_4d, k=self.config.focus_stack_kernel_size)

        if isinstance(stacked, tuple):
            stacked = stacked[0]

        if stacked.dtype != np.uint8:
            stacked = np.clip(stacked, 0, 255).astype(np.uint8)
        return stacked

    def _resolve_output_paths(self) -> tuple[str, str]:
        ndpi_name = Path(self.config.ndpi_path).name
        ndpi_stem = ndpi_name[: -len(".ndpi")] if ndpi_name.lower().endswith(".ndpi") else Path(ndpi_name).stem

        output_dir = os.path.abspath(self.config.output_dir)
        csv_path = os.path.join(output_dir, f"{ndpi_stem}.csv")
        ndpa_path = os.path.join(output_dir, f"{ndpi_name}.ndpa")
        return csv_path, ndpa_path

    def _write_csv(self, detections: list[Detection], output_csv_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        fieldnames = [
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

    def _write_ndpa(self, detections: list[Detection], output_ndpa_path: str) -> None:
        writer = NDPAWriter(output_path=output_ndpa_path)
        for det in detections:
            writer.add_bounding_box(
                label=self.config.annotation_class,
                lens=self.config.magnification,
                x_nm=det.x_nm,
                y_nm=det.y_nm,
                width_nm=det.width_nm,
                height_nm=det.height_nm,
            )
        writer.save()

    def run(self) -> list[Detection]:
        """Execute the full NDPI annotation pipeline and return merged detections."""
        tiles = self._build_tile_grid()
        merged_detections: list[Detection] = []

        for x, y, w, h in tqdm(tiles, desc="Annotating tiles", unit="tile"):
            tile_4d = self.ndpi.get_tile(x=x, y=y, w=w, h=h, magnification=self.config.magnification)
            tile_3d = self._focus_stack_tile(tile_4d)

            boxes, scores = self.detector.predict(tile_3d)
            if len(boxes) == 0:
                continue

            boxes, valid = self._clip_boxes_xyxy(boxes, width=w, height=h)
            if len(boxes) == 0:
                continue

            scores = np.asarray(scores, dtype=np.float32)
            scores = scores[valid]

            for box, score in zip(boxes, scores):
                gx1 = float(x + box[0])
                gy1 = float(y + box[1])
                gx2 = float(x + box[2])
                gy2 = float(y + box[3])
                x_nm, y_nm, width_nm, height_nm = self._to_nm_bbox(gx1, gy1, gx2, gy2)
                merged_detections.append(
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

        final_detections = non_max_suppression(
            detections=merged_detections,
            iou_threshold=self.config.nms_iou_threshold,
        )
        csv_path, ndpa_path = self._resolve_output_paths()
        self._write_csv(final_detections, csv_path)
        self._write_ndpa(final_detections, ndpa_path)
        self.csv_output_path = csv_path
        self.ndpa_output_path = ndpa_path
        return final_detections
