from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class TileDetector(Protocol):
    """Interface for tile-level detection models."""

    def predict(self, tile_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (boxes_xyxy, scores) for an RGB tile."""


class YOLOTileDetector:
    """Ultralytics YOLO adapter."""

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(checkpoint_path)
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.device = device

    def predict(self, tile_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
        kwargs = {
            "source": tile_bgr,
            "conf": self.confidence_threshold,
            "verbose": False,
        }
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)

        boxes_tensor = results[0].boxes
        if boxes_tensor is None or len(boxes_tensor) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32)

        boxes = boxes_tensor.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes_tensor.conf.cpu().numpy().astype(np.float32)
        return boxes, conf


class RFDETRTileDetector:
    """RF-DETR adapter with ndarray-first inference and file-path fallback."""

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float,
        resolution: int,
        variant: str = "base",
    ) -> None:
        from src.models.rfdetr.train import _MODEL_CLASSES

        if variant not in _MODEL_CLASSES:
            valid = ", ".join(sorted(_MODEL_CLASSES))
            raise ValueError(f"Unknown RF-DETR variant '{variant}'. Choose one of: {valid}.")

        self.model = _MODEL_CLASSES[variant](
            pretrain_weights=checkpoint_path,
            num_classes=1,
            resolution=resolution,
        )
        self.model.optimize_for_inference()
        self.confidence_threshold = confidence_threshold
        self._supports_ndarray: bool | None = None

    def _parse_detections(self, detections: object) -> tuple[np.ndarray, np.ndarray]:
        boxes = getattr(detections, "xyxy", None)
        scores = getattr(detections, "confidence", None)

        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32)

        boxes_np = np.asarray(boxes, dtype=np.float32)
        if scores is None:
            scores_np = np.ones(len(boxes_np), dtype=np.float32)
        else:
            scores_np = np.asarray(scores, dtype=np.float32)
        return boxes_np, scores_np

    def _predict_with_path(self, tile_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = Path(handle.name)

        try:
            cv2.imwrite(str(temp_path), tile_bgr)
            detections = self.model.predict(str(temp_path), threshold=self.confidence_threshold)
            return self._parse_detections(detections)
        finally:
            temp_path.unlink(missing_ok=True)

    def predict(self, tile_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)

        if self._supports_ndarray is not False:
            try:
                detections = self.model.predict(tile_bgr, threshold=self.confidence_threshold)
                self._supports_ndarray = True
                return self._parse_detections(detections)
            except TypeError:
                self._supports_ndarray = False

        return self._predict_with_path(tile_bgr)


def build_detector(
    model_name: str,
    checkpoint_path: str,
    confidence_threshold: float,
    tile_size: int,
    rfdetr_variant: str = "base",
    yolo_imgsz: int | None = None,
    device: str | None = None,
) -> TileDetector:
    """Create a detector adapter for the requested model family."""
    model_key = model_name.strip().lower().replace("-", "")

    if model_key == "yolo":
        return YOLOTileDetector(
            checkpoint_path=checkpoint_path,
            confidence_threshold=confidence_threshold,
            imgsz=yolo_imgsz,
            device=device,
        )

    if model_key == "rfdetr":
        return RFDETRTileDetector(
            checkpoint_path=checkpoint_path,
            confidence_threshold=confidence_threshold,
            resolution=tile_size,
            variant=rfdetr_variant,
        )

    raise ValueError("model_name must be either 'yolo' or 'rfdetr'.")
