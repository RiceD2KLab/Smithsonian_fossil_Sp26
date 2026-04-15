"""Model adapters and box sanitation helpers for tile-level detection."""

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

@dataclass
class ModelConfig:
    """Static model configuration used throughout the annotator pipeline."""

    key: str # Unique model name, e.g. "yolo" or "rfdetr".
    tile_size: int # Tile size in pixels for model inference, e.g. 1024 or 1008.
    ndpa_color: str # Color of output NDPA annotations for this model, e.g. "#ff0000" or "#00ff00".

MODEL_CONFIGS: dict[str, ModelConfig] = {
    "yolo": ModelConfig(key="yolo", tile_size=1024, ndpa_color="#ff0000"),
    "rfdetr": ModelConfig(key="rfdetr", tile_size=1008, ndpa_color="#00ff00"),
}

def sanitize_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip boxes to tile bounds and remove degenerate boxes."""
    boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)

    if len(boxes_np) == 0 or len(scores_np) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32)

    n = min(len(boxes_np), len(scores_np))
    boxes_np = boxes_np[:n]
    scores_np = scores_np[:n]

    clipped = boxes_np.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
    clipped[:, 2] = np.clip(clipped[:, 2], 0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
    clipped[:, 3] = np.clip(clipped[:, 3], 0, height)

    valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    return clipped[valid], scores_np[valid]

class TileDetector(Protocol):
    """Interface for tile-level detection models."""

    def predict(self, tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `(boxes_xyxy, scores)` for a single tile."""
        ...

class YOLOTileDetector(TileDetector):
    """Tile detector wrapper for Ultralytics YOLO."""

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float,
        tile_size: int,
    ) -> None:
        """Load the YOLO checkpoint and store inference parameters."""
        from ultralytics import YOLO

        self.model = YOLO(checkpoint_path)
        self.confidence_threshold = confidence_threshold
        self.imgsz = tile_size

    def predict(self, tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run YOLO on a tile and return sanitized xyxy boxes plus scores."""
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        h, w = tile.shape[:2]
        results = self.model.predict(
            source=tile_bgr,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

        boxes_tensor = results[0].boxes
        if boxes_tensor is None or len(boxes_tensor) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32)

        boxes = boxes_tensor.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes_tensor.conf.cpu().numpy().astype(np.float32)
        return sanitize_boxes(boxes, conf, width=w, height=h)


class RFDETRTileDetector(TileDetector):
    """Tile detector wrapper for RF-DETR."""

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float,
        tile_size: int,
    ) -> None:
        """Load the RF-DETR checkpoint and prepare it for inference."""
        from src.models.rfdetr.train import _MODEL_CLASSES

        model_cls = _MODEL_CLASSES.get("base")
        if model_cls is None:
            raise ValueError("RF-DETR variant is unavailable in _MODEL_CLASSES.")

        self.model = model_cls(
            pretrain_weights=checkpoint_path,
            num_classes=1,
            resolution=tile_size,
        )
        self.model.optimize_for_inference()
        self.confidence_threshold = confidence_threshold

    def predict(self, tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run RF-DETR on a tile and return sanitized xyxy boxes plus scores."""
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        h, w = tile.shape[:2]
        detections = self.model.predict(tile_bgr, threshold=self.confidence_threshold)

        boxes = getattr(detections, "xyxy", None)
        scores = getattr(detections, "confidence", None)

        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32)

        boxes_np = np.asarray(boxes, dtype=np.float32)
        if scores is None:
            scores_np = np.ones(len(boxes_np), dtype=np.float32)
        else:
            scores_np = np.asarray(scores, dtype=np.float32)

        return sanitize_boxes(boxes_np, scores_np, width=w, height=h)

def build_detector(
    model_name: str,
    checkpoint_path: str,
    confidence_threshold: float,
    tile_size: int,
) -> TileDetector:
    """Create a detector adapter for the requested model family."""
    if model_name == "yolo":
        return YOLOTileDetector(
            checkpoint_path=checkpoint_path,
            confidence_threshold=confidence_threshold,
            tile_size=tile_size,
        )

    if model_name == "rfdetr":
        return RFDETRTileDetector(
            checkpoint_path=checkpoint_path,
            confidence_threshold=confidence_threshold,
            tile_size=tile_size,
        )

    raise ValueError("model_name must be either 'yolo' or 'rfdetr'.")
