import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

"""
This file contains model definitions and adapters for tile-level detection 
in the annotator pipeline. The TileDetector protocol defines a common interface 
for different model families, and the build_detector function creates adapters 
based on user input.
"""

# Relevant model-specific configuration parameters
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
    """
    Clip boxes to tile bounds and remove invalid boxes.
    """
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

# A generic TileDetector protocol
class TileDetector(Protocol):
    """Interface for tile-level detection models."""

    def predict(self, tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (boxes_xyxy, scores) for a tile."""

class YOLOTileDetector (TileDetector):
    """Implementation of TileDetector for Ultralytics YOLO26"""

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float,
        tile_size: int,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(checkpoint_path)
        self.confidence_threshold = confidence_threshold
        self.imgsz = tile_size

    def predict(self, tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        
        # YOLO expects BGR input when using the model.predict interface, so convert from RGB.
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        h, w = tile.shape[:2]
        
        # Run inference on single tile
        kwargs = {
            "source": tile_bgr,
            "conf": self.confidence_threshold,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        results = self.model.predict(**kwargs)

        # Convery to numpy arrays and return
        boxes_tensor = results[0].boxes
        if boxes_tensor is None or len(boxes_tensor) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32)

        boxes = boxes_tensor.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes_tensor.conf.cpu().numpy().astype(np.float32)
        return sanitize_boxes(boxes, conf, width=w, height=h)


class RFDETRTileDetector (TileDetector):
    """Implementation of TileDetector for RF-DETR"""

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float,
        tile_size: int,
    ) -> None:
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
        return boxes_np, scores_np
        # return sanitize_boxes(boxes_np, scores_np, width=w, height=h)

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
