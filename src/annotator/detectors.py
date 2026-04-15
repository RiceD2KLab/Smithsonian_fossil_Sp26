"""
This module provides model-specific detector wrappers behind a shared protocol.
"""

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

def sanitize_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clip predicted boxes to image bounds and remove invalid boxes.

    Args:
        boxes: Predicted boxes in `xyxy` format. Can be any array-like object
            convertible to shape `N x 4`.
        scores: Confidence scores aligned with `boxes`.
        width: Tile width in pixels.
        height: Tile height in pixels.

    Returns:
        A tuple `(boxes_xyxy, scores)` where boxes are clipped to
        `[0, width] x [0, height]` and only boxes with positive area are kept.
        Returned arrays use `float32` dtype.
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

class TileDetector(Protocol):
    """
    Protocol for tile-level detection model adapters.

    Implementations accept an RGB tile and return bounding boxes with
    confidence scores.
    """

    def predict(self, tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a single tile.

        Args:
            tile: Input RGB image tile as a NumPy array.

        Returns:
            A tuple `(boxes_xyxy, scores)` where boxes are shape `N x 4` in
            pixel `xyxy` coordinates and scores are shape `N`.
        """
        ...

class YOLOTileDetector(TileDetector):
    """Tile detector adapter for Ultralytics YOLO checkpoints."""

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float,
        tile_size: int,
    ) -> None:
        """
        Initialize a YOLO detector adapter.

        Args:
            checkpoint_path: Path to the YOLO checkpoint file.
            confidence_threshold: Minimum confidence score for detections.
            tile_size: Input image size used for YOLO inference.

        Returns:
            None.
        """
        from ultralytics import YOLO

        self.model = YOLO(checkpoint_path)
        self.confidence_threshold = confidence_threshold
        self.imgsz = tile_size

    def predict(self, tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run YOLO inference on one tile.

        Args:
            tile: Input RGB tile as a NumPy array with shape `H x W x C`.

        Returns:
            A tuple `(boxes_xyxy, scores)` where boxes are sanitized `float32`
            `xyxy` coordinates and scores are `float32` confidences.
        """
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
    """Tile detector adapter for RF-DETR checkpoints."""

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float,
        tile_size: int,
    ) -> None:
        """
        Initialize an RF-DETR detector adapter.

        Args:
            checkpoint_path: Path to RF-DETR pretrained weights.
            confidence_threshold: Minimum confidence score for detections.
            tile_size: Input resolution passed to RF-DETR.

        Returns:
            None.

        Raises:
            ValueError: If the RF-DETR base model class is not available.
        """
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
        """
        Run RF-DETR inference on one tile.

        Args:
            tile: Input RGB tile as a NumPy array with shape `H x W x C`.

        Returns:
            A tuple `(boxes_xyxy, scores)` where boxes are sanitized `float32`
            `xyxy` coordinates and scores are `float32` confidences.
        """
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
    """
    Construct a detector adapter for a supported model family.

    Args:
        model_name: Model family key. Supported values are `yolo` and `rfdetr`.
        checkpoint_path: Path to model weights.
        confidence_threshold: Minimum confidence score for detections.
        tile_size: Input tile size for model inference.

    Returns:
        A detector implementing the :class:`TileDetector` protocol.

    Raises:
        ValueError: If `model_name` is not supported.
    """
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
