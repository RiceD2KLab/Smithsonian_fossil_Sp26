"""Validated configuration objects for the annotator pipeline."""

import os
from dataclasses import dataclass

from src.annotator.detectors import MODEL_CONFIGS, ModelConfig

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
    tenengrad_ksize: int = 3
    annotation_class: str = "paly"

    @property
    def model_key(self) -> str:
        """Return the normalized model identifier."""
        return self.model_name.lower().strip()

    @property
    def model_config(self) -> ModelConfig:
        """Return the static model config for the selected model family."""
        return MODEL_CONFIGS[self.model_key]

    @property
    def tile_size(self) -> int:
        """Return the tile size required by the selected detector."""
        return self.model_config.tile_size

    def validate(self) -> None:
        """Validate filesystem paths and parameter ranges."""
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
        if self.focus_stack_kernel_size not in {1, 3, 5, 7}:
            raise ValueError("focus_stack_kernel_size must be one of: 1, 3, 5, 7.")
        if self.tenengrad_ksize not in {1, 3, 5, 7}:
            raise ValueError("tenengrad_ksize must be one of: 1, 3, 5, 7.")
