"""Configuration objects and validation for the annotator pipeline.

This module defines immutable-style dataclasses that carry model metadata and
runtime settings for the NDPI annotation workflow.
"""

import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Static model configuration used throughout the annotator pipeline."""

    key: str # Unique model name, e.g. "yolo" or "rfdetr".
    tile_size: int # Tile size in pixels for model inference, e.g. 1024 or 1008.
    ndpa_color: str # Color of output NDPA annotations for this model, e.g. "#ff0000" or "#00ff00".

# Predefined static configs for supported model families.
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "yolo": ModelConfig(key="yolo", tile_size=1024, ndpa_color="#ff0000"),
    "rfdetr": ModelConfig(key="rfdetr", tile_size=1008, ndpa_color="#00ff00"),
}

@dataclass
class AnnotatorConfig:
    """Runtime configuration for end-to-end NDPI annotation."""

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
    export_crops: bool = False

    @property
    def model_config(self) -> ModelConfig:
        """Return the static model config for the selected model family."""
        return MODEL_CONFIGS[self.model_name]

    @property
    def tile_size(self) -> int:
        """Return the detector tile size for the configured model."""
        return self.model_config.tile_size

    def validate(self) -> None:
        """
        Validate configuration paths and parameter ranges.

        Raises:
            FileNotFoundError: If `ndpi_path` or `checkpoint_path` does not
                exist.
            ValueError: If any numeric threshold/range is invalid, if
                `output_dir` is empty, or if compression/kernel parameters are
                unsupported.
        """
        if not os.path.isfile(self.ndpi_path):
            raise FileNotFoundError(f"NDPI file not found: {self.ndpi_path}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        _ = self.model_name
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
