"""
YOLO26 training script for palynomorph detection using H5-backed datasets.

Train YOLO(26) models on focus-stacked microscopy tiles stored in HDF5 format.

COMMAND (typical usage):
    python -m src.modeling.yolo26.train \\
        --model yolo26s.pt \\
        --h5_root data/tiles \\
        --splits_json data/tiles/train_val_test.json \\
        --epochs 100 \\
        --imgsz 1024 \\
        --batch 8 \\
        --workers 4 \\
        --project runs/detect \\
        --name yolo26 \\
        --cache_labels \\
        --single_cls

YOLO26/ULTRALYTICS ASSUMPTIONS:
    Loss Function:
        - Detection uses composite loss: box_loss + cls_loss + dfl_loss
        - box_loss: CIoU (Complete IoU) for bounding box regression
        - cls_loss: Binary Cross-Entropy for class predictions
        - dfl_loss: Distribution Focal Loss for box refinement

    Data Augmentation (when augment=True, i.e., training mode):
        - Mosaic: 4 images combined into one (prob=1.0, disabled last 10 epochs)
        - MixUp: Blend two images (prob=0.0 by default)
        - HSV shifts: hue=0.015, saturation=0.7, value=0.4
        - Geometric: scale=0.5, translate=0.1, rotation=0.0, shear=0.0
        - Flips: horizontal=0.5, vertical=0.0
        - Copy-paste: disabled by default

    Optimizer:
        - SGD with momentum=0.937 (default)
        - Initial LR=0.01, final LR=0.01 (cosine annealing)
        - Weight decay=0.0005
        - Warmup: 3 epochs with bias_lr=0.1, momentum=0.8

    Model Architecture (yolo26s.pt):
        - Backbone: CSPDarknet53 variant
        - Neck: PANet (Path Aggregation Network)
        - Head: Decoupled detection head
        - Stride: [8, 16, 32] (3 detection scales)
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from src.modeling.yolo26.dataset import H5YOLODataset
from src.modeling.yolo26.utils import prepare_h5_yaml


# Global arguments for access by the H5DetectionTrainer
# We need this because Ultralytics trainer instantiation doesn't pass custom arguments
GLOBAL_ARGS: argparse.Namespace | None = None


# Dataloader Configuration Helpers

def compute_model_stride(trainer: DetectionTrainer) -> int:
    """
    Determine the model's maximum stride for the dataset configuration.

    The stride is needed to ensure image sizes are divisible by the model's
    grid size. Ultralytics YOLO defaults to stride 32 for the largest feature map.

    Args:
        trainer: DetectionTrainer instance with model attribute.

    Returns:
        Integer stride value
    """
    # Try to get stride from the loaded model
    if hasattr(trainer, "model") and trainer.model is not None:
        unwrapped_model = unwrap_model(trainer.model)
        return max(int(unwrapped_model.stride.max()), 32)

    # Try to get stride from the trainer attribute
    if hasattr(trainer, "stride") and trainer.stride is not None:
        return int(trainer.stride)

    # Default YOLO stride
    return 32


def should_disable_shuffle_for_rect_mode(dataset: H5YOLODataset) -> bool:
    """
    Check if shuffle should be disabled due to rectangular training.

    Rectangular training uses different batch shapes for efficiency, which
    is incompatible with random shuffling across the batches.

    Args:
        dataset: H5YOLODataset instance with potential rect mode enabled.

    Returns:
        True if shuffle should be disabled, False otherwise.
    """
    if not getattr(dataset, "rect", False):
        return False

    # Check if batch shapes vary
    try:
        batch_shapes: np.ndarray = dataset.batch_shapes
        all_shapes_equal: bool = np.all(batch_shapes == batch_shapes[0])
        return not all_shapes_equal
    except (AttributeError, IndexError):
        return False


def create_h5_dataloader(
    trainer: DetectionTrainer,
    h5_root: str,
    splits_json: str,
    mode: str,
    batch_size: int,
    rank: int,
    cache_labels: bool,
    single_cls: bool,
) -> Any:
    """
    Create a PyTorch DataLoader with the H5YOLODataset.

    Args:
        trainer: DetectionTrainer instance for accessing model/arguments.
        h5_root: Directory containing .h5 tile files.
        splits_json: Path to train_val_test.json.
        mode: Dataset mode - "train" or "val".
        batch_size: Number of samples per batch.
        rank: Distributed training rank.
        cache_labels: Whether to cache the label metadata to disk.
        single_cls: Whether to use single-class detection mode.

    Returns:
        PyTorch DataLoader configured for H5 dataset.
    """
    # Compute the model stride for proper image sizing
    model_stride: int = compute_model_stride(trainer)

    # Configure the dataset parameters based on the mode
    is_training: bool = (mode == "train")
    augment_enabled: bool = is_training
    rect_enabled: bool = not is_training  # Rectangular mode for validation
    padding_factor: float = 0.0 if is_training else 0.5

    # If distributed training, synchronize dataset creation across processes
    with torch_distributed_zero_first(rank):
        dataset = H5YOLODataset(
            h5_root=h5_root,
            splits_json_path=splits_json,
            split_name=mode,
            data=trainer.data,
            cache_labels=cache_labels,
            single_cls=single_cls,
            imgsz=trainer.args.imgsz,
            batch_size=batch_size,
            augment=augment_enabled,
            hyp=trainer.args,
            rect=rect_enabled,
            stride=model_stride,
            pad=padding_factor,
            prefix=colorstr(f"{mode}: "),
            task="detect",
            classes=getattr(trainer.args, "classes", None),
            fraction=trainer.args.fraction if is_training else 1.0,
        )

    # Determine shuffle behavior
    shuffle_enabled: bool = is_training

    # Disable shuffle if rectangular training requires consistent batch shapes
    if shuffle_enabled and should_disable_shuffle_for_rect_mode(dataset):
        LOGGER.warning(
            "'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False"
        )
        shuffle_enabled = False

    worker_count: int = trainer.args.workers if is_training else trainer.args.workers * 2
    drop_last_batch: bool = getattr(trainer.args, "compile", False) and is_training

    return build_dataloader(
        dataset,
        batch=batch_size,
        workers=worker_count,
        shuffle=shuffle_enabled,
        rank=rank,
        drop_last=drop_last_batch,
    )


class H5DetectionTrainer(DetectionTrainer):
    """
    Detection trainer that supports H5-backed datasets.

    Extends Ultralytics DetectionTrainer to intercept the dataloader creation process
    and use our H5YOLODataset instead of the default YOLODataset.
    """

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int,
        rank: int,
        mode: str,
    ) -> Any:
        """
        Create a dataloader for training or validation.

        Args:
            dataset_path: Path to the dataset (unused for H5 mode).
            batch_size: Number of samples per batch.
            rank: Distributed training rank.
            mode: Dataset mode - "train" or "val".

        Returns:
            PyTorch DataLoader for the requested mode.
        """
        # Validate mode parameter
        valid_modes: set[str] = {"train", "val"}
        assert mode in valid_modes, f"Mode must be one of {valid_modes}, not '{mode}'."

        global GLOBAL_ARGS
        h5_root: str | None = getattr(GLOBAL_ARGS, "h5_root", None)
        splits_json: str | None = getattr(GLOBAL_ARGS, "splits_json", None)

        if not h5_root or not splits_json:
            raise ValueError(
                "H5 training requires --h5_root and --splits_json. "
            )

        return create_h5_dataloader(
            trainer=self,
            h5_root=h5_root,
            splits_json=splits_json,
            mode=mode,
            batch_size=batch_size,
            rank=rank,
            cache_labels=getattr(GLOBAL_ARGS, "cache_labels", True),
            single_cls=getattr(GLOBAL_ARGS, "single_cls", False),
        )


def parse_training_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for YOLO training.

    All dataset input is H5-backed; see module docstring for example command.

    Returns:
        Namespace with parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train YOLO(26) model on H5-backed palynomorph tile datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m src.modeling.yolo26.train --model yolo26s.pt --h5_root data/tiles \\
        --splits_json data/tiles/train_val_test.json --epochs 100 --imgsz 1024 --batch 8 \\
        --workers 4 --project runs/detect --name yolo26 --cache_labels --single_cls
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model weights (.pt) or model variant name.",
    )

    parser.add_argument(
        "--h5_root",
        type=str,
        required=True,
        help="Directory containing .h5 tile files.",
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        required=True,
        help="Path to train_val_test.json.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=False,
        help="Path to tiles metadata.json for class names/counts. Optional.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        required=True,
        help="Input image size. Use 1024 for H5 tiles.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=True,
        help="Number of workers for the dataloader.",
    )

    # Device and output configuration
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device to train on (e.g., '0', '0,1', 'cpu'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project directory for saving runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name within project directory.",
    )

    # H5-specific options
    parser.add_argument(
        "--cache_labels",
        action="store_true",
        help="Cache H5 label metadata to disk for faster subsequent runs.",
    )
    parser.add_argument(
        "--no_cache_labels",
        action="store_false",
        dest="cache_labels",
        help="Disable label caching.",
    )
    parser.add_argument(
        "--single_cls",
        action="store_true",
        help="Map all palynomorph types to single class (detection-only mode).",
    )

    return parser.parse_args()


def validate_h5_paths(h5_root: str, splits_json: str) -> None:
    """
    Validate that the H5 dataset paths exist.

    Args:
        h5_root: Directory path to validate.
        splits_json: File path to validate.

    """
    if not os.path.isdir(h5_root):
        raise FileNotFoundError(f"h5_root directory not found: {h5_root}")
    if not os.path.isfile(splits_json):
        raise FileNotFoundError(f"splits_json file not found: {splits_json}")


def auto_detect_single_class_model(model: YOLO) -> bool:
    """
    Check if a model is configured for single-class detection mode.

    Args:
        model: Loaded YOLO model.

    Returns:
        True if model has nc=1 (single class), False otherwise.
    """
    try:
        if hasattr(model.model, "nc") and model.model.nc == 1:
            return True
    except AttributeError:
        pass
    return False


def build_training_overrides(
    yaml_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    project: str,
    name: str,
    device: str | None,
) -> dict[str, Any]:
    """
    Build the overrides dictionary for YOLO training (H5 dataset only).

    Args:
        yaml_path: Path to the H5 dataset YAML.
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Number of samples per batch.
        workers: Number of workers for the dataloader.
        project: Project output directory.
        name: Run name.
        device: Device string or None for auto-detection.

    Returns:
        Dictionary of training overrides.
    """
    overrides: dict[str, Any] = {
        "data": yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "project": project,
        "name": name,
    }

    if device is not None:
        overrides["device"] = device

    return overrides


def train_with_h5_dataset(model: YOLO, args: argparse.Namespace) -> None:
    """
    Training using the H5-backed dataset.

    Args:
        model: Loaded YOLO model.
        args: Parsed command-line arguments.
    """
    # Validate that the paths exist
    validate_h5_paths(args.h5_root, args.splits_json)

    # Generate the YAML configuration pointing to the H5 data
    yaml_path: str = prepare_h5_yaml(
        h5_root=args.h5_root,
        metadata_path=args.metadata,
        single_cls=args.single_cls,
    )

    # Build the training configuration
    overrides: dict[str, Any] = build_training_overrides(
        yaml_path=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        device=args.device,
    )

    model.train(**overrides, trainer=H5DetectionTrainer)


def main() -> None:
    """
    Main entry point for YOLO26 training.

    Parses arguments, loads model, and dispatches to appropriate training
    function based on dataset configuration.
    """
    # Parse command-line arguments
    args: argparse.Namespace = parse_training_arguments()

    # Store args globally for access by H5DetectionTrainer
    global GLOBAL_ARGS
    GLOBAL_ARGS = args

    # Load YOLO model from weights or variant name
    model: YOLO = YOLO(args.model)

    # Auto-detect single_cls for models trained on single class
    if not args.single_cls and not args.metadata:
        if auto_detect_single_class_model(model):
            print("Auto-detected single-class model (nc=1). Enabling --single_cls.")
            args.single_cls = True

    train_with_h5_dataset(model, args)


if __name__ == "__main__":
    main()
