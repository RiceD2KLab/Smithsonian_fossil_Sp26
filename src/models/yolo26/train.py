"""
YOLO training script for palynomorph detection using COCO-format datasets.

Expects a COCO export produced by ``export_coco.py``, which writes:

    coco_dir/
      images/train/   images/val/   images/test/
      annotations/    instances_train.json  instances_val.json  instances_test.json
      dataset.yaml

COMMAND (typical usage):
    python -m src.modeling.yolo26.train \\
        --model yolo26s.pt \\
        --coco_dir data/coco_export \\
        --epochs 100 \\
        --imgsz 1024 \\
        --batch 8 \\
        --workers 4 \\
        --project runs/detect \\
        --name yolo26 \\
        --optimizer adamw \\
        --lr0 0.001 \\
        --lrf 0.001

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
        - AdamW with default parameters (default)
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

from ultralytics import YOLO

from src.modeling.yolo26.utils import get_coco_yaml_path


def parse_training_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for YOLO training.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a YOLO26 model on a COCO-format palynomorph dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m src.modeling.yolo26.train \\
        --model yolo26s.pt \\
        --coco_dir data/coco_export \\
        --epochs 100 \\
        --imgsz 1024 \\
        --batch 8 \\
        --workers 4 \\
        --project runs/detect \\
        --name yolo26
        """,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model weights or variant name (e.g. 'yolo26s.pt').",
    )

    # Dataset
    parser.add_argument(
        "--coco_dir",
        type=str,
        required=True,
        help=(
            "Root directory of the COCO export produced by export_coco.py. "
            "Must contain dataset.yaml, images/, and annotations/."
        ),
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
        help="Input image size (pixels).",
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
        help="Number of dataloader worker processes.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="Optimizer (e.g. 'adam', 'adamw', 'sgd'). Default: adamw.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate. Default: 0.001.",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.001,
        help="Final learning rate factor. Default: 0.001.",
    )

    # Device and output
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (e.g. '0', '0,1', 'cpu'). Auto-detected if not set.",
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
        help="Run name within the project directory.",
    )

    # Optional metadata
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to tiles metadata.json. Optional — only used for single-class auto-detection.",
    )

    return parser.parse_args()


def validate_coco_dir(coco_dir: str) -> None:
    """
    Validate that the COCO export directory has the expected structure.

    Args:
        coco_dir: Root directory of the COCO export.

    Raises:
        FileNotFoundError: If coco_dir or dataset.yaml are missing.
    """
    if not os.path.isdir(coco_dir):
        raise FileNotFoundError(f"coco_dir not found: {coco_dir}")
    yaml_path = get_coco_yaml_path(coco_dir)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f"dataset.yaml not found at {yaml_path}. "
            "Run export_coco.py first to generate the COCO dataset."
        )


def auto_detect_single_class_model(model: YOLO) -> bool:
    """
    Check if a model is configured for single-class detection.

    Args:
        model: Loaded YOLO model instance.

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
    optimizer: str,
    lr0: float,
    lrf: float,
) -> dict[str, Any]:
    """
    Build the overrides dictionary for YOLO training.

    Args:
        yaml_path: Path to the COCO dataset.yaml.
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Batch size.
        workers: Number of dataloader workers.
        project: Project output directory.
        name: Run name.
        device: Device string, or None for auto-detection.
        optimizer: Optimizer name.
        lr0: Initial learning rate.
        lrf: Final learning rate factor.

    Returns:
        Dictionary of training overrides for model.train().
    """
    overrides: dict[str, Any] = {
        "data": yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "project": project,
        "name": name,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
    }

    if device is not None:
        overrides["device"] = device

    return overrides


def main() -> None:
    """
    Main entry point for YOLO26 training.

    Validates the COCO export directory, loads the model, and starts training
    using the dataset.yaml generated by export_coco.py.
    """
    args = parse_training_arguments()

    validate_coco_dir(args.coco_dir)

    model: YOLO = YOLO(args.model)

    if auto_detect_single_class_model(model):
        print("Auto-detected single-class model (nc=1).")

    yaml_path = get_coco_yaml_path(args.coco_dir)

    overrides = build_training_overrides(
        yaml_path=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
    )

    model.train(**overrides)


if __name__ == "__main__":
    main()
