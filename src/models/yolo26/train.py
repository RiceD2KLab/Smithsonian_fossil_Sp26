"""
YOLO training script for palynomorph detection using COCO-format datasets.

Expects a COCO export produced by export_coco.py, which writes:

    coco_dir/
      images/train/   images/val/   images/test/
      annotations/    instances_train.json  instances_val.json  instances_test.json
      dataset.yaml
"""

from __future__ import annotations

import argparse
from typing import Any

from ultralytics import YOLO

from src.models.yolo26.utils import convert_coco_labels_to_yolo, get_coco_yaml_path, validate_coco_dir

AUG_CONFIG: dict[str, dict[str, Any]] = {
    # Matches the RF-DETR Albumentations pipeline
    "custom": {
        "fliplr": 0.5,       # HorizontalFlip(p=0.5)
        "flipud": 0.5,       # VerticalFlip(p=0.5)
        "degrees": 180,      # Rotate(limit=180)
        "hsv_h": 0.028,      # HueSaturationValue(hue_shift_limit=10)
        "hsv_s": 0.078,      # HueSaturationValue(sat_shift_limit=20)
        "hsv_v": 0.2,        # RandomBrightnessContrast(brightness_limit=0.2)
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "auto_augment": "",
        "erasing": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
    },
    # Use Ultralytics default augmentation settings (nothing overridden)
    "default": {},
    # All augmentations off
    "none": {
        "fliplr": 0.0,
        "flipud": 0.0,
        "degrees": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "auto_augment": "",
        "erasing": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
    },
}


def parse_training_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for YOLO training.

    Args:
        None

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a YOLO26 model on a COCO-format palynomorph dataset.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model weights or variant name (e.g. 'yolo26s.pt').",
    )

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

    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        help="Optimizer (e.g. 'adam', 'adamw', 'sgd').",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        required=True,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        required=True,
        help="Final learning rate factor.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        required=True,
        help="Number of epochs to wait for improvement before stopping.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Optimizer momentum (passed to model.train(); omit to use Ultralytics default).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Optimizer weight decay; omit to use Ultralytics default.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=None,
        help="LR warmup epochs; omit to use Ultralytics default.",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=None,
        help="Box loss gain; omit to use Ultralytics default.",
    )
    parser.add_argument(
        "--cls",
        type=float,
        default=None,
        help="Classification loss gain; omit to use Ultralytics default.",
    )
    parser.add_argument(
        "--dfl",
        type=float,
        default=None,
        help="DFL loss gain; omit to use Ultralytics default.",
    )

    # Augmentation
    parser.add_argument(
        "--aug_config",
        choices=list(AUG_CONFIG.keys()),
        default="custom",
        help=(
            "Augmentation configuration (default: custom). "
            "'custom' matches the RF-DETR Albumentations pipeline. "
            "'default' uses Ultralytics built-in defaults. "
            "'none' disables all augmentations."
        ),
    )

    # Augmentation
    parser.add_argument(
        "--aug_config",
        choices=list(AUG_CONFIG.keys()),
        default="custom",
        help=(
            "Augmentation configuration (default: custom). "
            "'custom' matches the RF-DETR Albumentations pipeline. "
            "'default' uses Ultralytics built-in defaults. "
            "'none' disables all augmentations."
        ),
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

    return parser.parse_args()


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
    patience: int,
    aug_config: str,
    momentum: float | None = None,
    weight_decay: float | None = None,
    warmup_epochs: float | None = None,
    box: float | None = None,
    cls_gain: float | None = None,
    dfl: float | None = None,
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
        optimizer: Optimizer name (must not be 'auto' if tuning lr0/momentum).
        lr0: Initial learning rate.
        lrf: Final learning rate factor.
        patience: Early-stopping patience (epochs without improvement).
        aug_config: Augmentation configuration key from AUG_CONFIG.
        momentum, weight_decay, warmup_epochs, box, cls_gain, dfl: Optional;
            only keys with non-None values are added (matches tune.run_trial params).

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
        "patience": patience,
    }

    if device is not None:
        overrides["device"] = device

    overrides.update(AUG_CONFIG[aug_config])

    optional_train: dict[str, float] = {}
    if momentum is not None:
        optional_train["momentum"] = momentum
    if weight_decay is not None:
        optional_train["weight_decay"] = weight_decay
    if warmup_epochs is not None:
        optional_train["warmup_epochs"] = warmup_epochs
    if box is not None:
        optional_train["box"] = box
    if cls_gain is not None:
        optional_train["cls"] = cls_gain
    if dfl is not None:
        optional_train["dfl"] = dfl
    overrides.update(optional_train)

    return overrides


def main() -> None:
    """
    Main entry point for YOLO26 training.

    Workflow:
        1. Parses command-line arguments.
        2. Validates the COCO export directory structure.
        3. Converts COCO annotations to YOLO txt format (idempotent).
        4. Loads the YOLO model and runs model.train() with the built overrides.

    Args:
        None

    Returns:
        None
    """

    args = parse_training_arguments()
    validate_coco_dir(args.coco_dir)
    convert_coco_labels_to_yolo(args.coco_dir)

    model: YOLO = YOLO(args.model)
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
        patience=args.patience,
        aug_config=args.aug_config,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        box=args.box,
        cls_gain=args.cls,
        dfl=args.dfl,
    )

    model.train(**overrides)


if __name__ == "__main__":
    main()
