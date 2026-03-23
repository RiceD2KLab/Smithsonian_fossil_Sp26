"""
Evaluates trained YOLO26 models on the test (or val) split.
Computes standard detection metrics including mAP@0.5, mAP@0.5:0.95, precision, recall, and F1.

Expects a COCO export produced by export_coco.py, which writes:

    coco_dir/
      images/train/   images/val/   images/test/
      annotations/    instances_train.json  instances_val.json  instances_test.json
      dataset.yaml

Evaluation Metrics:
    - mAP@0.5: Mean Average Precision at IoU threshold 0.5
    - mAP@0.5:0.95: Mean Average Precision averaged over IoU 0.5 to 0.95
    - Precision: TP / (TP + FP) at confidence threshold
    - Recall: TP / (TP + FN) at confidence threshold
    - F1: Harmonic mean of precision and recall

Outputs:
    - Confusion matrix plot
    - PR curve (Precision-Recall)
    - F1 curve across confidence thresholds
    - predictions.json if --save_json is enabled
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from ultralytics import YOLO

from src.models.yolo26.utils import convert_coco_labels_to_yolo, get_coco_yaml_path, validate_coco_dir


def parse_validation_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for YOLO validation.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO26 model on a COCO-format palynomorph dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m src.modeling.yolo26.val \\
        --model runs/detect/yolo26/weights/best.pt \\
        --coco_dir data/coco_export \\
        --split test \\
        --imgsz 1024 \\
        --batch 8 \\
        --workers 4 \\
        --project runs/detect \\
        --name yolo26_val
        """,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (.pt).",
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
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Which split to evaluate.",
    )

    # Evaluation configuration
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
        help="Batch size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=True,
        help="Number of dataloader worker processes.",
    )

    # Device and output
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. '0', 'cpu'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project directory for saving results.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name within the project directory.",
    )

    # Output options
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Save predictions to COCO-format predictions.json.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate evaluation plots (confusion matrix, PR curve, etc.).",
    )

    return parser.parse_args()


def build_validation_overrides(
    yaml_path: str,
    split: str,
    imgsz: int,
    batch: int,
    workers: int,
    project: str,
    name: str,
    device: str | None,
    save_json: bool,
    plots: bool,
) -> dict[str, Any]:
    """
    Build the overrides dictionary for YOLO validation.

    Args:
        yaml_path: Path to the COCO dataset.yaml.
        split: Which split to evaluate ("train", "val", or "test").
        imgsz: Input image size.
        batch: Batch size.
        workers: Number of dataloader workers.
        project: Project output directory.
        name: Run name.
        device: Device string, or None for auto-detection.
        save_json: Whether to save COCO-format predictions JSON.
        plots: Whether to generate evaluation plots.

    Returns:
        Dictionary of validation overrides for model.val().
    """
    overrides: dict[str, Any] = {
        "data": yaml_path,
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "project": project,
        "name": name,
        "save_json": save_json,
        "plots": plots,
    }

    if device is not None:
        overrides["device"] = device

    return overrides


def main() -> None:
    """
    Main entry point for YOLO26 evaluation.
    """
    args = parse_validation_arguments()

    validate_coco_dir(args.coco_dir)
    convert_coco_labels_to_yolo(args.coco_dir)

    model: YOLO = YOLO(args.model)

    yaml_path = get_coco_yaml_path(args.coco_dir)

    overrides = build_validation_overrides(
        yaml_path=yaml_path,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        device=args.device,
        save_json=args.save_json,
        plots=args.plots,
    )

    model.val(**overrides)


if __name__ == "__main__":
    main()
