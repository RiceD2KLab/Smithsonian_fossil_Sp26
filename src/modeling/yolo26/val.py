"""
Evaluates trained YOLO26 models on the test (or val) split. 
Computes standard detection metrics including mAP@0.5, mAP@0.5:0.95, precision, recall, and F1.

COMMAND (typical usage):
    python -m src.modeling.yolo26.val \\
        --model runs/detect/yolo26/weights/best.pt \\
        --h5_root data/tiles \\
        --splits_json data/tiles/train_val_test.json \\
        --split test \\
        --imgsz 1024 \\
        --batch 8 \\
        --workers 4 \\
        --project runs/detect \\
        --name yolo26_val \\
        --plots \\
        --single_cls

EVALUATION METRICS:
    - mAP@0.5: Mean Average Precision at IoU threshold 0.5
    - mAP@0.5:0.95: Mean Average Precision averaged over IoU 0.5 to 0.95
    - Precision: TP / (TP + FP) at confidence threshold
    - Recall: TP / (TP + FN) at confidence threshold
    - F1: Harmonic mean of precision and recall

OUTPUTS:
    - Confusion matrix plot
    - PR curve (Precision-Recall)
    - F1 curve across confidence thresholds
    - Predictions visualization (optional)
    - results.json if --save_json is enabled
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from src.modeling.yolo26.dataset import H5YOLODataset
from src.modeling.yolo26.utils import prepare_h5_yaml


# Global arguments for access by H5DetectionValidator
# Required because Ultralytics validator instantiation doesn't pass custom args
GLOBAL_ARGS: argparse.Namespace | None = None


# Dataloader Configuration Helpers

def compute_validator_stride(validator: DetectionValidator) -> int:
    """
    Determine the model's maximum stride for dataset configuration.

    Args:
        validator: DetectionValidator instance with model/stride attributes.

    Returns:
        Integer stride value, or Ultralytics YOLO defaults to stride 32 if not determinable.
    """
    # Try to get stride from validator attribute
    if hasattr(validator, "stride") and validator.stride is not None:
        return int(validator.stride)

    # Try to get stride from loaded model
    if hasattr(validator, "model") and validator.model is not None:
        unwrapped_model = unwrap_model(validator.model)
        return max(int(unwrapped_model.stride.max()), 32)

    # Default YOLO stride
    return 32


def create_h5_validation_dataloader(
    validator: DetectionValidator,
    h5_root: str,
    splits_json: str,
    split_name: str,
    batch_size: int,
    rank: int,
    cache_labels: bool,
    single_cls: bool,
    use_best_plane: bool,
) -> Any:
    """
    Create a PyTorch DataLoader for the H5-backed validation/testing.

    Unlike training, validation uses:
        - No augmentation (augment=False)
        - Rectangular mode for efficiency (rect=True)
        - Higher padding factor (pad=0.5)
        - No shuffling

    Args:
        validator: DetectionValidator instance for accessing the model/args.
        h5_root: Directory containing .h5 tile files.
        splits_json: Path to train_val_test.json.
        split_name: Which split to evaluate ("train", "val", or "test").
        batch_size: Number of samples per batch.
        rank: Distributed training rank (-1 for single GPU).
        cache_labels: Whether to cache label metadata to disk.
        single_cls: Whether to use single-class detection mode.
        use_best_plane: Whether to use the best focal plane.

    Returns:
        PyTorch DataLoader configured for H5 validation dataset.
    """
    # Compute model stride for proper image sizing
    model_stride: int = compute_validator_stride(validator)

    # Synchronize dataset creation across distributed processes
    with torch_distributed_zero_first(rank):
        dataset = H5YOLODataset(
            h5_root=h5_root,
            splits_json_path=splits_json,
            split_name=split_name,
            data=validator.data,
            cache_labels=cache_labels,
            single_cls=single_cls,
            use_best_plane=use_best_plane,
            imgsz=validator.args.imgsz,
            batch_size=batch_size,
            augment=False,  # No augmentation during evaluation
            hyp=validator.args,
            rect=True,  # Rectangular mode for efficiency
            stride=model_stride,
            pad=0.5,  # Standard validation padding
            prefix=colorstr(f"{split_name}: "),
            task="detect",
            classes=None,  # Evaluate all classes
            fraction=1.0,  # Use all data
        )

    # Build and return the dataloader
    return build_dataloader(
        dataset,
        batch=batch_size,
        workers=validator.args.workers,
        shuffle=False,  # Never shuffle during evaluation
        rank=rank,
        drop_last=False,  # Evaluate all samples
    )


class H5DetectionValidator(DetectionValidator):
    """
    Detection validator that supports the H5-backed datasets.

    Extends Ultralytics DetectionValidator to intercept the 
    dataloader creation and substitute the H5YOLODataset.
    """

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int,
        rank: int = 0,
        mode: str = "",
    ) -> Any:
        """
        Create dataloader for validation/testing.

        Ultralytics calls this with only (dataset_path, batch_size); rank and mode
        default so the signature matches the parent API (will error if not provided for some reason).

        Args:
            dataset_path: Path to dataset (unused for H5 mode).
            batch_size: Samples per batch.
            rank: Distributed training rank (default 0).
            mode: Fallback split name if not in args (default "val").

        Returns:
            PyTorch DataLoader for evaluation.
        """
        # Access global configuration
        global GLOBAL_ARGS
        h5_root: str = GLOBAL_ARGS.h5_root
        splits_json: str = GLOBAL_ARGS.splits_json

        if not h5_root or not splits_json:
            raise ValueError(
                "H5 validation requires --h5_root and --splits_json. "
            )

        split_name: str = getattr(GLOBAL_ARGS, "split", mode)

        return create_h5_validation_dataloader(
            validator=self,
            h5_root=h5_root,
            splits_json=splits_json,
            split_name=split_name,
            batch_size=batch_size,
            rank=rank,
            cache_labels=getattr(GLOBAL_ARGS, "cache_labels", True),
            single_cls=getattr(GLOBAL_ARGS, "single_cls", False),
            use_best_plane=getattr(GLOBAL_ARGS, "use_best_plane", False),
        )


def parse_validation_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for YOLO validation.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO26 model on H5-backed test/val split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m src.modeling.yolo26.val \\
        --model runs/detect/yolo26/weights/best.pt \\
        --h5_root data/tiles \\
        --splits_json data/tiles/train_val_test.json \\
        --split test \\
        --imgsz 1024 \\
        --batch 8 \\
        --workers 4 \\
        --project runs/detect \\
        --name yolo26_val
        """,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (.pt).",
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
        help="Path to tiles metadata.json for class names. Optional.",
    )

    # Split selection
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Split to evaluate: 'train', 'val', or 'test'.",
    )

    # Evaluation configuration
    parser.add_argument(
        "--imgsz",
        type=int,
        required=True,
        help="Input image size.",
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
        help="Number of dataloader workers.",
    )

    # Device and output configuration
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device to run on (e.g., '0', 'cpu'). Auto-detected if not set.",
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
        help="Run name within project directory.",
    )

    # Detection mode
    parser.add_argument(
        "--single_cls",
        action="store_true",
        help="Map all palynomorph types to single class.",
    )
    parser.add_argument(
        "--use_best_plane",
        action="store_true",
        help="Use the best focal plane instead of the focus-stacked image.",
    )

    # Caching
    parser.add_argument(
        "--cache_labels",
        action="store_true",
        help="Cache H5 label metadata to disk.",
    )

    # Output options
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Save results to COCO-format JSON.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate evaluation plots (confusion matrix, PR curve, etc.).",
    )
    parser.add_argument(
        "--no_plots",
        action="store_false",
        dest="plots",
        help="Disable plot generation.",
    )

    return parser.parse_args()


def validate_paths(h5_root: str, splits_json: str) -> None:
    """
    Validate that the required paths exist.

    Args:
        h5_root: Directory path to validate.
        splits_json: File path to validate.

    Raises:
        FileNotFoundError: If either path doesn't exist.
    """
    if not os.path.isdir(h5_root):
        raise FileNotFoundError(f"h5_root directory not found: {h5_root}")
    if not os.path.isfile(splits_json):
        raise FileNotFoundError(f"splits_json file not found: {splits_json}")


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
    Build the overrides dictionary for the YOLO validation.

    Args:
        yaml_path: Path to H5 dataset YAML.
        split: Which split to evaluate.
        imgsz: Input image size.
        batch: Batch size.
        workers: Number of workers.
        project: Project output directory.
        name: Run name.
        device: Device string or None for auto-detection.
        save_json: Whether to save COCO-format JSON.
        plots: Whether to generate plots.

    Returns:
        Dictionary of validation overrides.
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

    Parses arguments, loads model, generates YAML config, and runs
    validation with custom H5-aware validator.
    """
    # Parse command-line arguments
    args: argparse.Namespace = parse_validation_arguments()

    # Store args globally for access by H5DetectionValidator
    global GLOBAL_ARGS
    GLOBAL_ARGS = args

    # Validate required paths exist
    validate_paths(args.h5_root, args.splits_json)

    # Load trained model
    model: YOLO = YOLO(args.model)

    # Auto-detect single_cls for models trained on single class
    if not args.single_cls and not args.metadata:
        if auto_detect_single_class_model(model):
            print("Auto-detected single-class model (nc=1). Enabling --single_cls.")
            args.single_cls = True

    # Generate YAML configuration pointing to H5 data
    yaml_path: str = prepare_h5_yaml(
        h5_root=args.h5_root,
        metadata_path=args.metadata,
        single_cls=args.single_cls,
    )

    # Build validation configuration
    overrides: dict[str, Any] = build_validation_overrides(
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

    model.val(**overrides, validator=H5DetectionValidator)


if __name__ == "__main__":
    main()
