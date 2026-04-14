"""
RF-DETR training, using COCO-format data exported by export_coco.py.

Creates a staging directory with symlinks to bridge our COCO export
layout to rfdetr's expected structure for roboflow.

export_coco layout           rfdetr roboflow layout
------------------           ------------------------
images/train/*.jpeg          {staging}/train/*.jpeg
annotations/                 {staging}/train/_annotations.coco.json
    instances_train.json
images/val/*.jpeg            {staging}/valid/*.jpeg
annotations/                 {staging}/valid/_annotations.coco.json
    instances_val.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from PIL import Image

from rfdetr.detr import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall
from rfdetr_plus import RFDETR2XLarge, RFDETRXLarge

# Maps variant name -> model class imported from rfdetr / rfdetr_plus.
_MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "base": RFDETRBase,
    "large": RFDETRLarge,
    "xlarge": RFDETRXLarge,
    "2xlarge": RFDETR2XLarge,
}

# patch_size × num_windows = minimum block size the backbone requires.
# base/small/nano/large: patch_size=14, num_windows=4 → 56
# xlarge/2xlarge:        patch_size=20, num_windows=2 → 40
_MODEL_IMGSZ_DIVISOR = {
    "nano":    56,
    "small":   56,
    "base":    56,
    "large":   56,
    "xlarge":  40,
    "2xlarge": 40,
}

# Maps split names (from export_coco.py) to rfdetr's expected names.
_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}

# Augmentation presets passed to rfdetr's aug_config kwarg.
# "custom" applies flips, rotation, brightness/contrast, and HSV jitter.
# "none"   disables all augmentation.
AUG_CONFIG = {
    "custom": {
        "HorizontalFlip": {"p": 0.5},
        "VerticalFlip": {"p": 0.5},
        "Rotate": {"limit": 180, "p": 0.7},
        "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.4},
        "HueSaturationValue": {"hue_shift_limit": 10, "sat_shift_limit": 20, "val_shift_limit": 20, "p": 0.3},
    },
    "none": {},
}


def _read_class_names(coco_dir: str) -> list[str]:
    """Read category names from the annotations files in the COCO export.

    Args:
        coco_dir: Root directory of the COCO export produced by export_coco.py:
            Expected to contain annotations/instances_{split}.json files.

    Returns:
        List of category name strings, sorted by COCO category id.
    """
    for split in ("train", "val", "test"):
        ann_file = Path(coco_dir) / "annotations" / f"instances_{split}.json"
        if ann_file.exists():
            with open(ann_file) as f:
                coco = json.load(f)
            categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
            return [c["name"] for c in categories]
    
    raise ValueError(f"No annotation file found in {coco_dir}")


def _prepare_roboflow_layout(coco_dir: str, staging_dir: str) -> str:
    """Bridge our COCO export layout to rfdetr's expected roboflow format via symlinks.

    Creates a staging directory tree mirroring the roboflow layout
    (train/, valid/, test/ each containing images and _annotations.coco.json)
    using symbolic links that point back into the original COCO export directory.

    Args:
        coco_dir: Root directory of the COCO export. Expected structure:
            images/train/, images/val/, images/test/
            annotations/instances_train.json, annotations/instances_val.json, annotations/instances_test.json
        staging_dir: Destination directory for the roboflow-style layout.

    Returns:
        The resolved path to staging_dir as a string.
    """
    coco_path = Path(coco_dir).resolve()
    staging = Path(staging_dir)

    for src_split, dst_split in _SPLIT_MAP.items():
        img_src_dir = coco_path / "images" / src_split
        ann_src = coco_path / "annotations" / f"instances_{src_split}.json"

        if not img_src_dir.is_dir():
            continue

        dst = staging / dst_split
        dst.mkdir(parents=True, exist_ok=True)

        ann_dst = dst / "_annotations.coco.json"
        if ann_src.is_file() and not ann_dst.exists():
            ann_dst.symlink_to(ann_src)

        for img_file in img_src_dir.iterdir():
            if img_file.is_file() and not img_file.name.startswith("."):
                link = dst / img_file.name
                if not link.exists():
                    link.symlink_to(img_file)

    return str(staging)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RF-DETR training.

    Returns:
        argparse.Namespace with the following attributes:

        - model (str): RF-DETR variant key (e.g. "base").
        - coco_dir (str): Path to export_coco.py output directory.
        - epochs (int): Total training epochs.
        - batch_size (int): Samples per batch per GPU.
        - grad_accum (int): Gradient accumulation steps.
        - lr (float): Initial learning rate.
        - lr_scheduler (str): "step" or "cosine".
        - lr_min_factor (float): Minimum LR as fraction of initial LR.
        - warmup_epochs (float): Linear LR warmup duration in epochs.
        - weight_decay (float): AdamW weight decay.
        - imgsz (int): Square input resolution in pixels.
        - output_dir (str): Checkpoint and staging output directory.
        - workers (int): Dataloader worker processes.
        - early_stopping_patience (int): Epochs without improvement before stopping.
        - aug_config (str): Augmentation preset key.
        - aug_json (str | None): Per-transform override JSON string.
        - drop_path (float): Stochastic depth drop-path rate.
    """
    parser = argparse.ArgumentParser(
        description="Train RF-DETR on COCO-format data exported by export_coco.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=list(_MODEL_CLASSES.keys()),
        default="base",
        help="RF-DETR model variant (default: base).",
    )
    parser.add_argument("--coco_dir", required=True, help="Path to export_coco.py output.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of samples per batch.")
    parser.add_argument(
        "--grad_accum",
        type=int,
        required=True,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum).",
    )
    parser.add_argument("--lr", type=float, required=True, help="Learning rate.")
    parser.add_argument(
        "--lr_scheduler",
        choices=["step", "cosine"],
        default="cosine",
        help="LR scheduler type (default: cosine).",
    )
    parser.add_argument(
        "--lr_min_factor",
        type=float,
        required=True,
        help="Minimum LR as a fraction of initial LR for cosine annealing.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        required=True,
        help="Linear LR warmup duration in epochs.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=True,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        required=True,
        help=(
            "Image resolution. Must be divisible by the model's block size "
            "(patch_size * num_windows): 56 for nano/small/base/large, "
            "40 for xlarge/2xlarge."
        ),
    )
    parser.add_argument("--output_dir", required=True, help="Checkpoint output directory.")
    parser.add_argument("--workers", type=int, required=True, help="Number of dataloader worker processes.")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        required=True,
        help="Epochs without mAP improvement before stopping.",
    )
    parser.add_argument(
        "--aug_config",
        choices=list(AUG_CONFIG.keys()),
        default="custom",
        help="Augmentation configuration (default: custom).",
    )
    parser.add_argument(
        "--aug_json",
        type=str,
        default=None,
        help=(
            "JSON string of per-transform overrides applied on top of --aug_config.  "
            "E.g. '{\"Rotate\": {\"p\": 0.5}}'.  Use this for hyperparameter tuning; "
            "use --aug_config for manual runs."
        ),
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        required=True,
        help="Stochastic depth drop path rate for ViT backbone.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for RF-DETR training.

    Workflow:
        1. Validates --coco_dir exists and --imgsz is divisible by
           the selected model's required block size.
        2. Reads class names from the COCO export.
        3. Merges --aug_config with any per-transform --aug_json
           overrides to build the final augmentation config dict.
        4. Creates the staging directory
        5. Instantiates the selected model variant and calls model.train().

    Raises:
        FileNotFoundError: If --coco_dir does not exist.
        ValueError: If --imgsz is not divisible by the selected model's required
            block size (see _MODEL_IMGSZ_DIVISOR).
    """
    args = parse_args()

    if not os.path.isdir(args.coco_dir):
        raise FileNotFoundError(f"coco_dir not found: {args.coco_dir}")

    divisor = _MODEL_IMGSZ_DIVISOR[args.model]
    if args.imgsz % divisor != 0:
        raise ValueError(
            f"--imgsz {args.imgsz} is not divisible by {divisor} "
            f"(required for --model {args.model})."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    staging_dir = os.path.join(args.output_dir, ".rfdetr_dataset")
    class_names = _read_class_names(args.coco_dir)

    # Build augmentation config; if --aug_json is provided, overlay with JSON overrides.
    aug = dict(AUG_CONFIG[args.aug_config])
    if args.aug_json:
        overrides = json.loads(args.aug_json)
        for transform_name, kwargs in overrides.items():
            if transform_name in aug:
                aug[transform_name] = {**aug[transform_name], **kwargs}
            else:
                aug[transform_name] = kwargs

    # For DDP training with torchrun, each GPU spawns its own process.
    # Only LOCAL_RANK 0 should create the staging directory;
    # the other ranks wait for the files to exist before training starts.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        _prepare_roboflow_layout(coco_dir=args.coco_dir, staging_dir=staging_dir)

    dataset_dir = staging_dir

    model = _MODEL_CLASSES[args.model]()
    model.train(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_min_factor=args.lr_min_factor,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        resolution=args.imgsz,
        output_dir=args.output_dir,
        num_workers=args.workers,
        class_names=class_names,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_use_ema=True,
        aug_config=aug,
        drop_path=args.drop_path,
    )


if __name__ == "__main__":
    main()
