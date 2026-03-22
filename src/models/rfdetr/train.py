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
from pathlib import Path

from rfdetr.detr import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall

_MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}

_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}

# Custom augmentation: flips, rotation, brightness/contrast, HSV
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
    """Read category names from the first available COCO annotations file."""
    for split in ("train", "val", "test"):
        ann_file = Path(coco_dir) / "annotations" / f"instances_{split}.json"
        if ann_file.exists():
            with open(ann_file) as f:
                coco = json.load(f)
            categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
            return [c["name"] for c in categories]
    return ["palynomorph"]


def _prepare_roboflow_layout(coco_dir: str, staging_dir: str) -> str:
    """
    Bridge our COCO export layout to rfdetr's expected roboflow format via symlinks.
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
    """
    Parse command-line arguments for RF-DETR training.

    Returns:
        Namespace with parsed arguments.
    """
    p = argparse.ArgumentParser(
        description="Train RF-DETR on COCO-format data exported by export_coco.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=list(_MODEL_CLASSES.keys()),
        default="base",
        help="RF-DETR model variant (default: base).",
    )
    p.add_argument("--coco_dir", required=True, help="Path to export_coco.py output.")
    p.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    p.add_argument("--batch_size", type=int, required=True, help="Number of samples per batch.")
    p.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum).",
    )
    p.add_argument("--lr", type=float, required=True, help="Learning rate.")
    p.add_argument(
        "--lr_scheduler",
        choices=["step", "cosine"],
        default="cosine",
        help="LR scheduler type (default: cosine).",
    )
    p.add_argument(
        "--lr_min_factor",
        type=float,
        required=True,
        help="Minimum LR as a fraction of initial LR for cosine annealing.",
    )
    p.add_argument(
        "--warmup_epochs",
        type=float,
        required=True,
        help="Linear LR warmup duration in epochs.",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        required=True,
        help="AdamW weight decay.",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        required=True,
        help="Image resolution (must be divisible by 56..rfdetr backbone req).",
    )
    p.add_argument("--output_dir", default="runs/rfdetr", help="Checkpoint output directory.")
    p.add_argument("--workers", type=int, required=True, help="Number of dataloader worker processes.")
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        required=True,
        help="Epochs without mAP improvement before stopping.",
    )
    p.add_argument(
        "--aug_config",
        choices=list(AUG_CONFIG.keys()),
        default="custom",
        help="Augmentation configuration (default: custom).",
    )
    p.add_argument(
        "--drop_path",
        type=float,
        required=True,
        help="Stochastic depth drop path rate for ViT backbone.",
    )

    return p.parse_args()


def main() -> None:
    """
    Main entry point for RF-DETR training.
    """
    args = parse_args()

    if not os.path.isdir(args.coco_dir):
        raise FileNotFoundError(f"coco_dir not found: {args.coco_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    staging_dir = os.path.join(args.output_dir, ".rfdetr_dataset")
    class_names = _read_class_names(args.coco_dir)

    # For DDP training with torchrun, each GPU spawns its own process.
    # Only LOCAL_RANK 0 creates should create the staging symlinks;
    # The other ranks should skip straight to training; the symlinks
    # will already exist by the time model.train() needs them.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        _prepare_roboflow_layout(args.coco_dir, staging_dir)

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
        aug_config=AUG_CONFIG[args.aug_config],
        drop_path=args.drop_path,
    )


if __name__ == "__main__":
    main()
