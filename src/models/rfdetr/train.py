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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum).",
    )
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument(
        "--lr_scheduler",
        choices=["step", "cosine"],
        default="cosine",
        help="LR scheduler type (default: cosine).",
    )
    p.add_argument(
        "--lr_min_factor",
        type=float,
        default=0.01,
        help="Minimum LR as a fraction of initial LR for cosine annealing (default: 0.01).",
    )
    p.add_argument(
        "--warmup_epochs",
        type=float,
        default=5.0,
        help="Linear LR warmup duration in epochs (default: 5).",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1008,
        help="Image resolution (must be divisible by 56..rfdetr backbone req).",
    )
    p.add_argument("--output_dir", default="runs/rfdetr", help="Checkpoint output directory.")
    p.add_argument("--workers", type=int, default=4, help="DataLoader num_workers.")
    return p.parse_args()


def _validate_category_ids(coco_dir: str, class_names: list[str]) -> None:
    """
    Verify that all annotation category_ids are in [0, num_classes-1].
    rfdetr uses category_id directly as a 0-indexed class label; any value
    >= num_classes causes a suppressed CUDA index-out-of-bounds during training.
    """
    num_classes = len(class_names)
    for split in ("train", "val", "test"):
        ann_file = Path(coco_dir) / "annotations" / f"instances_{split}.json"
        if not ann_file.exists():
            continue
        with open(ann_file) as f:
            coco = json.load(f)
        bad = [
            ann["category_id"]
            for ann in coco.get("annotations", [])
            if ann["category_id"] >= num_classes
        ]
        if bad:
            raise ValueError(
                f"[{split}] found category_id(s) {sorted(set(bad))} but "
                f"num_classes={num_classes} (max valid id={num_classes - 1}).\n"
                f"Re-run export_coco.py with --single_cls and delete the staging "
                f"dir before resubmitting."
            )


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.coco_dir):
        raise FileNotFoundError(f"coco_dir not found: {args.coco_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    staging_dir = os.path.join(args.output_dir, ".rfdetr_dataset")
    class_names = _read_class_names(args.coco_dir)
    _validate_category_ids(args.coco_dir, class_names)
    dataset_dir = _prepare_roboflow_layout(args.coco_dir, staging_dir)

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
        multi_scale=False,
        expanded_scales=False,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
