"""
Configuration helpers for YOLO training and validation with COCO-format datasets.

The COCO dataset structure is produced by ``export_coco.py``:

    coco_dir/
      images/
        train/
        val/
        test/
      annotations/
        instances_train.json
        instances_val.json
        instances_test.json
      dataset.yaml

Typical usage:
    from src.modeling.yolo26.utils import get_coco_yaml_path, get_coco_annotations_path

    yaml_path  = get_coco_yaml_path("data/coco_export")
    ann_path   = get_coco_annotations_path("data/coco_export", "val")
"""

from __future__ import annotations

import os


def get_coco_yaml_path(coco_dir: str) -> str:
    """
    Return the path to the dataset.yaml inside a COCO export directory.

    Args:
        coco_dir: Root directory of the COCO export produced by export_coco.py.

    Returns:
        Absolute path to dataset.yaml.
    """
    return os.path.join(os.path.abspath(coco_dir), "dataset.yaml")


def get_coco_annotations_path(coco_dir: str, split: str) -> str:
    """
    Return the path to the COCO annotations JSON for a given split.

    Args:
        coco_dir: Root directory of the COCO export produced by export_coco.py.
        split: Split name — "train", "val", or "test".

    Returns:
        Absolute path to ``annotations/instances_<split>.json``.
    """
    return os.path.join(os.path.abspath(coco_dir), "annotations", f"instances_{split}.json")


def get_coco_images_dir(coco_dir: str, split: str) -> str:
    """
    Return the path to the images directory for a given split.

    Args:
        coco_dir: Root directory of the COCO export produced by export_coco.py.
        split: Split name — "train", "val", or "test".

    Returns:
        Absolute path to ``images/<split>/``.
    """
    return os.path.join(os.path.abspath(coco_dir), "images", split)
