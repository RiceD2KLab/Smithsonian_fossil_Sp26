"""
Helpers for YOLO in accessing COCO format dataset from export_coco.py.
"""

from __future__ import annotations

import json
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


def convert_coco_labels_to_yolo(coco_dir: str) -> None:
    """
    Convert COCO JSON annotations to YOLO txt labels.

    For each split (train/val/test), reads annotations/instances_<split>.json
    and writes labels/<split>/<image_stem>.txt with one line per annotation:

        class_id cx cy w h

    where cx, cy, w, h are normalized center-XYWH in [0, 1].
    Images with no annotations get an empty txt file so YOLO treats them as
    valid background images rather than missing labels.

    The conversion is skipped for a split if labels/<split>/ already
    contains the same number of .txt files as images in the JSON.

    Args:
        coco_dir: Root directory of the COCO export produced by export_coco.py.
    """
    coco_dir = os.path.abspath(coco_dir)

    for split in ("train", "val", "test"):
        ann_path = os.path.join(coco_dir, "annotations", f"instances_{split}.json")
        if not os.path.isfile(ann_path):
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        images = coco.get("images", [])
        if not images:
            continue

        labels_dir = os.path.join(coco_dir, "labels", split)

        # skip if .txt count already matches image count
        if os.path.isdir(labels_dir):
            existing = [name for name in os.listdir(labels_dir) if name.endswith(".txt")]
            if len(existing) == len(images):
                continue

        os.makedirs(labels_dir, exist_ok=True)

        # Build image_id -> (file_name, width, height)
        id_to_image: dict[int, tuple[str, int, int]] = {
            img["id"]: (img["file_name"], img["width"], img["height"])
            for img in images
        }

        # Group annotations by image_id
        anns_by_image: dict[int, list] = {img["id"]: [] for img in images}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in anns_by_image:
                anns_by_image[img_id].append(ann)

        written = 0
        for img_id, (file_name, img_w, img_h) in id_to_image.items():
            stem = os.path.splitext(os.path.basename(file_name))[0]
            txt_path = os.path.join(labels_dir, f"{stem}.txt")

            lines = []
            for ann in anns_by_image.get(img_id, []):
                x, y, w, h = ann["bbox"]
                class_id = ann["category_id"]
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
            written += 1

        print(f"Wrote {written} label files to labels/{split}/")

def validate_coco_dir(coco_dir: str) -> None:
    """
    Validate that the COCO export directory has the expected structure.
    """
    if not os.path.isdir(coco_dir):
        raise FileNotFoundError(f"coco_dir not found: {coco_dir}")
    yaml_path = get_coco_yaml_path(coco_dir)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f"dataset.yaml not found at {yaml_path}. "
            "Run export_coco.py first to generate the COCO dataset."
        )