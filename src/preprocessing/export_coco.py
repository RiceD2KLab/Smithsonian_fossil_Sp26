"""
Convert H5 tile dataset to COCO JSON + image files.

Different extraction modes: focus_stack, best_plane, plane N,
planes N1,N2, and all_planes. 

Outputs standard COCO JSON + images. 
Also writes a dataset.yaml (Ultralytics format) for direct YOLO use.

Output structure:
    output_dir/
      images/
        train/
        val/
        test/
      annotations/
        instances_train.json
        instances_val.json
        instances_test.json
      dataset.yaml

Usage:
    python -m src.preprocessing.export_coco \
        --h5_root data/tiles \
        --output_dir data/coco_export \
        --mode best_plane \
        --splits_json data/train_val_test.json \
        --workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from src.exploration.h5_utils import list_tile_jobs

# Bbox filtering

MINIMUM_BOX_AREA_PX: int = 16
MINIMUM_ASPECT_RATIO: float = 0.25


def _is_valid_bbox(w: float, h: float) -> bool:
    if w <= 0 or h <= 0:
        return False
    if w * h < MINIMUM_BOX_AREA_PX:
        return False
    if h < MINIMUM_ASPECT_RATIO * w:
        return False
    if w < MINIMUM_ASPECT_RATIO * h:
        return False
    return True


@dataclasses.dataclass
class ExportMode:
    kind: str               # focus_stack | best_plane | plane | planes | all_planes
    plane_indices: list[int]  # populated for plane/planes; empty for focus_stack/best_plane/all_planes


def build_export_mode(args: argparse.Namespace) -> ExportMode:
    kind = args.mode
    if kind == "plane":
        if args.plane_index is None:
            raise ValueError("--plane_index is required when --mode plane")
        return ExportMode(kind=kind, plane_indices=[args.plane_index])
    if kind == "planes":
        if not args.plane_indices:
            raise ValueError("--plane_indices is required when --mode planes")
        indices = [int(x.strip()) for x in args.plane_indices.split(",")]
        return ExportMode(kind=kind, plane_indices=indices)
    return ExportMode(kind=kind, plane_indices=[])


def _export_tile(
    h5_path: str,
    group_name: str,
    image_stem: str,
    split_dir: str,
    mode: ExportMode,
    image_format: str,
    single_cls: bool,
) -> list[dict]:
    """
    Extract image(s) for one tile and write them to split_dir.

    Returns a list of record dicts (one per image extracted):
        file_name, width, height, bboxes (list of [x,y,w,h] in pixels),
        labels (list of int category_ids), skipped (bool), skip_reason (str).
    """
    ext = f".{image_format}"
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 95] if image_format == "jpeg" else []

    try:
        raw_images: list[tuple[np.ndarray, Optional[int]]] = []  # (HWC, z_idx or None)

        # 1. Open the H5 file and extract the image arrays
        
        with h5py.File(h5_path, "r") as h5f:
            if group_name not in h5f:
                return [{"skipped": True, "skip_reason": f"group {group_name} missing in {h5_path}"}]
            grp = h5f[group_name]

            if mode.kind == "focus_stack":
                if "focus_stacked" not in grp:
                    return [{"skipped": True, "skip_reason": "focus_stacked dataset missing"}]
                raw_images = [(np.array(grp["focus_stacked"]), None)]

            elif mode.kind == "best_plane":
                if "data" not in grp or "focal_plane_ranking" not in grp:
                    return [{"skipped": True, "skip_reason": "data or focal_plane_ranking missing"}]
                best_z = int(np.array(grp["focal_plane_ranking"])[0])
                raw_images = [(np.array(grp["data"][:, :, :, best_z]), None)]

            else:  # plane | planes | all_planes
                if "data" not in grp:
                    return [{"skipped": True, "skip_reason": "data dataset missing"}]
                data = np.array(grp["data"])  # (H, W, C, Z)
                n_z = data.shape[3]
                if mode.kind == "all_planes":
                    z_list = list(range(n_z))
                else:
                    z_list = [z for z in mode.plane_indices if z < n_z]
                if not z_list:
                    return [{"skipped": True, "skip_reason": "no valid plane indices for this tile"}]
                raw_images = [(data[:, :, :, z], z) for z in z_list]

            bboxes_raw = (
                np.array(grp["bboxes"], dtype=np.float32)
                if "bboxes" in grp
                else np.zeros((0, 4), dtype=np.float32)
            )
            labels_raw = (
                np.array(grp["labels"], dtype=np.int64)
                if "labels" in grp
                else np.zeros((0,), dtype=np.int64)
            )

        # 2. Filter bboxes (same for every plane of this tile)
        
        valid_bboxes: list[list[float]] = []
        valid_labels: list[int] = []
        for bbox, label in zip(bboxes_raw.tolist(), labels_raw.tolist()):
            x, y, bw, bh = bbox
            if not _is_valid_bbox(bw, bh):
                continue
            cat_id = 1 if single_cls else int(label)
            valid_bboxes.append([float(x), float(y), float(bw), float(bh)])
            valid_labels.append(cat_id)

        # 3. Write images and build records
        multi_plane = mode.kind in ("plane", "planes", "all_planes")
        records: list[dict] = []

        for img_arr, z_idx in raw_images:
            if multi_plane and z_idx is not None:
                file_name = f"{image_stem}_{group_name}_z{z_idx:02d}{ext}"
            else:
                file_name = f"{image_stem}_{group_name}{ext}"

            out_path = os.path.join(split_dir, file_name)
            h, w = img_arr.shape[:2]

            if not os.path.exists(out_path):
                # H5 stores RGB; OpenCV writes BGR
                if img_arr.ndim == 3 and img_arr.shape[2] == 3:
                    bgr = cv2.cvtColor(img_arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
                else:
                    bgr = img_arr.astype(np.uint8)
                cv2.imwrite(out_path, bgr, jpeg_params)

            records.append({
                "file_name": file_name,
                "width": w,
                "height": h,
                "bboxes": valid_bboxes,
                "labels": valid_labels,
                "skipped": False,
                "skip_reason": "",
            })

        return records

    except Exception as exc:
        return [{"skipped": True, "skip_reason": f"exception: {exc}"}]

# COCO JSON helpers

def _build_categories(metadata_json_path: Optional[str], single_cls: bool) -> list[dict]:
    if single_cls:
        return [{"id": 1, "name": "palynomorph", "supercategory": "palynomorph"}]

    if metadata_json_path and os.path.isfile(metadata_json_path):
        with open(metadata_json_path) as f:
            meta = json.load(f)
        label_map: dict[str, int] = meta.get("label_map", {})
        if label_map:
            return [
                {"id": int(idx), "name": name, "supercategory": "palynomorph"}
                for name, idx in sorted(label_map.items(), key=lambda kv: kv[1])
            ]

    return [{"id": 1, "name": "palynomorph", "supercategory": "palynomorph"}]


def _assemble_coco_json(records: list[dict], categories: list[dict]) -> dict:
    images: list[dict] = []
    annotations: list[dict] = []
    img_id = 0
    ann_id = 0

    for record in records:
        if record.get("skipped"):
            continue
        images.append({
            "id": img_id,
            "file_name": record["file_name"],
            "width": record["width"],
            "height": record["height"],
        })
        for bbox, cat_id in zip(record["bboxes"], record["labels"]):
            x, y, bw, bh = bbox
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1
        img_id += 1

    return {
        "info": {"description": "Palynomorph dataset COCO export", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def run(
    h5_root: str,
    output_dir: str,
    mode: ExportMode,
    splits_json: Optional[str],
    workers: int,
    image_format: str,
    single_cls: bool,
    metadata_json: Optional[str],
) -> None:
    # 1. Discover tiles and partition them by split
    all_tiles = list_tile_jobs(h5_root)
    if not all_tiles:
        raise FileNotFoundError(f"No tiles found in {h5_root}")

    if splits_json:
        with open(splits_json) as f:
            splits_data: dict[str, list[str]] = json.load(f)
        split_names = [k for k in ("train", "val", "test") if k in splits_data]
        stem_to_split = {
            stem: split_name
            for split_name in split_names
            for stem in splits_data[split_name]
        }
        splits: dict[str, list[tuple[str, str, str]]] = {s: [] for s in split_names}
        for h5_path, stem, group in all_tiles:
            split = stem_to_split.get(stem)
            if split:
                splits[split].append((h5_path, stem, group))
    else:
        splits = {"all": all_tiles}

    # 2. Create output directories
    images_root = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    for split_name in splits:
        os.makedirs(os.path.join(images_root, split_name), exist_ok=True)

    categories = _build_categories(metadata_json, single_cls)

    # 3. Process each split
    for split_name, tiles in splits.items():
        if not tiles:
            print(f"[{split_name}] No tiles — skipping.")
            continue

        split_img_dir = os.path.join(images_root, split_name)
        print(f"\n[{split_name}] Exporting {len(tiles)} tiles → {split_img_dir}")

        all_records: list[dict] = []
        n_skipped = 0

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _export_tile,
                    h5_path, group_name, image_stem,
                    split_img_dir, mode, image_format, single_cls,
                ): (image_stem, group_name)
                for h5_path, image_stem, group_name in tiles
            }

            with tqdm(total=len(futures), unit="tile", desc=split_name) as pbar:
                for future in as_completed(futures):
                    records = future.result()
                    for rec in records:
                        if rec.get("skipped"):
                            n_skipped += 1
                        else:
                            all_records.append(rec)
                    pbar.update(1)

        print(f"[{split_name}] {len(all_records)} images, {n_skipped} skipped.")

        # 4. Assemble and atomically write COCO JSON
        coco_data = _assemble_coco_json(all_records, categories)
        ann_path = os.path.join(annotations_dir, f"instances_{split_name}.json")
        tmp_path = ann_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(coco_data, f)
        os.replace(tmp_path, ann_path)
        print(
            f"[{split_name}] → {ann_path} "
            f"({len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations)"
        )

    # 5. Write dataset.yaml (Ultralytics YOLO format)
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    lines = [f"path: {os.path.abspath(output_dir)}"]
    for sn in ("train", "val", "test"):
        if sn in splits:
            lines.append(f"{sn}: images/{sn}")
    if "all" in splits:
        lines.append("train: images/all")
    cat_names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
    lines.append(f"nc: {len(cat_names)}")
    lines.append(f"names: {cat_names}")
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {yaml_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export H5 palynomorph tile dataset to COCO JSON + image files."
    )
    parser.add_argument("--h5_root", required=True, help="Source H5 directory")
    parser.add_argument("--output_dir", required=True, help="Output root directory")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["focus_stack", "best_plane", "plane", "planes", "all_planes"],
        help=(
            "Image extraction mode: "
            "focus_stack=precomputed focus stack, "
            "best_plane=sharpest Z-plane via focal_plane_ranking, "
            "plane=single Z-plane (requires --plane_index), "
            "planes=selected Z-planes (requires --plane_indices), "
            "all_planes=every Z-plane"
        ),
    )
    parser.add_argument(
        "--plane_index",
        type=int,
        default=None,
        help="Z-plane index (required when --mode plane)",
    )
    parser.add_argument(
        "--plane_indices",
        type=str,
        default=None,
        help='Comma-separated Z-plane indices, e.g. "0,5,10" (required when --mode planes)',
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default=None,
        help="Path to train_val_test.json; if absent, exports all tiles under 'all/'",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--image_format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Output image format (default: jpeg)",
    )
    parser.add_argument(
        "--single_cls",
        action="store_true",
        help="Collapse all category labels to category_id=1",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        default=None,
        help="Optional path to metadata.json with label_map for multi-class category names",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = build_export_mode(args)
    run(
        h5_root=args.h5_root,
        output_dir=args.output_dir,
        mode=mode,
        splits_json=args.splits_json,
        workers=args.workers,
        image_format=args.image_format,
        single_cls=args.single_cls,
        metadata_json=args.metadata_json,
    )


if __name__ == "__main__":
    main()
