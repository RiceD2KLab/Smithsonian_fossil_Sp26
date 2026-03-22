"""
Post-processing script to visualise FP and FN detections after Ultralytics validation.

This avoids intercepting Ultralytics internals by reading what `val` already saved:
  - COCO predictions JSON (`predictions.json` created by `--save_json`)
  - Ground truth labels and images from your H5 dataset

Critically, it keys tiles by the exact `file_name` used in `predictions.json`, which in your runs
looks like:

    "file_name": "<slide_id>.h5::tile_30215_51559"

So we don't need to guess anything from `train_val_test.json` — we parse the H5 path and tile key
directly from `file_name`, and use `image_id` only for nicer output filenames.

Run:
    python -m src.modeling.yolo26.inspect_fp_fn \
        --val_dir runs/detect/yolo26_val \
        --h5_root data/tiles \
        --iou_thresh 0.5 \
        --conf_thresh 0.25 \
        --max_images 200

Notes / assumptions you may need to adapt:
  - `load_image_and_labels_from_h5()` assumes each tile group contains datasets:
        f[tile_key]["image"]  and  f[tile_key]["labels"]  (labels are YOLO: cls cx cy w h, normalised)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

from src.modeling.shared.fp_fn_utils import (
    box_iou_xyxy,
    greedy_match,
    is_valid_bounding_box,
    load_image_and_labels_from_h5,
    xywh_to_xyxy,
)


def visualise(args: argparse.Namespace) -> None:
    # Work out where predictions.json lives and where to write outputs.
    if args.predictions:
        pred_json_path = Path(args.predictions)
        if not pred_json_path.exists():
            raise FileNotFoundError(f"predictions.json not found at {pred_json_path}")
        base_out_dir = Path(args.out_dir) if args.out_dir else pred_json_path.parent
    else:
        if not args.val_dir:
            raise ValueError("Either --predictions or --val_dir must be provided.")
        val_dir = Path(args.val_dir)
        pred_json_path = val_dir / "predictions.json"
        if not pred_json_path.exists():
            raise FileNotFoundError(
                f"predictions.json not found at {pred_json_path}. Re-run val.py with --save_json "
                "or pass --predictions explicitly."
            )
        base_out_dir = Path(args.out_dir) if args.out_dir else val_dir

    with open(pred_json_path) as f:
        predictions = json.load(f)

    # Basic predictions.json sanity stats
    if not isinstance(predictions, list):
        raise TypeError(f"Expected predictions.json to be a list, got {type(predictions)}")
    num_preds = len(predictions)
    if num_preds == 0:
        print("predictions.json is empty – no detections were saved.")
        return
    scores = [float(p.get("score", 0.0)) for p in predictions]
    print("==== predictions.json summary ====")
    print(f"Total prediction entries: {num_preds}")
    print(f"Score range: min={min(scores):.4f}, max={max(scores):.4f}")
    file_names = {p.get("file_name", "") for p in predictions}
    image_ids = {p.get("image_id", "") for p in predictions}
    cat_ids = sorted({int(p.get("category_id", 0)) for p in predictions})
    print(f"Unique tiles (file_name): {len(file_names)}")
    print(f"Unique image_ids:          {len(image_ids)}")
    if cat_ids:
        print(f"Category_id values:        {cat_ids} (global offset = {cat_ids[0]})")
    print()

    # Decide global category_id offset so we align prediction classes with H5 labels.
    # Ultralytics COCO JSON typically uses category_id = class_index + 1, while our H5
    # labels are zero-based indices. We therefore subtract the global minimum
    # category_id from all predictions.
    global_cat_offset = cat_ids[0] if cat_ids else 0

    # Group predictions by exact tile file_name ("<slide>.h5::tile_x_y").
    preds_by_tile: dict[str, list[dict]] = {}
    for p in predictions:
        preds_by_tile.setdefault(p["file_name"], []).append(p)

    fp_dir = base_out_dir / "fp"
    fn_dir = base_out_dir / "fn"
    fp_dir.mkdir(exist_ok=True, parents=True)
    fn_dir.mkdir(exist_ok=True, parents=True)

    fp_saved = 0
    fn_saved = 0

    # Global stats for sanity-checking against Ultralytics metrics
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for idx, (file_name, raw_preds_for_tile) in enumerate(sorted(preds_by_tile.items())):
        try:
            h5_name, tile_key = file_name.split("::", 1)
        except ValueError:
            print(f"Warning: could not parse file_name '{file_name}'")
            continue

        h5_path = str(Path(args.h5_root) / h5_name)

        try:
            img_stacked_bgr, img_best_plane_bgr, labels = load_image_and_labels_from_h5(h5_path, tile_key)
        except Exception as e:
            print(f"Warning: could not load {h5_path}::{tile_key}: {e}")
            continue

        h, w = img_stacked_bgr.shape[:2]

        bboxes_px: np.ndarray = labels["bboxes"]
        cls_ids: np.ndarray = labels["labels"]
        if bboxes_px.size > 0:
            # Convert pixel xywh → xyxy
            x, y, bw, bh = (
                bboxes_px[:, 0],
                bboxes_px[:, 1],
                bboxes_px[:, 2],
                bboxes_px[:, 3],
            )
            gt_boxes = np.stack([x, y, x + bw, y + bh], axis=1)
            gt_cls = cls_ids.astype(int)
            if args.single_cls:
                gt_cls = np.zeros_like(gt_cls)
        else:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_cls = np.array([], dtype=int)

        total_gt += len(gt_boxes)

        if raw_preds_for_tile:
            pred_boxes_xywh = np.array([p["bbox"] for p in raw_preds_for_tile], dtype=np.float32)
            pred_boxes = xywh_to_xyxy(pred_boxes_xywh)
            pred_conf = np.array([p["score"] for p in raw_preds_for_tile], dtype=np.float32)
            # Map global category_id values back to zero-based class indices using the
            # global offset computed above.
            raw_cat = np.array([p["category_id"] for p in raw_preds_for_tile], dtype=int)
            pred_cls = raw_cat - global_cat_offset if raw_cat.size else raw_cat

            # Sort predictions by score descending for Ultralytics matching
            sort_idx = np.argsort(pred_conf)[::-1]
            pred_boxes = pred_boxes[sort_idx]
            pred_conf = pred_conf[sort_idx]
            pred_cls = pred_cls[sort_idx]

            keep = pred_conf >= args.conf_thresh
            pred_boxes = pred_boxes[keep]
            pred_conf = pred_conf[keep]
            pred_cls = pred_cls[keep]
        else:
            pred_boxes = np.zeros((0, 4))
            pred_conf = np.array([], dtype=np.float32)
            pred_cls = np.array([], dtype=int)

        total_pred += len(pred_boxes)

        fp_mask, fn_mask = greedy_match(pred_boxes, pred_cls, gt_boxes, gt_cls, args.iou_thresh)

        # Aggregate global counts
        fp_count = int(fp_mask.sum())
        fn_count = int(fn_mask.sum())
        tp_count = max(len(pred_boxes) - fp_count, 0)
        total_fp += fp_count
        total_fn += fn_count
        total_tp += tp_count

        has_fp = bool(fp_mask.any())
        has_fn = bool(fn_mask.any())
        if not has_fp and not has_fn:
            continue
        image_id = str(raw_preds_for_tile[0]["image_id"]) if raw_preds_for_tile else "unknown"
        safe_tile = tile_key.replace(":", "_")
        unique_prefix = f"{idx:06d}_{image_id}_{safe_tile}"

        if has_fp and fp_saved < args.max_images:
            for img_bgr, suffix in ((img_stacked_bgr, "stacked"), (img_best_plane_bgr, "best_plane")):
                ann = Annotator(img_bgr.copy(), line_width=2, font_size=10)
                for pi in np.where(fp_mask)[0]:
                    box = pred_boxes[pi].tolist()
                    conf = float(pred_conf[pi])
                    cls = int(pred_cls[pi])
                    ann.box_label(box, f"FP cls{cls} {conf:.2f}", color=(0, 0, 220))
                cv2.imwrite(
                    str(fp_dir / f"{unique_prefix}_fp{int(fp_mask.sum())}_{suffix}.jpg"),
                    ann.result(),
                )
            fp_saved += 1

        if has_fn and fn_saved < args.max_images:
            for img_bgr, suffix in ((img_stacked_bgr, "stacked"), (img_best_plane_bgr, "best_plane")):
                ann = Annotator(img_bgr.copy(), line_width=2, font_size=10)
                for gi in np.where(fn_mask)[0]:
                    box = gt_boxes[gi].tolist()
                    cls = int(gt_cls[gi])
                    ann.box_label(box, f"FN cls{cls}", color=(0, 140, 255))
                cv2.imwrite(
                    str(fn_dir / f"{unique_prefix}_fn{int(fn_mask.sum())}_{suffix}.jpg"),
                    ann.result(),
                )
            fn_saved += 1

    print("==== FP/FN summary (tile-level) ====")
    print(f"Total GT boxes:      {total_gt}")
    print(f"Total predictions:   {total_pred} (conf >= {args.conf_thresh})")
    print(f"Total TP:            {total_tp}")
    print(f"Total FP:            {total_fp}")
    print(f"Total FN:            {total_fn}")
    print()
    print(f"Saved {fp_saved} FP images → {fp_dir}")
    print(f"Saved {fn_saved} FN images → {fn_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise FP/FN from saved YOLO val JSON.")
    p.add_argument(
        "--predictions",
        help="Path to predictions.json. If omitted, use --val_dir/predictions.json.",
    )
    p.add_argument(
        "--val_dir",
        help="Val output dir (contains predictions.json). Only needed if --predictions "
        "is not given.",
    )
    p.add_argument(
        "--out_dir",
        help="Directory to write fp/fn images. Defaults to the directory containing "
        "predictions.json (or --val_dir).",
    )
    p.add_argument("--h5_root", required=True, help="Directory containing .h5 tile files.")
    p.add_argument("--iou_thresh", type=float, default=0.5)
    p.add_argument("--conf_thresh", type=float, default=0.25)
    p.add_argument("--max_images", type=int, default=200)
    p.add_argument("--single_cls", action="store_true", help="Treat all ground truth classes as 0")
    return p.parse_args()


if __name__ == "__main__":
    visualise(parse_args())
