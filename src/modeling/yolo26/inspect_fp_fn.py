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
import h5py
import numpy as np
from ultralytics.utils.plotting import Annotator


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """COCO xywh (x_topleft, y_topleft, w, h) → xyxy."""
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def greedy_match(
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    iou_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact implementation of Ultralytics ConfusionMatrix matching logic.
    Assumes pred_boxes are already sorted by confidence descending.
    """
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    if n_pred == 0:
        return np.ones(0, dtype=bool), np.ones(n_gt, dtype=bool)
    if n_gt == 0:
        return np.ones(n_pred, dtype=bool), np.ones(0, dtype=bool)

    iou = box_iou_xyxy(gt_boxes, pred_boxes)  # shape: (n_gt, n_pred)
    # where iou > thresh, taking class match into account as well
    cls_match = gt_cls[:, None] == pred_cls[None, :]
    gi, pi = np.where((iou > iou_thresh) & cls_match)
    
    if len(gi) == 0:
        return np.ones(n_pred, dtype=bool), np.ones(n_gt, dtype=bool)

    ious = iou[gi, pi]
    matches = np.column_stack((gi, pi, ious))
    
    if len(matches) > 1:
        # sort by IoU descending
        matches = matches[matches[:, 2].argsort()[::-1]]
        # unique on predictions (keep highest IoU for each prediction)
        _, idx = np.unique(matches[:, 1], return_index=True)
        matches = matches[idx]
        # sort by IoU again
        matches = matches[matches[:, 2].argsort()[::-1]]
        # unique on GT (keep highest IoU for each GT)
        _, idx = np.unique(matches[:, 0], return_index=True)
        matches = matches[idx]
    
    matched_gt = set(matches[:, 0].astype(int))
    matched_pred = set(matches[:, 1].astype(int))

    fp_mask = np.array([i not in matched_pred for i in range(n_pred)])
    fn_mask = np.array([i not in matched_gt for i in range(n_gt)])
    return fp_mask, fn_mask


def is_valid_bounding_box(box_width: float, box_height: float) -> bool:
    """Match H5YOLODataset exact filtering."""
    if box_width <= 0 or box_height <= 0:
        return False
    if box_width * box_height < 16:
        return False
    if box_height < 0.25 * box_width:
        return False
    if box_width < 0.25 * box_height:
        return False
    return True


def load_image_and_labels_from_h5(h5_path: str, tile_key: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single tile's image and raw labels from an H5 file.

    Returns:
        img_bgr: (H, W, 3) uint8 BGR numpy array
        labels:  dict with:
            - "bboxes": (N, 4) float array — x, y, w, h in pixels
            - "labels": (N,) int array — class indices
    """
    with h5py.File(h5_path, "r") as f:
        group = f[tile_key]

        # Match the training dataset convention: "focus_stacked", "bboxes", "labels"
        if "focus_stacked" not in group:
            raise KeyError(f"Tile group '{tile_key}' in {h5_path} has no 'focus_stacked' dataset")
        img = group["focus_stacked"][:]  # (H, W, C) RGB
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))  # CHW → HWC
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        bboxes = np.array(group["bboxes"], dtype=np.float32) if "bboxes" in group else np.zeros((0, 4), dtype=np.float32)
        cls = np.array(group["labels"], dtype=np.int64) if "labels" in group else np.zeros((0,), dtype=np.int64)
        
        # Filter matching H5YOLODataset logic
        valid_indices = []
        for i, (bbox, c) in enumerate(zip(bboxes, cls)):
            # bbox is [x, y, w, h]
            if is_valid_bounding_box(bbox[2], bbox[3]):
                valid_indices.append(i)
                
        if valid_indices:
            bboxes = bboxes[valid_indices]
            cls = cls[valid_indices]
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.int64)

        labels = {"bboxes": bboxes, "labels": cls}

    return img_bgr, labels


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
            img_bgr, labels = load_image_and_labels_from_h5(h5_path, tile_key)
        except Exception as e:
            print(f"Warning: could not load {h5_path}::{tile_key}: {e}")
            continue

        h, w = img_bgr.shape[:2]

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
            ann = Annotator(img_bgr.copy(), line_width=2, font_size=10)
            for pi in np.where(fp_mask)[0]:
                box = pred_boxes[pi].tolist()
                conf = float(pred_conf[pi])
                cls = int(pred_cls[pi])
                ann.box_label(box, f"FP cls{cls} {conf:.2f}", color=(0, 0, 220))
            cv2.imwrite(str(fp_dir / f"{unique_prefix}_fp{int(fp_mask.sum())}.jpg"), ann.result())
            fp_saved += 1

        if has_fn and fn_saved < args.max_images:
            ann = Annotator(img_bgr.copy(), line_width=2, font_size=10)
            for gi in np.where(fn_mask)[0]:
                box = gt_boxes[gi].tolist()
                cls = int(gt_cls[gi])
                ann.box_label(box, f"FN cls{cls}", color=(0, 140, 255))
            cv2.imwrite(str(fn_dir / f"{unique_prefix}_fn{int(fn_mask.sum())}.jpg"), ann.result())
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
