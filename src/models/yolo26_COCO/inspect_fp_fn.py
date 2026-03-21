"""
Post-processing script to visualise FP and FN detections after Ultralytics validation.

This avoids intercepting Ultralytics internals by reading what ``val`` already saved:
  - COCO predictions JSON (``predictions.json`` created by ``--save_json``)
  - Ground truth from the COCO annotations JSON produced by ``export_coco.py``
  - Images from the COCO images directory produced by ``export_coco.py``

``predictions.json`` uses plain image filenames as ``file_name``, matching
the filenames written by ``export_coco.py``, e.g.:

    "file_name": "slide01_tile_19829_76026.jpeg"

Run:
    python -m src.modeling.yolo26.inspect_fp_fn \\
        --val_dir runs/detect/yolo26_val \\
        --coco_dir data/coco_export \\
        --split test \\
        --iou_thresh 0.5 \\
        --conf_thresh 0.25 \\
        --max_images 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

from src.modeling.yolo26.utils import get_coco_annotations_path, get_coco_images_dir


# IoU and matching utilities

def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of XYXY bounding boxes.

    Args:
        a: Array of shape (N, 4) in XYXY format.
        b: Array of shape (M, 4) in XYXY format.

    Returns:
        IoU matrix of shape (N, M).
    """
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
    """
    Convert COCO XYWH boxes (x_topleft, y_topleft, w, h) to XYXY.

    Args:
        boxes: Array of shape (N, 4) in XYWH format.

    Returns:
        Array of shape (N, 4) in XYXY format.
    """
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
    Match predictions to ground-truth using greedy IoU matching.

    Implements the same logic as Ultralytics ConfusionMatrix. Assumes
    pred_boxes are already sorted by confidence descending.

    Args:
        pred_boxes: (N, 4) XYXY predicted boxes.
        pred_cls:   (N,) predicted class indices.
        gt_boxes:   (M, 4) XYXY ground-truth boxes.
        gt_cls:     (M,) ground-truth class indices.
        iou_thresh: IoU threshold for a match to count as TP.

    Returns:
        Tuple of (fp_mask, fn_mask):
            fp_mask: Boolean array of length N — True where prediction is a FP.
            fn_mask: Boolean array of length M — True where GT box is a FN.
    """
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    if n_pred == 0:
        return np.ones(0, dtype=bool), np.ones(n_gt, dtype=bool)
    if n_gt == 0:
        return np.ones(n_pred, dtype=bool), np.ones(0, dtype=bool)

    iou = box_iou_xyxy(gt_boxes, pred_boxes)  # (n_gt, n_pred)
    cls_match = gt_cls[:, None] == pred_cls[None, :]
    gi, pi = np.where((iou > iou_thresh) & cls_match)

    if len(gi) == 0:
        return np.ones(n_pred, dtype=bool), np.ones(n_gt, dtype=bool)

    matches = np.column_stack((gi, pi, iou[gi, pi]))

    if len(matches) > 1:
        matches = matches[matches[:, 2].argsort()[::-1]]
        _, idx = np.unique(matches[:, 1], return_index=True)
        matches = matches[idx]
        matches = matches[matches[:, 2].argsort()[::-1]]
        _, idx = np.unique(matches[:, 0], return_index=True)
        matches = matches[idx]

    matched_gt = set(matches[:, 0].astype(int))
    matched_pred = set(matches[:, 1].astype(int))

    fp_mask = np.array([i not in matched_pred for i in range(n_pred)])
    fn_mask = np.array([i not in matched_gt for i in range(n_gt)])
    return fp_mask, fn_mask


# COCO ground-truth loading

def load_coco_ground_truth(
    annotations_path: str,
) -> tuple[dict[int, str], dict[str, list[dict]]]:
    """
    Load ground-truth annotations from a COCO instances JSON file.

    Args:
        annotations_path: Path to ``instances_<split>.json``.

    Returns:
        Tuple of:
            image_id_to_filename: dict mapping image_id (int) → file_name (str).
            gt_by_filename:       dict mapping file_name → list of annotation dicts,
                                  each with keys ``bbox`` (XYWH) and ``category_id``.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    image_id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"] for img in coco["images"]
    }

    gt_by_filename: dict[str, list[dict]] = {img["file_name"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        file_name = image_id_to_filename.get(ann["image_id"])
        if file_name is not None:
            gt_by_filename[file_name].append(ann)

    return image_id_to_filename, gt_by_filename


def visualise(args: argparse.Namespace) -> None:
    """
    Load predictions and ground truth, match them, and save annotated FP/FN images.

    Args:
        args: Parsed command-line arguments.
    """
    # Resolve predictions.json path and output directory
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
                f"predictions.json not found at {pred_json_path}. "
                "Re-run val.py with --save_json or pass --predictions explicitly."
            )
        base_out_dir = Path(args.out_dir) if args.out_dir else val_dir

    with open(pred_json_path) as f:
        predictions = json.load(f)

    if not isinstance(predictions, list):
        raise TypeError(f"Expected predictions.json to be a list, got {type(predictions)}")
    if len(predictions) == 0:
        print("predictions.json is empty — no detections were saved.")
        return

    # Print sanity summary
    scores = [float(p.get("score", 0.0)) for p in predictions]
    cat_ids = sorted({int(p.get("category_id", 0)) for p in predictions})
    print("==== predictions.json summary ====")
    print(f"Total prediction entries: {len(predictions)}")
    print(f"Score range: min={min(scores):.4f}, max={max(scores):.4f}")
    print(f"Unique file_names: {len({p.get('file_name', '') for p in predictions})}")
    if cat_ids:
        print(f"Category_id values: {cat_ids}")
    print()

    # Load ground truth from COCO annotations JSON
    annotations_path = get_coco_annotations_path(args.coco_dir, args.split)
    if not Path(annotations_path).exists():
        raise FileNotFoundError(
            f"Annotations file not found: {annotations_path}. "
            f"Check --coco_dir and --split."
        )
    _, gt_by_filename = load_coco_ground_truth(annotations_path)

    images_dir = get_coco_images_dir(args.coco_dir, args.split)

    # Group predictions by file_name
    preds_by_filename: dict[str, list[dict]] = {}
    for p in predictions:
        preds_by_filename.setdefault(p["file_name"], []).append(p)

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

    for idx, (file_name, raw_preds) in enumerate(sorted(preds_by_filename.items())):
        image_path = Path(images_dir) / file_name
        if not image_path.exists():
            print(f"Warning: image not found at {image_path}")
            continue

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"Warning: could not read image {image_path}")
            continue

        # Build GT arrays from COCO annotations
        gt_anns = gt_by_filename.get(file_name, [])
        if gt_anns:
            gt_bboxes_xywh = np.array([a["bbox"] for a in gt_anns], dtype=np.float32)
            gt_cls = np.array([a["category_id"] for a in gt_anns], dtype=int)
            if args.single_cls:
                gt_cls = np.zeros_like(gt_cls)
            x, y, bw, bh = (
                gt_bboxes_xywh[:, 0], gt_bboxes_xywh[:, 1],
                gt_bboxes_xywh[:, 2], gt_bboxes_xywh[:, 3],
            )
            gt_boxes = np.stack([x, y, x + bw, y + bh], axis=1)
        else:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_cls = np.array([], dtype=int)

        total_gt += len(gt_boxes)

        # Build prediction arrays
        if raw_preds:
            pred_boxes = xywh_to_xyxy(
                np.array([p["bbox"] for p in raw_preds], dtype=np.float32)
            )
            pred_conf = np.array([p["score"] for p in raw_preds], dtype=np.float32)
            pred_cls = np.array([p["category_id"] for p in raw_preds], dtype=int)

            # Sort by confidence descending (required for greedy matching)
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

        image_id = str(raw_preds[0]["image_id"]) if raw_preds else "unknown"
        stem = Path(file_name).stem
        unique_prefix = f"{idx:06d}_{image_id}_{stem}"

        if has_fp and fp_saved < args.max_images:
            ann = Annotator(img_bgr.copy(), line_width=2, font_size=10)
            for pi in np.where(fp_mask)[0]:
                box = pred_boxes[pi].tolist()
                conf = float(pred_conf[pi])
                cls = int(pred_cls[pi])
                ann.box_label(box, f"FP cls{cls} {conf:.2f}", color=(0, 0, 220))
            cv2.imwrite(str(fp_dir / f"{unique_prefix}_fp{fp_count}.jpg"), ann.result())
            fp_saved += 1

        if has_fn and fn_saved < args.max_images:
            ann = Annotator(img_bgr.copy(), line_width=2, font_size=10)
            for gi in np.where(fn_mask)[0]:
                box = gt_boxes[gi].tolist()
                cls = int(gt_cls[gi])
                ann.box_label(box, f"FN cls{cls}", color=(0, 140, 255))
            cv2.imwrite(str(fn_dir / f"{unique_prefix}_fn{fn_count}.jpg"), ann.result())
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
    """
    Parse command-line arguments for FP/FN visualisation.

    Returns:
        Namespace with parsed arguments.
    """
    p = argparse.ArgumentParser(description="Visualise FP/FN detections from YOLO val predictions.")
    p.add_argument(
        "--predictions",
        help="Path to predictions.json. If omitted, uses --val_dir/predictions.json.",
    )
    p.add_argument(
        "--val_dir",
        help="Val output directory containing predictions.json. Only needed if --predictions is not given.",
    )
    p.add_argument(
        "--out_dir",
        help="Directory to write fp/ and fn/ images. Defaults to the directory containing predictions.json.",
    )
    p.add_argument(
        "--coco_dir",
        required=True,
        help="Root directory of the COCO export produced by export_coco.py.",
    )
    p.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="Which split was evaluated (used to locate the correct annotations and images).",
    )
    p.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for TP matching.")
    p.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold for predictions.")
    p.add_argument("--max_images", type=int, default=200, help="Maximum number of FP or FN images to save.")
    p.add_argument("--single_cls", action="store_true", help="Treat all ground-truth classes as 0.")
    return p.parse_args()


if __name__ == "__main__":
    visualise(parse_args())
