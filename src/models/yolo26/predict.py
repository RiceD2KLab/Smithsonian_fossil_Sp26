"""
YOLO inference and visualisation on image tiles.

Loads a trained YOLO checkpoint, runs inference on a flat directory of
images, and saves annotated outputs.  Ground-truth, FP, and FN modes are
available when a COCO annotation file is provided.

COMMAND (typical usage):

  # predictions only
  python -m src.models.yolo26.predict \\
      --weights runs/detect/best.pt \\
      --image_dir data/coco_export/images/test/ \\
      --out_dir runs/detect/predictions/ \\
      --conf_thresh 0.5 \\
      --imgsz 640 \\
      --iou_thresh 0.5 \\
      --max_images 0

  # predictions + GT overlay + FP/FN breakdown
  python -m src.models.yolo26.predict \\
      --weights runs/detect/best.pt \\
      --image_dir data/coco_export/images/test/ \\
      --out_dir runs/detect/predictions/ \\
      --coco_ann data/coco_export/annotations/instances_test.json \\
      --show_gt \\
      --show_fp_fn \\
      --conf_thresh 0.5 \\
      --imgsz 640 \\
      --iou_thresh 0.5 \\
      --max_images 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.models.utils import greedy_match, xywh_to_xyxy, draw_box

# export_coco.py only exports JPEG and PNG images
_IMG_EXTENSIONS = {".jpeg", ".jpg", ".png"}

# BGR colours
_COL_PRED = (0, 200, 0)    # green
_COL_GT   = (200, 0, 0)    # blue
_COL_FP   = (0, 0, 220)    # red
_COL_FN   = (0, 140, 255)  # orange


def _load_coco_gt(ann_file: str) -> tuple[dict[str, list[dict]], dict[int, str]]:
    """Parse a COCO annotation JSON into per-filename ground-truth dictionaries.

    Args:
        ann_file: Path to a COCO-format annotation JSON file (e.g.
            annotations/instances_test.json).

    Returns:
        A two-element tuple:
        - gt_by_filename: mapping from image filename (str) to a list of
          COCO annotation dicts for that image.
        - id_to_filename: mapping from COCO image ID (int) to filename (str).
    """
    with open(ann_file) as f:
        coco = json.load(f)

    id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"] for img in coco["images"]
    }
    gt_by_filename: dict[str, list[dict]] = {fn: [] for fn in id_to_filename.values()}
    for ann in coco["annotations"]:
        fn = id_to_filename.get(ann["image_id"])
        if fn:
            gt_by_filename[fn].append(ann)

    return gt_by_filename, id_to_filename


def _gt_arrays(
    gt_anns: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of COCO annotations to xyxy box and class arrays.

    Args:
        gt_anns: List of COCO annotation dicts, each containing a bbox key
            key with [x, y, w, h] values in absolute pixel coordinates.

    Returns:
        A two-element tuple:
        - boxes_xyxy: float32 array of shape (N, 4) with boxes in
          [x1, y1, x2, y2] format. Empty array of shape (0, 4) when gt_anns is empty.
        - classes: int array of shape (N,) filled with zeros (single class). Empty array when gt_anns is empty.
    """
    if not gt_anns:
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=int)
    boxes_xywh = np.array([a["bbox"] for a in gt_anns], dtype=np.float32)
    return xywh_to_xyxy(boxes_xywh), np.zeros(len(gt_anns), dtype=int)


def predict(args: argparse.Namespace) -> None:
    """Run YOLO inference and save annotated prediction, GT, FP, and FN images.

    Iterates over all images in args.image_dir, runs the YOLO model, and
    writes annotated copies to sub-directories of args.out_dir. When a
    COCO annotation file is provided the function also computes TP/FP/FN counts
    via greedy IoU matching and saves per-category breakdown images.

    Args:
        args: Parsed argparse.Namespace from parse_prediction_arguments():
            Expected attributes: weights, image_dir, out_dir, conf_thresh,
            imgsz, coco_ann (optional), show_gt, show_fp_fn, iou_thresh,
            max_images.
            Optional: coco_ann, show_gt, show_fp_fn.

    Returns:
        None

    Raises:
        ValueError: If show_gt or show_fp_fn is set without
            providing coco_ann.
        FileNotFoundError: If image_dir or coco_ann (when provided)
            does not exist on disk.
    """
    if (args.show_gt or args.show_fp_fn) and not args.coco_ann:
        raise ValueError("--show_gt and --show_fp_fn require --coco_ann.")

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = fp_dir = fn_dir = None
    if args.show_gt:
        gt_dir = out_dir / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
    if args.show_fp_fn:
        fp_dir = out_dir / "fp"
        fn_dir = out_dir / "fn"
        fp_dir.mkdir(parents=True, exist_ok=True)
        fn_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth from COCO annotation JSON
    gt_by_filename: dict[str, list[dict]] = {}
    if args.coco_ann:
        if not Path(args.coco_ann).exists():
            raise FileNotFoundError(f"COCO annotation JSON not found: {args.coco_ann}")
        gt_by_filename, _ = _load_coco_gt(args.coco_ann)

    # Load model from checkpoint
    print(f"Loading YOLO model from {args.weights} …")
    model = YOLO(args.weights)

    # Collect images from input directory
    img_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in _IMG_EXTENSIONS
    )
    if args.max_images:
        img_paths = img_paths[: args.max_images]

    if not img_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Running YOLO inference on {len(img_paths)} images …")

    total_pred = total_gt = total_tp = total_fp_count = total_fn_count = 0
    pred_saved = gt_saved = fp_saved = fn_saved = 0

    for img_path in img_paths:
        file_name = img_path.name

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: could not read image {img_path}")
            continue

        results = model.predict(
            source=str(img_path),
            conf=args.conf_thresh,
            imgsz=args.imgsz,
            verbose=False,
        )

        boxes_tensor = results[0].boxes
        if boxes_tensor is not None and len(boxes_tensor) > 0:
            pred_boxes = boxes_tensor.xyxy.cpu().numpy().astype(np.float32)
            pred_conf  = boxes_tensor.conf.cpu().numpy().astype(np.float32)
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_conf  = np.array([], dtype=np.float32)

        pred_cls = np.zeros(len(pred_boxes), dtype=int)
        total_pred += len(pred_boxes)

        # Save predictions
        img_pred = img_bgr.copy()
        for box, conf in zip(pred_boxes, pred_conf):
            draw_box(img_pred, box.tolist(), f"{float(conf):.2f}", _COL_PRED)
        cv2.imwrite(str(pred_dir / file_name), img_pred)
        pred_saved += 1

        # Get ground truth boxes and classes
        gt_anns = gt_by_filename.get(file_name, [])
        gt_boxes, gt_cls = _gt_arrays(gt_anns)
        total_gt += len(gt_boxes)

        # Save ground truth
        if gt_dir is not None and gt_anns:
            img_gt = img_bgr.copy()
            for box in gt_boxes:
                draw_box(img_gt, box.tolist(), "GT", _COL_GT)
            cv2.imwrite(str(gt_dir / file_name), img_gt)
            gt_saved += 1

        # Get FP and FN masks
        if fp_dir is not None and fn_dir is not None:
            # Sort by confidence descending before matching
            if len(pred_boxes) > 0:
                order = np.argsort(pred_conf)[::-1]
                pred_boxes_s = pred_boxes[order]
                pred_conf_s  = pred_conf[order]
                pred_cls_s   = pred_cls[order]
            else:
                pred_boxes_s = pred_boxes
                pred_conf_s  = pred_conf
                pred_cls_s   = pred_cls

            fp_mask, fn_mask = greedy_match(
                pred_boxes_s, pred_cls_s, gt_boxes, gt_cls, args.iou_thresh
            )

            fp_count = int(fp_mask.sum()) if fp_mask.size else 0
            fn_count = int(fn_mask.sum()) if fn_mask.size else 0
            tp_count = max(len(pred_boxes_s) - fp_count, 0)
            total_fp_count += fp_count
            total_fn_count += fn_count
            total_tp        += tp_count

            if fp_mask.any():
                img_fp = img_bgr.copy()
                for pi in np.where(fp_mask)[0]:
                    draw_box(img_fp, pred_boxes_s[pi].tolist(),
                              f"FP {float(pred_conf_s[pi]):.2f}", _COL_FP)
                cv2.imwrite(str(fp_dir / file_name), img_fp)
                fp_saved += 1

            if fn_mask.size and fn_mask.any():
                img_fn = img_bgr.copy()
                for gi in np.where(fn_mask)[0]:
                    draw_box(img_fn, gt_boxes[gi].tolist(), "FN", _COL_FN)
                cv2.imwrite(str(fn_dir / file_name), img_fn)
                fn_saved += 1

    print()
    print("==== YOLO prediction summary ====")
    print(f"Images processed:    {pred_saved}")
    print(f"Total predictions:   {total_pred} (conf >= {args.conf_thresh})")
    if args.coco_ann:
        print(f"Total GT boxes:      {total_gt}")
    if args.show_fp_fn:
        print(f"Total TP:            {total_tp}")
        print(f"Total FP:            {total_fp_count}")
        print(f"Total FN:            {total_fn_count}")
    print()
    print(f"Saved {pred_saved} prediction images → {pred_dir}")
    if gt_dir:
        print(f"Saved {gt_saved} ground truth images → {gt_dir}")
    if fp_dir:
        print(f"Saved {fp_saved} FP images → {fp_dir}")
        print(f"Saved {fn_saved} FN images → {fn_dir}")


def parse_prediction_arguments() -> argparse.Namespace:
    """Parse command-line arguments for YOLO inference and visualisation.

    Returns:
        argparse.Namespace with the following attributes:

        - weights (str): Path to trained YOLO checkpoint (.pt).
        - image_dir (str): Flat directory of input images.
        - out_dir (str): Root output directory.
        - conf_thresh (float): Confidence threshold (default: 0.5).
        - imgsz (int): Inference image size in pixels (default: 640).
        - coco_ann (str | None): Optional COCO annotation JSON path.
        - show_gt (bool): Save ground-truth overlay images.
        - show_fp_fn (bool): Save FP and FN breakdown images.
        - iou_thresh (float): IoU threshold for FP/FN matching (default: 0.5).
        - max_images (int): Max images to process; 0 means no limit.
    """
    parser = argparse.ArgumentParser(
        description="YOLO inference and visualisation on image tiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", required=True,
                        help="Path to trained YOLO checkpoint (.pt).")
    parser.add_argument("--image_dir", required=True,
                        help="Flat directory of input images.")
    parser.add_argument("--out_dir", required=True,
                        help="Root output directory.")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size in pixels (default: 640).")
    parser.add_argument("--coco_ann",
                        help="Optional COCO annotation JSON. Required for --show_gt / --show_fp_fn.")
    parser.add_argument("--show_gt", action="store_true",
                        help="Save separate images with ground-truth boxes (requires --coco_ann).")
    parser.add_argument("--show_fp_fn", action="store_true",
                        help="Save FP and FN images (requires --coco_ann).")
    parser.add_argument("--iou_thresh", type=float, default=0.5,
                        help="IoU threshold for FP/FN matching (default: 0.5).")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Limit number of images processed (0 = no limit).")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_prediction_arguments())
