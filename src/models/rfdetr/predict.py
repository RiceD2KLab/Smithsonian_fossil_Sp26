"""
RF-DETR inference and visualisation on image tiles.

Loads a trained RF-DETR checkpoint, runs inference on a flat directory of
images, and saves annotated outputs. Ground-truth, FP, and FN modes are
shown when a COCO annotation file is provided.

COMMAND (typical usage):

  # predictions only
  python -m src.models.rfdetr.predict \\
      --weights runs/rfdetr/best.pt \\
      --image_dir data/coco_export/images/test/ \\
      --out_dir runs/rfdetr/predictions/
      --conf_thresh 0.5 \\ (confidence threshold for predictions)
      --resolution 1008 \\ (must be divisible by 56..rfdetr backbone req)
      --iou_thresh 0.5 \\ (for FP/FN matching)
      --max_images 0 \\ (0 is no limit)

  # predictions + GT overlay + FP/FN breakdown
  python -m src.models.rfdetr.predict \\
      --weights runs/rfdetr/best.pt \\
      --image_dir data/coco_export/images/test/ \\
      --out_dir runs/rfdetr/predictions/ \\
      --coco_ann data/coco_export/annotations/instances_test.json \\
      --show_gt \\
      --show_fp_fn \\
      --conf_thresh 0.5 \\ (confidence threshold for predictions)
      --resolution 1008 \\ (must be divisible by 56..rfdetr backbone req)
      --iou_thresh 0.5 \\ (for FP/FN matching)
      --max_images 0 \\ (0 is no limit)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.models.rfdetr.train import _MODEL_CLASSES
from src.models.utils import greedy_match, xywh_to_xyxy, draw_box

# export_coco.py only exports JPEG and PNG images.
_IMG_EXTENSIONS = {".jpeg", ".png"}

# BGR colours for annotated output images.
_COL_PRED = (0, 200, 0)       # green  — model predictions
_COL_GT   = (200, 0, 0)       # blue   — ground-truth boxes
_COL_FP   = (0, 0, 220)       # red    — false positives
_COL_FN   = (0, 140, 255)     # orange — false negatives


def _load_coco_gt(ann_file: str) -> tuple[dict[str, list[dict]], dict[int, str]]:
    """Parse a COCO annotation JSON into per-filename (key) -> list of GT dicts (value).

    Args:
        ann_file: Path to a COCO-format annotation JSON file that contains
            images and annotations keys.

    Returns:
        A two-element tuple:

        - gt_by_filename (dict[str, list[dict]]): Maps each image
          file_name to the list of its COCO annotation dicts. All image filenames from coco["images"]
           are present as keys, even if they have no annotations.
        - id_to_filename (dict[int, str]): Maps COCO image id to file_name.
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
    """Convert a list of COCO annotation dicts to NumPy arrays for matching.

    Args:
        gt_anns: List of COCO annotation dicts, each containing a
            bbox key with [x, y, w, h] values in pixels.

    Returns:
        A two-element tuple:

        - boxes (np.ndarray, shape (N, 4), dtype float32): Bounding
          boxes in [x1, y1, x2, y2] format.
        - classes (np.ndarray, shape (N,), dtype int): All-zeros
          class array (single-class detection).

        Both arrays have length 0 when gt_anns is empty.
    """
    if not gt_anns:
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=int)
    boxes_xywh = np.array([a["bbox"] for a in gt_anns], dtype=np.float32)
    return xywh_to_xyxy(boxes_xywh), np.zeros(len(gt_anns), dtype=int)


def predict(args: argparse.Namespace) -> None:
    """Run RF-DETR inference on a flat image directory and save annotated outputs.

    For each image, the model is queried and predicted boxes are drawn in
    green and saved to predictions/.  When coco_ann is provided,
    the following overlays are also generated:

    - ground_truth/ (requires --show_gt): images with GT boxes in blue.
    - fp/ (requires --show_fp_fn): images containing at least one false
      positive, with FP boxes in red.
    - fn/ (requires --show_fp_fn): images containing at least one false
      negative, with FN boxes in orange.

    FP/FN matching uses greedy IoU matching (confidence-descending order)
    with the threshold from --iou_thresh.

    Args:
        args: Parsed argparse.Namespace from parse_args():
            supplies the required fields: weights, image_dir, out_dir, model,
            conf_thresh, resolution, iou_thresh, max_images.
            Optional: coco_ann, show_gt, show_fp_fn.

    Raises:
        ValueError: If show_gt or show_fp_fn is set without coco_ann.
        FileNotFoundError: If image_dir or coco_ann does not exist.
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

    # Load ground truth from COCO annotation JSON (if provided).
    gt_by_filename: dict[str, list[dict]] = {}
    if args.coco_ann:
        if not Path(args.coco_ann).exists():
            raise FileNotFoundError(f"COCO annotation JSON not found: {args.coco_ann}")
        gt_by_filename, _ = _load_coco_gt(args.coco_ann)

    # Load model from checkpoint.
    print(f"Loading {args.model} model from {args.weights} checkpoint ...")
    model = _MODEL_CLASSES[args.model](
        pretrain_weights=args.weights,
        num_classes=1,
        resolution=args.resolution,
    )
    model.optimize_for_inference()

    img_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in _IMG_EXTENSIONS
    )
    if args.max_images:
        img_paths = img_paths[: args.max_images]

    if not img_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Running RF-DETR inference on {len(img_paths)} images ...")

    total_pred = total_gt = total_tp = total_fp_count = total_fn_count = 0
    pred_saved = gt_saved = fp_saved = fn_saved = 0

    for img_path in img_paths:
        file_name = img_path.name

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: could not read image {img_path}")
            continue

        # --- Phase 1: run inference ---
        detections = model.predict(str(img_path), threshold=args.conf_thresh)

        pred_boxes = detections.xyxy
        pred_conf  = detections.confidence if detections.confidence is not None else np.array([])

        if pred_boxes is None or len(pred_boxes) == 0:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_conf  = np.array([], dtype=np.float32)

        pred_cls = np.zeros(len(pred_boxes), dtype=int)
        total_pred += len(pred_boxes)

        # --- Phase 2: save prediction overlay ---
        img_pred = img_bgr.copy()
        for box, conf in zip(pred_boxes, pred_conf):
            draw_box(img_pred, box.tolist(), f"{float(conf):.2f}", _COL_PRED)
        cv2.imwrite(str(pred_dir / file_name), img_pred)
        pred_saved += 1

        # --- Phase 3: GT overlay and FP/FN matching (requires --coco_ann) ---
        gt_anns = gt_by_filename.get(file_name, [])
        gt_boxes, gt_cls = _gt_arrays(gt_anns)
        total_gt += len(gt_boxes)

        if gt_dir is not None and gt_anns:
            img_gt = img_bgr.copy()
            for box in gt_boxes:
                draw_box(img_gt, box.tolist(), "GT", _COL_GT)
            cv2.imwrite(str(gt_dir / file_name), img_gt)
            gt_saved += 1

        if fp_dir is not None and fn_dir is not None:
            # Sort predictions by confidence descending before matching.
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
    print("==== RF-DETR prediction summary ====")
    print(f"Images processed:    {pred_saved}")
    print(f"Total predictions:   {total_pred} (conf >= {args.conf_thresh})")
    if args.coco_ann:
        print(f"Total GT boxes:      {total_gt}")
    if args.show_fp_fn:
        print(f"Total TP:            {total_tp}")
        print(f"Total FP:            {total_fp_count}")
        print(f"Total FN:            {total_fn_count}")
    print()
    print(f"Saved {pred_saved} prediction images -> {pred_dir}")
    if gt_dir:
        print(f"Saved {gt_saved} ground truth images -> {gt_dir}")
    if fp_dir:
        print(f"Saved {fp_saved} FP images -> {fp_dir}")
        print(f"Saved {fn_saved} FN images -> {fn_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RF-DETR inference.

    Returns:
        argparse.Namespace with the following attributes:

        - weights (str): Path to trained RF-DETR checkpoint (.pt or .pth).
        - image_dir (str): Flat directory of input images.
        - out_dir (str): Root output directory.
        - model (str): RF-DETR variant key (must match the checkpoint).
        - conf_thresh (float): Minimum confidence score for a detection.
        - resolution (int): Inference resolution in pixels (must match
          training resolution and be divisible by 56).
        - coco_ann (str | None): Optional COCO annotation JSON path.
        - show_gt (bool): Save GT overlay images (requires ``coco_ann``).
        - show_fp_fn (bool): Save FP/FN images (requires ``coco_ann``).
        - iou_thresh (float): IoU threshold for FP/FN matching.
        - max_images (int): Maximum images to process (0 = no limit).
    """
    parser = argparse.ArgumentParser(
        description="RF-DETR inference and visualisation on image tiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", required=True,
                        help="Path to trained RF-DETR checkpoint (.pt).")
    parser.add_argument("--image_dir", required=True,
                        help="Flat directory of input images.")
    parser.add_argument("--out_dir", required=True,
                        help="Root output directory.")
    parser.add_argument("--model", choices=list(_MODEL_CLASSES.keys()), default="base",
                        help="RF-DETR model variant (default: base).")
    parser.add_argument("--conf_thresh", type=float, required=True,
                        help="Confidence threshold.")
    parser.add_argument("--resolution", type=int, required=True,
                        help="Inference resolution in pixels.")
    parser.add_argument("--coco_ann",
                        help="Optional COCO annotation JSON. Required for --show_gt / --show_fp_fn.")
    parser.add_argument("--show_gt", action="store_true",
                        help="Save separate images with ground-truth boxes (requires --coco_ann).")
    parser.add_argument("--show_fp_fn", action="store_true",
                        help="Save FP and FN images (requires --coco_ann).")
    parser.add_argument("--iou_thresh", type=float, required=True,
                        help="IoU threshold for FP/FN matching.")
    parser.add_argument("--max_images", type=int, required=True,
                        help="Limit number of images processed (0 = no limit).")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
