"""
Shared FP/FN visualisation utilities used by both yolo26 and rfdetr inspect_fp_fn scripts.
"""

from __future__ import annotations

import numpy as np
import cv2
import h5py


MINIMUM_BOX_AREA_PX: int = 16
MINIMUM_ASPECT_RATIO: float = 0.25


def is_valid_bounding_box(box_width: float, box_height: float) -> bool:
    """Match H5YOLODataset exact filtering."""
    if box_width <= 0 or box_height <= 0:
        return False
    if box_width * box_height < MINIMUM_BOX_AREA_PX:
        return False
    if box_height < MINIMUM_ASPECT_RATIO * box_width:
        return False
    if box_width < MINIMUM_ASPECT_RATIO * box_height:
        return False
    return True


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
    Greedy IoU matching (mirrors Ultralytics ConfusionMatrix logic).
    Predictions must be sorted by confidence descending before calling.
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

    ious = iou[gi, pi]
    matches = np.column_stack((gi, pi, ious))

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


def _rgb_hwc_to_bgr(img: np.ndarray) -> np.ndarray:
    """Normalise an RGB array to (H, W, 3) uint8 BGR for OpenCV."""
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def load_image_and_labels_from_h5(
    h5_path: str, tile_key: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a single tile's images and raw labels from an H5 file.

    Returns both the precomputed focus-stacked image and the single sharpest
    focal plane (determined by ``focal_plane_ranking[0]``) so callers can
    annotate and save both outputs.

    Args:
        h5_path:  Absolute path to the ``.h5`` file.
        tile_key: H5 group key for the tile, e.g. ``"tile_3_7"``.

    Returns:
        img_stacked_bgr:    (H, W, 3) uint8 BGR — precomputed focus-stacked image.
        img_best_plane_bgr: (H, W, 3) uint8 BGR — sharpest focal plane from ``data``
                            via ``focal_plane_ranking[0]``.
        labels:             dict with:
                                "bboxes": (N, 4) float array — x, y, w, h in pixels
                                "labels": (N,) int array — class indices
    """
    with h5py.File(h5_path, "r") as f:
        group = f[tile_key]

        if "focus_stacked" not in group:
            raise KeyError(f"Tile group '{tile_key}' in {h5_path} has no 'focus_stacked' dataset")
        img_stacked_bgr = _rgb_hwc_to_bgr(group["focus_stacked"][:])

        if "focal_plane_ranking" not in group or "data" not in group:
            raise KeyError(
                f"Tile group '{tile_key}' in {h5_path} is missing "
                "'focal_plane_ranking' or 'data' dataset"
            )
        best_z = int(np.array(group["focal_plane_ranking"])[0])
        img_best_plane_bgr = _rgb_hwc_to_bgr(np.array(group["data"])[:, :, :, best_z])

        bboxes = np.array(group["bboxes"], dtype=np.float32) if "bboxes" in group else np.zeros((0, 4), dtype=np.float32)
        cls = np.array(group["labels"], dtype=np.int64) if "labels" in group else np.zeros((0,), dtype=np.int64)

        valid_indices = [
            i for i, (bbox, _) in enumerate(zip(bboxes, cls))
            if is_valid_bounding_box(bbox[2], bbox[3])
        ]

        if valid_indices:
            bboxes = bboxes[valid_indices]
            cls = cls[valid_indices]
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.int64)

        labels = {"bboxes": bboxes, "labels": cls}

    return img_stacked_bgr, img_best_plane_bgr, labels
