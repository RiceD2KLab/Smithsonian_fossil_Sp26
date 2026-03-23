"""
Shared FP/FN visualisation utilities used by both yolo26 and rfdetr inspect_fp_fn scripts.
"""

from __future__ import annotations

import numpy as np
import cv2

def draw_box(img: np.ndarray, box: list, label: str, color: tuple) -> None:
    """Draw a filled-background label + rectangle on *img* in-place (BGR)."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    top = max(y1 - th - 4, 0)
    cv2.rectangle(img, (x1, top), (x1 + tw, top + th + 4), color, -1)
    cv2.putText(img, label, (x1, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)


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