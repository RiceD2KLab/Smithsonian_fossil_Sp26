from __future__ import annotations

import numpy as np

from src.annotator.types import Detection


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two arrays of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(ix2 - ix1, 0)
    inter_h = np.maximum(iy2 - iy1, 0)
    inter = inter_w * inter_h

    area_a = np.maximum(ax2 - ax1, 0) * np.maximum(ay2 - ay1, 0)
    area_b = np.maximum(bx2 - bx1, 0) * np.maximum(by2 - by1, 0)

    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def non_max_suppression(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    """Class-agnostic NMS for already scored detections."""
    if not detections:
        return []

    boxes = np.array(
        [[d.x1_px, d.y1_px, d.x2_px, d.y2_px] for d in detections],
        dtype=np.float32,
    )
    scores = np.array([d.score for d in detections], dtype=np.float32)

    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while len(order) > 0:
        current = int(order[0])
        keep.append(current)

        if len(order) == 1:
            break

        rest = order[1:]
        iou = box_iou_xyxy(boxes[current : current + 1], boxes[rest])[0]
        order = rest[iou <= iou_threshold]

    return [detections[i] for i in keep]
