"""
Compare predicted NDPA annotations against ground truth and report detection metrics.

Reads palynomorph circles from both NDPA files, matches predictions to ground-truth
annotations by bounding-box IoU in nanometer space (greedy matching in descending
confidence order), and prints precision, recall, F1, and average precision
(area under the PR curve) to stdout.

Usage:
    python -m src.annotator.evaluate_ndpa \\
        --gt_ndpa /path/to/gt.ndpi.ndpa \\
        --pred_ndpa /path/to/pred.ndpi.ndpa \\
        [--iou_threshold 0.5]
"""

import argparse
from dataclasses import dataclass

from src.data.ndpa_reader import NDPAData, PalynomorphAnnotation


@dataclass(frozen=True)
class _Box:
    """Axis-aligned bounding box in nanometers.

    Attributes:
        x1: Left coordinate.
        y1: Top coordinate.
        x2: Right coordinate.
        y2: Bottom coordinate.
    """
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class _Pred:
    """Prediction with a bounding box and confidence score.

    Attributes:
        box: Bounding box.
        confidence: Confidence score.
    """
    box: _Box
    confidence: float


def _to_box(ann: PalynomorphAnnotation) -> _Box:
    """Convert a palynomorph annotation to an axis-aligned bounding box in nm.

    Args:
        ann: A palynomorph annotation parsed from an NDPAData instance.

    Returns:
        A _Box with (x1, y1, x2, y2) in nanometers.
    """
    x, y, w, h = ann.bounds.get_bounding_box()
    return _Box(x, y, x + w, y + h)


def _parse_confidence(details: str) -> float:
    """Extract the confidence score from an annotation details string.

    Args:
        details: Free-text details string, e.g. "source=yolo; confidence=0.912".

    Returns:
        The parsed confidence value, or 0.0 if not present or unparseable.
    """
    for part in details.split(";"):
        part = part.strip()
        if part.startswith("confidence="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                pass

    return 0.0


def _iou(a: _Box, b: _Box) -> float:
    """Compute intersection-over-union for two axis-aligned bounding boxes.

    Args:
        a: First bounding box.
        b: Second bounding box.

    Returns:
        IoU in [0, 1].
    """
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union


def _match_and_score(
    gt_boxes: list[_Box],
    predictions: list[_Pred],
    iou_threshold: float,
) -> tuple[int, int, int, float]:
    """Greedily match predictions to ground truth and compute TP, FP, FN, and AP.

    Predictions are evaluated in descending confidence order. Each prediction is
    matched to the highest-IoU unmatched ground-truth box at or above the threshold.
    AP is accumulated as the area under the precision-recall curve (trapezoidal rule)
    in the same pass.

    Args:
        gt_boxes: Ground-truth bounding boxes in nanometers.
        predictions: Predicted bounding boxes with associated confidence scores.
        iou_threshold: Minimum IoU required for a prediction to count as a true positive.

    Returns:
        Tuple of (tp, fp, fn, ap).
    """
    preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    matched_gt = [False] * len(gt_boxes)

    tp = fp = 0
    prev_recall = ap = 0.0

    for pred in preds:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_boxes):
            if matched_gt[i]:
                continue
            v = _iou(pred.box, gt)
            if v > best_iou:
                best_iou, best_idx = v, i

        if best_iou >= iou_threshold:
            matched_gt[best_idx] = True
            tp += 1
        else:
            fp += 1

        r = tp / len(gt_boxes)
        p = tp / (tp + fp)
        ap += (r - prev_recall) * p
        prev_recall = r

    fn = matched_gt.count(False)
    return tp, fp, fn, ap


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with gt_ndpa, pred_ndpa, and iou_threshold.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate predicted NDPA annotations against ground truth."
    )
    parser.add_argument("--gt_ndpa", required=True, help="Path to ground-truth NDPA file.")
    parser.add_argument("--pred_ndpa", required=True, help="Path to predicted NDPA file.")
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5,
        help="IoU threshold for a positive match (default: 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    """Run NDPA evaluation and print metrics to stdout."""
    args = _parse_args()

    gt_boxes = [_to_box(ann) for ann in NDPAData(args.gt_ndpa).palynomorphs]
    predictions = [
        _Pred(box=_to_box(ann), confidence=_parse_confidence(ann.details))
        for ann in NDPAData(args.pred_ndpa).palynomorphs
    ]

    if not gt_boxes:
        print("No ground-truth annotations found.")
        return

    tp, fp, fn, ap = _match_and_score(gt_boxes, predictions, args.iou_threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nEvaluation Results")
    print(f"==================")
    print(f"Ground truth:   {len(gt_boxes)} annotations")
    print(f"Predictions:    {len(predictions)} annotations")
    print(f"IoU threshold:  {args.iou_threshold:.2f}\n")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")
    print(f"AP:         {ap:.4f}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")


if __name__ == "__main__":
    main()
