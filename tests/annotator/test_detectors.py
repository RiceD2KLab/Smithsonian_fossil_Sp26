import argparse
import csv
import random
from pathlib import Path

import cv2
import h5py
import numpy as np

from src.annotator.detectors import MODEL_CONFIGS, build_detector

def _pick_random_tile(h5_path: str, seed: int) -> np.ndarray:
    """Return a random tile with palynomorph annotations from the H5 file."""
    rng = random.Random(seed)

    with h5py.File(h5_path, "r") as h5f:
        groups = [k for k in h5f.keys() if isinstance(h5f[k], h5py.Group)]
        if not groups:
            raise ValueError(f"No tile groups found in H5 file: {h5_path}")

        # Keep only tiles that contain at least two palynomorph annotations.
        positive_groups: list[str] = []
        for tile_name in groups:
            grp = h5f[tile_name]
            n_gt = int(grp.attrs.get("num_annotations", -1))
            if n_gt > 2:
                positive_groups.append(tile_name)

        if not positive_groups:
            raise ValueError(
                "No tile groups with palynomorph annotations were found. "
                "Expected num_annotations > 2 or non-empty bboxes in at least one group."
            )

        rng.shuffle(positive_groups)
        grp = h5f[positive_groups[0]]
        tile = np.array(grp["focus_stacked"])
        return tile


def _center_crop(tile: np.ndarray, size: int) -> tuple[np.ndarray, int, int]:
    """Center-crop tile to square size. Returns (crop, offset_x, offset_y)."""
    h, w = tile.shape[:2]
    if h < size or w < size:
        raise ValueError(f"Tile is too small for requested crop {size}: got {(h, w)}")

    x0 = (w - size) // 2
    y0 = (h - size) // 2
    return tile[y0 : y0 + size, x0 : x0 + size], x0, y0

def _draw_overlay(tile_rgb: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Draw boxes and scores on a copy of tile image."""
    canvas = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            canvas,
            f"{float(score):.3f}",
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas

def run_test(
    h5_path: str,
    yolo_checkpoint: str,
    rfdetr_checkpoint: str,
    output_dir: str,
    confidence_threshold: float = 0.5,
    seed: int = 0,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tile = _pick_random_tile(h5_path, seed=seed)

    rows: list[dict[str, float | int | str]] = []

    # YOLO on full model-sized tile
    yolo_size = MODEL_CONFIGS["yolo"].tile_size
    yolo_tile, yolo_off_x, yolo_off_y = _center_crop(tile, yolo_size)
    yolo_detector = build_detector("yolo", yolo_checkpoint, confidence_threshold, yolo_size)
    yolo_boxes, yolo_scores = yolo_detector.predict(yolo_tile)
    yolo_overlay = _draw_overlay(yolo_tile, yolo_boxes, yolo_scores)
    cv2.imwrite(str(out / "yolo_tile_annotated.png"), yolo_overlay)

    for box, score in zip(yolo_boxes, yolo_scores):
        rows.append(
            {
                "model": "yolo",
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "score": float(score),
            }
        )

    # RF-DETR on center-cropped tile (as requested)
    rfdetr_size = MODEL_CONFIGS["rfdetr"].tile_size
    rfdetr_tile, rfdetr_off_x, rfdetr_off_y = _center_crop(tile, rfdetr_size)
    rfdetr_detector = build_detector("rfdetr", rfdetr_checkpoint, confidence_threshold, rfdetr_size)
    rfdetr_boxes, rfdetr_scores = rfdetr_detector.predict(rfdetr_tile)
    rfdetr_overlay = _draw_overlay(rfdetr_tile, rfdetr_boxes, rfdetr_scores)
    cv2.imwrite(str(out / "rfdetr_tile_annotated.png"), rfdetr_overlay)

    for box, score in zip(rfdetr_boxes, rfdetr_scores):
        rows.append(
            {
                "model": "rfdetr",
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "score": float(score),
            }
        )

    csv_path = out / "model_tile_detections.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "x1",
                "y1",
                "x2",
                "y2",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {csv_path}")
    print(f"Saved: {out / 'yolo_tile_annotated.png'}")
    print(f"Saved: {out / 'rfdetr_tile_annotated.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO and RF-DETR tile detectors on a random preprocessed focus-stacked H5 tile."
    )
    parser.add_argument("--h5_path", required=True, help="Path to H5 file generated by generate_tiles + focus_stack.")
    parser.add_argument("--yolo_checkpoint", required=True, help="Path to YOLO checkpoint.")
    parser.add_argument("--rfdetr_checkpoint", required=True, help="Path to RF-DETR checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Output directory for CSV and overlay images.")
    parser.add_argument("--confidence_threshold", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tile-group sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_test(
        h5_path=args.h5_path,
        yolo_checkpoint=args.yolo_checkpoint,
        rfdetr_checkpoint=args.rfdetr_checkpoint,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
