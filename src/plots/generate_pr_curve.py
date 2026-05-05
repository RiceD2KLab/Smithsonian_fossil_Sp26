"""
Evaluates a trained RF-DETR model and generates a 
Precision-Recall (PR) curve using the official COCO evaluation metrics.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.models.rfdetr.train import _MODEL_CLASSES


def _plot_pr_curve(
    recalls: np.ndarray,
    precision: np.ndarray,
    scores: np.ndarray,
    *,
    ap: float,
    ap50: float,
    n_images_eval: int,
    n_gt: int,
    weights_name: str,
    model: str,
    resolution: int,
    out_path: Path,
) -> tuple[float, float, float, float]:
    """Save a presentation-style precision-recall figure.

    Args:
        recalls: The recall values.
        precision: The precision values.
        scores: The confidence scores.
        ap: The average precision.
        ap50: The average precision at IoU=0.50.
        n_images_eval: The number of images evaluated.
        n_gt: The number of ground truth objects.
        weights_name: The name of the weights file.
        model: The model variant.
        resolution: The resolution of the images.
        out_path: The path to save the figure.

    Returns:
        The max-F₁ operating point and the corresponding confidence threshold.
    """
    eps = 1e-12
    f1 = (2.0 * precision * recalls) / np.clip(precision + recalls, eps, None)
    f1 = np.where(np.isfinite(f1), f1, 0.0)
    j = int(np.argmax(f1))
    r_star, p_star, f1_star = float(recalls[j]), float(precision[j]), float(f1[j])
    raw_score = float(scores[j])
    score_star = raw_score if raw_score >= 0 else float("nan")

    style = {
        "figure.facecolor": "white",
        "axes.facecolor": "#f7f8fa",
        "axes.edgecolor": "#2c3e50",
        "axes.labelcolor": "#2c3e50",
        "axes.titlecolor": "#1a1a1a",
        "xtick.color": "#2c3e50",
        "ytick.color": "#2c3e50",
        "grid.color": "#cfd8dc",
        "grid.linestyle": ":",
        "grid.linewidth": 0.9,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cfd8dc",
    }

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
   
        curve_color = "#1565c0"
        fill_color = "#42a5f5"

        ax.fill_between(recalls, precision, color=fill_color, alpha=0.18, linewidth=0)
        (line,) = ax.plot(
            recalls,
            precision,
            color=curve_color,
            linewidth=2.8,
        )
        line.set_path_effects([pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()])

        ax.scatter(
            [r_star],
            [p_star],
            s=120,
            zorder=5,
            color="#c62828",
            edgecolors="white",
            linewidths=1.5,
            label=f"Max F1 (conf={score_star:.3f})",
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Recall", fontsize=16, fontweight="semibold")
        ax.set_ylabel("Precision", fontsize=16, fontweight="semibold")
        ax.set_title(
            "Precision Recall Curve (IoU = 0.50)",
            fontsize=20,
            fontweight="bold",
            pad=12,
        )
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.grid(True, which="major", alpha=0.85)
        ax.set_axisbelow(True)

        leg = ax.legend(loc="lower left", fontsize=14, frameon=True)
        leg.get_frame().set_linewidth(0.8)

        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return r_star, p_star, f1_star, score_star


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments required for evaluation.

    Inputs:
        None (Reads arguments from sys.argv).

    Outputs:
        argparse.Namespace: An object containing the following parsed attributes:
            - weights (str): Path to the trained checkpoint.
            - ann_file (str): Path to the ground-truth COCO JSON file.
            - image_dir (str): Path to the directory containing test images.
            - out_dir (str): Path to directory to save the PR curve and JSON results.
            - model (str): The RF-DETR model variant (e.g., 'base', 'small').
            - resolution (int): Image resolution for inference.
    """
    p = argparse.ArgumentParser(description="Generate PR Curve for RF-DETR")
    p.add_argument("--weights", required=True, help="Path to trained checkpoint (.pt or .pth)")
    p.add_argument("--ann_file", required=True, help="Path to COCO _annotations.coco.json")
    p.add_argument("--image_dir", required=True, help="Path to image directory")
    p.add_argument("--out_dir", required=True, help="Path to output directory")
    p.add_argument("--model", choices=list(_MODEL_CLASSES.keys()), default="base", help="Model variant")
    p.add_argument("--resolution", type=int, default=1008)
    return p.parse_args()


def main() -> None:
    """
    Main execution pipeline for generating the Precision-Recall curve.
    
    Workflow:
        1. Loads the specified RF-DETR model and weights.
        2. Loads the ground truth COCO annotations.
        3. Runs inference on all images at a near-zero threshold (0.001) to capture all predictions.
        4. Calculates metrics using COCOeval.
        5. Plots and saves the Precision-Recall curve to disk.

    Inputs:
        None (Relies on parsed command-line arguments via parse_args()).

    Outputs:
        None (Saves 'pr_curve.png' to the local directory and outputs JSON temp files).
    """
    args = parse_args()

    # 1. Initialize Model
    print(f"Loading {args.model} model from {args.weights}...")
    model = _MODEL_CLASSES[args.model](
        pretrain_weights=args.weights,
        num_classes=1,
        resolution=args.resolution,
    )
    # Optimize model for faster inference times
    model.optimize_for_inference()

    # 2. Load Ground Truth via pycocotools
    print(f"Loading Ground Truth from {args.ann_file}...")
    coco_gt = COCO(args.ann_file)
    img_ids = coco_gt.getImgIds()
    
    # Extract the category ID used for palynomorphs (defaults to the first available)
    cat_ids = coco_gt.getCatIds()
    cat_id = 0 if 0 in cat_ids else cat_ids[0]
    n_gt = len(coco_gt.getAnnIds(catIds=[cat_id]))

    # 3. Run Inference & format for COCO evaluation
    print(f"Running inference on {len(img_ids)} images for PR curve...")
    results = []
    n_images_eval = 0

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = Path(args.image_dir) / img_info['file_name']

        if not img_path.exists():
            continue

        n_images_eval += 1
        detections = model.predict(str(img_path), threshold=0.001)
        
        if detections.xyxy is None or len(detections.xyxy) == 0:
            continue
            
        # Convert bounding boxes from xyxy (PyTorch) to xywh (COCO Standard)
        for box, conf in zip(detections.xyxy, detections.confidence):
            x_min, y_min, x_max, y_max = map(float, box)
            w = x_max - x_min
            h = y_max - y_min
            
            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x_min, y_min, w, h],
                "score": float(conf)
            })

    if len(results) == 0:
        print("No predictions made! Check your paths.")
        return

    # 4. Save and evaluate temporary results
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    res_file = out_path / "temp_predictions.json"
    with open(res_file, "w") as f:
        json.dump(results, f)
        
    print("Calculating PR metrics...")
    coco_dt = coco_gt.loadRes(str(res_file))
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 5. Plot the PR curve (IoU=0.50, all areas, maxDets=100, first category slice)
    precisions = coco_eval.eval["precision"]
    pr_curve = precisions[0, :, 0, 0, 2]
    scores_curve = coco_eval.eval["scores"][0, :, 0, 0, 2]
    recalls = np.linspace(0.0, 1.0, 101)

    stats = coco_eval.stats
    ap = float(stats[0])
    ap50 = float(stats[1])
    weights_name = Path(args.weights).name

    out_img = out_path / "pr_curve.png"
    r_star, p_star, f1_star, score_star = _plot_pr_curve(
        recalls,
        pr_curve,
        scores_curve,
        ap=ap,
        ap50=ap50,
        n_images_eval=n_images_eval,
        n_gt=n_gt,
        weights_name=weights_name,
        model=args.model,
        resolution=args.resolution,
        out_path=out_img,
    )

    metrics = {
        "ap":               ap,
        "ap50":             ap50,
        "p_star":           p_star,
        "r_star":           r_star,
        "f1_star":          f1_star,
        "score_threshold":  score_star,
        "n_images_eval":    n_images_eval,
        "n_gt":             n_gt,
        "weights":          weights_name,
        "model":            args.model,
        "resolution":       args.resolution,
    }
    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone! PR curve saved to {out_img}")
    print(f"Metrics sidecar saved to {metrics_path}")

if __name__ == "__main__":
    main()