"""
Evaluates a trained RF-DETR model and generates a 
Precision-Recall (PR) curve using the official COCO evaluation metrics.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.models.rfdetr.train import _MODEL_CLASSES

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

    # 3. Run Inference & format for COCO evaluation
    print(f"Running inference on {len(img_ids)} images for PR curve...")
    results = []
    
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = Path(args.image_dir) / img_info['file_name']
        
        if not img_path.exists():
            continue
            
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
    
    # 5. Plot the PR Curve
    precisions = coco_eval.eval['precision']
    # Extract the curve: IoU=0.50 [0], All Recall Steps [:,], Class 0 [0], All Area [0], MaxDets=100 [2]
    pr_curve = precisions[0, :, 0, 0, 2]
    recalls = np.linspace(0.0, 1.0, 101)
    
    # Generate the visualization
    plt.figure(figsize=(10, 7))
    plt.plot(recalls, pr_curve, linewidth=3, color='#1f77b4')
    plt.fill_between(recalls, pr_curve, alpha=0.2, color='#1f77b4')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (IoU=0.50)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    out_img = out_path / 'pr_curve.png'
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"\nDone! PR curve saved to {out_img}")

if __name__ == "__main__":
    main()