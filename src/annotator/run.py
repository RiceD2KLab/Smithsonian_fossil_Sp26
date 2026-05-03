"""Command-line entrypoint for the annotator pipeline."""

import argparse

from src.annotator.config import AnnotatorConfig
from src.annotator.pipeline import NDPIAnnotator

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the annotator CLI.
    
    Returns:
        An `argparse.Namespace` containing parsed arguments.
    """
    
    parser = argparse.ArgumentParser(
        description="Run full-slide annotation on an NDPI file and export merged CSV + NDPA detections."
    )
    parser.add_argument("--ndpi_path", required=True, help="Path to NDPI file.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for x.csv and x.ndpi.ndpa.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        choices=["yolo", "rfdetr"],
        help="Model family to use for tile inference.",
    )
    parser.add_argument(
        "--rfdetr_variant",
        choices=["base", "2xlarge"],
        default="base",
        help="RF-DETR variant to use when --model_name=rfdetr (default: base).",
    )
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--overlap",
        type=float,
        required=True,
        help="Tile overlap fraction in [0.0, 1.0).",
    )
    parser.add_argument(
        "--magnification",
        type=float,
        required=True,
        help="Magnification level to read NDPI tiles at.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence for model detections.",
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for cross-tile NMS merging.",
    )
    parser.add_argument(
        "--compression_method",
        choices=["focus_stack", "best_focal_plane"],
        default="focus_stack",
        help="Method to compress W x H x C x Z tiles to W x H x C.",
    )
    parser.add_argument(
        "--focus_stack_kernel_size",
        type=int,
        default=5,
        help="Kernel size for LoG-based compression methods.",
    )
    parser.add_argument(
        "--tenengrad_ksize",
        type=int,
        choices=[1, 3, 5, 7],
        default=3,
        help="Sobel kernel size for Tenengrad-based best-focal-plane selection.",
    )
    parser.add_argument(
        "--annotation_class",
        default="paly",
        help="Class label written to every CSV row.",
    )
    parser.add_argument(
        "--export_crops",
        action="store_true",
        help="Export a crop for each final detection into a subfolder under output_dir.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the annotator using parsed command-line arguments."""
    args = parse_args()
    config = AnnotatorConfig(
        ndpi_path=args.ndpi_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        overlap=args.overlap,
        magnification=args.magnification,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        compression_method=args.compression_method,
        focus_stack_kernel_size=args.focus_stack_kernel_size,
        tenengrad_ksize=args.tenengrad_ksize,
        annotation_class=args.annotation_class,
        export_crops=args.export_crops,
        rfdetr_variant=args.rfdetr_variant,
    )

    annotator = NDPIAnnotator(config)
    detections = annotator.run()

    print(f"Wrote {len(detections)} detections.")

if __name__ == "__main__":
    main()
