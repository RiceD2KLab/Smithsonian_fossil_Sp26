"""
End-to-end palynomorph annotation pipeline for NDPI whole-slide images.

Modules:
    pipeline    — NDPIAnnotator: gets tiles, runs detection, NMS, and outputs to CSV and NDPA
    detectors   — tileDetector protocol and YOLO/RF-DETR detector implementations
    nms         — non-maximum suppression for deduplicating cross-tile detections
    compression — focus stacking and focal-plane selection for tile compression
    config      — configuration dataclasses (ModelConfig, AnnotatorConfig)
    types       — shared data types (Detection, TileSpec)
    run         — command line entry point
"""
