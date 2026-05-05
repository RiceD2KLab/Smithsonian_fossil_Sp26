"""
Preprocessing pipeline for converting NDPI/NDPA slides into model-ready datasets.

Modules:
    generate_tiles    — Tile NDPI slides into H5 files with bounding-box annotations
    postprocess_tiles — Add focus stacks, focal-plane rankings, and MIP to H5 tile files
    split_data        — Split H5 tile files into train / val / test sets
    export_coco       — Export H5 tile datasets to COCO JSON + image files
    focus_metrics     — Per-tile sharpness metrics (VoL, Tenengrad, percentile variants)
    h5_utils          — H5 file discovery and tile-job enumeration utilities
"""
