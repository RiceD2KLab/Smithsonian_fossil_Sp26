"""
YOLO26 model wrapper for palynomorph detection.

Modules:
    train   — Fine-tune YOLO26 on a COCO-format tile dataset
    predict — Run inference with a trained YOLO26 checkpoint
    tune    — Bayesian hyperparameter search over YOLO26 training configurations
    evaluate — Evaluate a trained YOLO26 checkpoint on a specific split of the dataset
    utils   — COCO <-> YOLO26 label conversion and path helpers
"""
