Two object-detection models for palynomorph detection. All models consume the **COCO-format dataset** produced by `export_coco.py`.

---

## Module Structure

```
src/models/
├── utils.py             # Shared visualisation helpers (bounding-box overlays)
│
├── yolo26/
│   ├── train.py         # Training: COCO → YOLO label conversion → model.train()
│   ├── evaluate.py      # Evaluation: mAP, PR curve, confusion matrix
│   ├── predict.py       # Inference + visualisation: predictions, GT overlay, FP/FN
│   ├── tune.py          # Optuna TPE hyperparameter search
│   └── utils.py         # COCO↔YOLO label conversion and path helpers
│
└── rfdetr/
    ├── train.py         # Training: COCO layout → rfdetr roboflow format → model.train()
    ├── predict.py       # Inference + visualisation: mirrors yolo26/predict.py
    └── tune.py          # Optuna Bayesian hyperparameter search
```

---

## Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.   # all scripts must be run as modules from the repo root
```

The upstream COCO export must already exist (`export_coco.py`).

---

## YOLO26

Standard Ultralytics YOLO model for single-class palynomorph detection. On every `train.py` or `evaluate.py` run, `convert_coco_labels_to_yolo()` writes normalised center-XYWH label files under `labels/{split}/` if they don't already exist.

### Train

```bash
python -m src.models.yolo26.train \
    --model yolo26l.pt \
    --coco_dir data/coco_export \
    --epochs 200 \
    --imgsz 1024 \
    --batch 16 \
    --workers 4 \
    --optimizer adamw \
    --lr0 0.001 \
    --lrf 0.001 \
    --patience 10 \
    --project runs/yolo26 \
    --name my_run
```

Outputs: `runs/yolo26/my_run/weights/best.pt`

### Evaluate

```bash
python -m src.models.yolo26.evaluate \
    --model runs/yolo26/my_run/weights/best.pt \
    --coco_dir data/coco_export \
    --split test \
    --imgsz 1024 \
    --batch 16 \
    --workers 4 \
    --project runs/yolo26 \
    --name my_eval \
    --plots \       # confusion matrix, PR curve, F1 curve
    --save_json     # COCO-format predictions.json (optional)
```

### Predict

```bash
python -m src.models.yolo26.predict \
    --weights runs/yolo26/my_run/weights/best.pt \
    --image_dir data/coco_export/images/test/ \
    --out_dir runs/yolo26/predictions/ \
    --coco_ann data/coco_export/annotations/instances_test.json \  # optional; enables GT/FP/FN
    --show_gt \
    --show_fp_fn \
    --conf_thresh 0.5 \
    --imgsz 1024 \
    --iou_thresh 0.5 \
    --max_images 0    # 0 = no limit
```

| Output dir | Contents |
|---|---|
| `predictions/` | Annotated images with predicted boxes (green) |
| `ground_truth/` | GT boxes (blue); created with `--show_gt` |
| `fp/` | Images with false-positive boxes (red); created with `--show_fp_fn` |
| `fn/` | Images with false-negative boxes (orange); created with `--show_fp_fn` |

### Tune

Optuna TPE Bayesian search over training (and optionally augmentation) hyperparameters. Requires `pip install optuna`.

```bash
python -m src.models.yolo26.tune \
    --coco_dir data/coco_export \
    --output_dir runs/yolo26_tune \
    --model l \
    --epochs 30 \
    --imgsz 640 \
    --batch 16 \
    --workers 4 \
    --patience 10 \
    --iterations 20 \
    --search_alg optuna \
    --search_space '{
      "lr0":           ["log_float", 1e-5, 1e-2],
      "lrf":           ["float",     0.01, 1.0],
      "momentum":      ["float",     0.7,  0.98],
      "weight_decay":  ["float",     0.0,  1e-3],
      "warmup_epochs": ["float",     0.0,  5.0],
      "box":           ["float",     1.0,  20.0],
      "cls":           ["float",     0.1,  4.0],
      "dfl":           ["float",     0.4,  12.0]
    }'
```

To also search augmentation parameters, add:

```bash
    --tune_aug \
    --aug_search_space '{
      "fliplr": ["float", 0.0, 1.0], "flipud": ["float", 0.0, 1.0],
      "degrees": ["float", 0.0, 180.0], "hsv_h": ["float", 0.0, 0.1],
      "hsv_s": ["float", 0.0, 0.9], "hsv_v": ["float", 0.0, 0.9],
      "translate": ["float", 0.0, 0.3], "scale": ["float", 0.0, 0.5],
      "mosaic": ["float", 0.0, 1.0], "mixup": ["float", 0.0, 0.3],
      "copy_paste": ["float", 0.0, 1.0]
    }'
```

**Resuming:** re-run the same command — the Optuna study persists in `output_dir/optuna.db` and incomplete trials pick up where they left off.

**Outputs:** `output_dir/trials/results.json` (all trials) and `output_dir/best_params.json`. Pass the best params as CLI flags to `train.py`.

**Search-space schema** (same for `--search_space` and `--aug_search_space`): each key maps to a spec list — `["log_float", low, high]`, `["float", low, high]`, `["int", low, high]`, or `["categorical", [c1, c2, ...]]`. Accepts an inline JSON string or a path to a JSON file.

---

## RF-DETR

A detection transformer (Roboflow) with a ViT backbone. Variants: `nano`, `small`, `base` (default), `large`, ``xlarge`, `2xlarge`.

**Roboflow layout bridging:** RF-DETR expects `train/`, `valid/`, `test/` each with images and `_annotations.coco.json`. `train.py` creates a staging directory at `--output_dir/.rfdetr_dataset/` populated with symlinks — no data is copied.

**Image resolution:** the ViT backbone, for nano - large variants requires resolution divisible by 56 (e.g. use 1008 instead of 1024). For xlarge and 2xlarge, it must be divisible by 40 (e.g. use 1000 instead of 1024).

**Augmentation:** the default `custom` config applies random flips, ±180° rotation, brightness/contrast jitter, and HSV shifts. Pass `--aug_config none` to disable. Edit `AUG_CONFIG` at the top of `train.py` to customise.

### Train

```bash
python -m src.models.rfdetr.train \
    --model base \
    --coco_dir data/coco_export \
    --epochs 60 \
    --batch_size 4 \
    --grad_accum 2 \        # effective batch = batch_size × grad_accum × num_gpus
    --lr 5e-5 \
    --lr_scheduler cosine \
    --lr_min_factor 0.1 \
    --warmup_epochs 2 \
    --weight_decay 0.01 \
    --imgsz 1008 \
    --output_dir runs/rfdetr/my_run \
    --workers 6 \
    --early_stopping_patience 8 \
    --drop_path 0.1 \
    --aug_config custom
```

Multi-GPU (DDP):

```bash
torchrun --nproc_per_node=2 --master_addr=localhost --master_port=29500 \
    -m src.models.rfdetr.train --model base --coco_dir data/coco_export ...
```

### Predict

```bash
python -m src.models.rfdetr.predict \
    --weights runs/rfdetr/my_run/checkpoint_best_ema.pth \
    --image_dir data/coco_export/images/test/ \
    --out_dir runs/rfdetr/predictions/ \
    --model base \
    --coco_ann data/coco_export/annotations/instances_test.json \  # optional; enables GT/FP/FN
    --show_gt \
    --show_fp_fn \
    --conf_thresh 0.5 \
    --resolution 1008 \
    --iou_thresh 0.5 \
    --max_images 0
```

Formal mAP metrics on the held-out test split are produced at the end of training. To re-run evaluation, use `src/plots/generate_pr_curve.py`.

### Tune

Optuna Bayesian search (or random) over training and augmentation hyperparameters. Each trial can run multi-GPU via `--nproc`. Requires `pip install optuna`.

```bash
python -m src.models.rfdetr.tune \
    --coco_dir data/coco_export \
    --output_dir runs/rfdetr_tune \
    --model base \
    --n_trials 20 \
    --method bayesian \
    --epochs 30 \
    --imgsz 1008 \
    --batch_size 4 \
    --grad_accum 2 \
    --workers 4 \
    --nproc 2 \
    --early_stopping_patience 10 \
    --search_space '{
      "lr":            ["log_float", 1e-6, 1e-3],
      "weight_decay":  ["log_float", 1e-4, 0.1],
      "drop_path":     ["float",     0.0,  0.3],
      "warmup_epochs": ["int",       1,    5],
      "lr_min_factor": ["float",     0.01, 0.3]
    }'
```

To also search augmentation parameters, add `--tune_aug --aug_search_space '{"rotate_p": ["float", 0.3, 0.9], ...}'`.

**Outputs:** `output_dir/trials/results.json` and `output_dir/best_params.json`. Pass the best params as CLI flags to `train.py`.

The search-space schema is identical to YOLO26's (see above).
---

## Configuration

- **Hyperparameters:** pass CLI flags, or edit the variables at the top of the relevant `.slurm` script.
- **Dataset:** all models read from `--coco_dir`. Regenerate with `export_coco.py` and point `--coco_dir` at the new directory.
- **Checkpoints — YOLO:** pass any `.pt` file to `--model` (train) or `--weights` (evaluate/predict).
- **Checkpoints — RF-DETR:** `--model {nano,small,base,large}` selects the architecture; `--weights` provides the checkpoint. The variant must match the checkpoint.
