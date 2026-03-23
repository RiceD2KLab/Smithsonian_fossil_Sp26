This module has two object-detection model used to detect palynomorphs. Both models inputs expects the **COCO-format dataset** produced by the `export_coco.py` step and output trained checkpoints plus optional visualisation artifacts.

---

## Module Structure

```
src/models/
├── utils.py             # Shared visualisation helpers (for FP, FN, GT, Predictions Bounding boxes) used by both models
│
├── yolo26/
│   ├── train.py         # Training entrypoint: COCO → YOLO label conversion → model.train()
│   ├── evaluate.py      # Evaluation entrypoint: mAP, PR curve, confusion matrix
│   ├── predict.py       # Inference + tile visualisation: predictions, GT overlay, FP/FN images
│   └── utils.py         # COCO↔YOLO label conversion and directory path helpers
│
└── rfdetr/
    ├── train.py         # Training entrypoint: bridges COCO layout → rfdetr roboflow format → model.train()
    └── predict.py       # Inference + tile visualisation: mirrors yolo26/predict.py for RF-DETR API
```

## Models

### YOLO26

Standard Ultralytics YOLO model (e.g. `yolo26l.pt`) for single-class palynomorph detection. Training is delegated to Ultralytics' `model.train()` API; this module provides thin argument parsing, COCO directory validation, and an automatic COCO → YOLO label conversion step. All hyperparameters are CLI flags passed through to Ultralytics' training. 

On every run of `train.py` or `evaluate.py`, `convert_coco_labels_to_yolo()` reads `annotations/instances_{split}.json` and writes per-image `labels/{split}/<stem>.txt` files in normalised center-XYWH format (if it doesn't already exist). This is the expected format for Ultralytics training. 

---

### RF-DETR (Roboflow)

A detection transformer from Roboflow with a ViT-based backbone. Four variants are available: `nano`, `small`, `base` (default), `large`.

Some key hyperparameters:

```
epochs=60, batch_size=4, grad_accum=2, lr=5e-5, lr_scheduler=cosine,
lr_min_factor=0.1, warmup_epochs=2, weight_decay=0.01, imgsz=1008,
early_stopping_patience=8, drop_path=0.1, aug_config=custom
```

`grad_accum` multiplies the effective batch size (`effective_batch = batch_size × grad_accum x num_gpus`).

**Roboflow layout bridging:** RF-DETR's training API expects a specific `roboflow` directory layout (`train/`, `valid/`, `test/` each containing images and `_annotations.coco.json`). `train.py` creates a staging directory inside `--output_dir/.rfdetr_dataset/` and populates it with **symlinks**.

**Augmentation:** The `custom` augmentation config (default) applies random horizontal/vertical flips, ±180° rotation, brightness/contrast jitter, and HSV shifts. Pass `--aug_config none` to disable all augmentation. To modify augmentations, edit at the top of `train.py`.

**Imporant note on image resolution:** The ViT backbone requires resolution to be divisible by 56. E.g, if you have size 1024, consider inputting 1008.

---

## How to Run

### Prerequisites

```bash
# From repo root
python -m venv env && source env/bin/activate
pip install -r requirements.txt

# All scripts must be run as modules from the repo root
export PYTHONPATH=.
```

The upstream COCO export must already exist. If it does not, run `export_coco.py` first (lives outside this module — see preprocessing docs).

---

### YOLO26

#### Train

```bash
python -m src.models.yolo26.train \
    --model yolo26l.pt \           # Ultralytics model weights or variant name
    --coco_dir data/coco_export \  # Root of the COCO export (must contain dataset.yaml)
    --epochs 200 \                 # Total training epochs
    --imgsz 1024 \                 # Input image size (pixels)
    --batch 16 \                   # Samples per batch
    --workers 4 \                  # DataLoader worker processes
    --optimizer adamw \            # Optimizer: adam | adamw | sgd
    --lr0 0.001 \                  # Initial learning rate
    --lrf 0.001 \                  # Final LR factor (final_lr = lr0 * lrf)
    --patience 10 \                # Early stopping patience (epochs)
    --project runs/yolo26 \        # Output root directory
    --name my_run                  # Subdirectory name for this run
```

Outputs land in `runs/yolo26/my_run/weights/best.pt`.

#### Evaluate

```bash
python -m src.models.yolo26.evaluate \
    --model runs/yolo26/my_run/weights/best.pt \
    --coco_dir data/coco_export \
    --split test \                 # train | val | test
    --imgsz 1024 \
    --batch 16 \
    --workers 4 \
    --project runs/yolo26 \
    --name my_eval \
    --plots \                      # Generate confusion matrix, PR curve, F1 curve
    --save_json                    # Save COCO-format predictions.json (optional)
```

#### Predict (inference + visualisation)

```bash
# Predictions only
python -m src.models.yolo26.predict \
    --weights runs/yolo26/my_run/weights/best.pt \
    --image_dir data/coco_export/images/test/ \
    --out_dir runs/yolo26/predictions/ \
    --conf_thresh 0.5 \            # Confidence threshold
    --imgsz 1024 \
    --iou_thresh 0.5 \             # IoU threshold for FP/FN matching
    --max_images 0                 # 0 = no limit

# With GT overlay and FP/FN breakdown (requires COCO annotation JSON)
python -m src.models.yolo26.predict \
    --weights runs/yolo26/my_run/weights/best.pt \
    --image_dir data/coco_export/images/test/ \
    --out_dir runs/yolo26/predictions/ \
    --coco_ann data/coco_export/annotations/instances_test.json \
    --show_gt \
    --show_fp_fn \
    --conf_thresh 0.5 \
    --imgsz 1024 \
    --iou_thresh 0.5 \
    --max_images 0
```

Output directories created under `--out_dir`:

| Directory | Contents |
|---|---|
| `predictions/` | Always created; annotated images with predicted boxes (green) |
| `ground_truth/` | Created when `--show_gt`; images with GT boxes (blue) |
| `fp/` | Created when `--show_fp_fn`; images where FP boxes occurred (red) |
| `fn/` | Created when `--show_fp_fn`; images where FN boxes occurred (orange) |

#### Running on NOTS

```bash
sbatch scripts/train_yolo.slurm     # 200 epochs, 24h, 1 GPU
sbatch scripts/evaluate_yolo.slurm  # 1h, 1 GPU; hardcoded weight path — edit before running
sbatch scripts/predict_yolo.slurm   # 1h, 1 GPU; hardcoded weight path — edit before running
```

Slurm scripts expect `$SHARED_SCRATCH` to point to the cluster scratch filesystem. Results are copied to `$HOME/yolo26/outputs/`.

---

### RF-DETR

#### Train

```bash
python -m src.models.rfdetr.train \
    --model base \                      # nano | small | base | large
    --coco_dir data/coco_export \       # Root of the COCO export
    --epochs 60 \
    --batch_size 4 \                    # Per-GPU batch size
    --grad_accum 2 \                    # Gradient accumulation steps
    --lr 5e-5 \                         # Learning rate
    --lr_scheduler cosine \             # step | cosine
    --lr_min_factor 0.1 \              # Min LR = lr * lr_min_factor (cosine annealing)
    --warmup_epochs 2 \                 # Linear LR warmup duration
    --weight_decay 0.01 \
    --imgsz 1008 \                      # Must be divisible by 56
    --output_dir runs/rfdetr/my_run \
    --workers 6 \
    --early_stopping_patience 8 \
    --drop_path 0.1 \                   # Stochastic depth rate for ViT backbone
    --aug_config custom                 # custom | none
```

For multi-GPU training use `torchrun` (can see Slurm script):

```bash
torchrun --nproc_per_node=2 --master_addr=localhost --master_port=29500 \
    -m src.models.rfdetr.train \
    --model base \
    --coco_dir data/coco_export \
    ... (same flags as above)
```

#### Predict (inference + visualisation)

```bash
# Predictions only
python -m src.models.rfdetr.predict \
    --weights runs/rfdetr/my_run/checkpoint_best_ema.pth \
    --image_dir data/coco_export/images/test/ \
    --out_dir runs/rfdetr/predictions/ \
    --model base \                      # Must match the variant used during training
    --conf_thresh 0.5 \
    --resolution 1008 \                 # Must match training resolution and be divisible by 56
    --iou_thresh 0.5 \
    --max_images 0

# With GT overlay and FP/FN breakdown
python -m src.models.rfdetr.predict \
    --weights runs/rfdetr/my_run/checkpoint_best_ema.pth \
    --image_dir data/coco_export/images/test/ \
    --out_dir runs/rfdetr/predictions/ \
    --coco_ann data/coco_export/annotations/instances_test.json \
    --show_gt \
    --show_fp_fn \
    --model base \
    --conf_thresh 0.5 \
    --resolution 1008 \
    --iou_thresh 0.5 \
    --max_images 0
```

**IMPORANT** RF-DETR does not have a standalone evaluate.py like YOLO. Formal mAP metrics on the test held-out split are produced during training after early stopping.

#### Running on NOTS

```bash
sbatch scripts/train_rfdetr.slurm    # 60 epochs, 5h, 2 GPUs (DDP)
sbatch scripts/predict_rfdetr.slurm # 1h, 1 GPU; hardcoded weight path — edit before running
```

> The RF-DETR Slurm script clears `$PYTORCH_KERNEL_CACHE_PATH` before training to avoid a known PyTorch JIT kernel cache corruption bug (`pytorch/pytorch#132756`) that surfaces as `CUDA driver error: invalid argument` on `torch.prod()`. If you run RF-DETR training without the Slurm script, do this manually or set a fresh cache path.

---

## Configuration

### Changing hyperparameters

Pass different values directly as CLI flags. For cluster runs, edit the relevant `.slurm` script variables at the top of the file.

### Pointing to a different dataset

Both models read from `--coco_dir`. Regenerate the COCO export with `export_coco.py` targeting a new output directory, then pass that directory as `--coco_dir`.

### Swapping checkpoints

- **YOLO**: pass any `.pt` file to `--model` for training (e.g. a different YOLO variant or a previously trained checkpoint), or to `--model`/`--weights` for evaluation/inference.
- **RF-DETR**: pass `--model {nano,small,base,large}` to select the architecture, and `--weights` for the checkpoint path. The variant must match the checkpoint, for example a `base` checkpoint cannot be loaded into a `large` model.

### Adding augmentation (RF-DETR)

Edit the `AUG_CONFIG` dict in `src/models/rfdetr/train.py` to add or adjust Albumentations transforms, then pass the config key via `--aug_config`.
