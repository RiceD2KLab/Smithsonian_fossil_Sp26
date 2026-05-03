#!/bin/bash
#
# Hyperparameter tuning for YOLO26.
#
# Before sbatch, set paths if defaults do not match your environment:
#   export SCRATCH_ROOT=/path/to/scratch          # large writable disk (default: ~/smithsonian_fossil_scratch)
#   export COCO_DIR=/path/to/coco                 # COCO-format dataset root
#   export VENV_DIR=/path/to/conda/prefix         # optional: conda env location
#   export JOB_OUTPUT_DIR=/path/to/tuning/outputs # optional: tune run artifacts
#
# Edit #SBATCH directives below for your site (account, partition, GPU request, walltime).

#SBATCH --job-name=yolo26-tune
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-yolo26-tune-%j.out
#SBATCH --error=slurm-yolo26-tune-%j.err

module purge

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${SHARED_SCRATCH:-$HOME}/smithsonian_fossil_scratch}"
VENV_DIR="${VENV_DIR:-$SCRATCH_ROOT/venvs/smithsonian_fossil}"
JOB_OUTPUT_DIR="${JOB_OUTPUT_DIR:-$SCRATCH_ROOT/tuning_outputs/yolo26}"
COCO_DIR="${COCO_DIR:-$SCRATCH_ROOT/data/coco}"
RUN_NAME="yolo26_tune_${SLURM_JOB_ID}"

unset PYTHONHOME
unset PYTHONPATH

export PYTHONNOUSERSITE=1

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if [[ ! -d "$VENV_DIR" || ! -d "$VENV_DIR/conda-meta" ]]; then
  echo "Conda env not found at $VENV_DIR. Creating (python=${PYTHON_VERSION}) and installing requirements..."
  conda create --yes --prefix "$VENV_DIR" "python=${PYTHON_VERSION}"
  conda activate "$VENV_DIR"
  pip install --upgrade pip
  pip install -r "$REPO_DIR/requirements.txt"
else
  echo "Activating existing conda env at $VENV_DIR..."
  conda activate "$VENV_DIR"
fi

cd "$REPO_DIR"

# PyTorch JIT kernel cache can trigger spurious CUDA errors on some driver stacks;
# clear only if this variable is set (optionally set it to a per-job directory).
if [[ -n "${PYTORCH_KERNEL_CACHE_PATH:-}" ]]; then
  rm -rf "$PYTORCH_KERNEL_CACHE_PATH"
fi

# Local CUDA device indices 0..N-1 for visible GPUs (matches Ultralytics expectations).
DEVICE=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | awk '{printf "%s%s", (NR>1?",":""), NR-1}')

# Update me! These are the default search spaces for YOLO26.
SEARCH_SPACE='{
  "lr0":           ["log_float", 1e-5, 1e-2],
  "lrf":           ["float",     0.01, 1.0],
  "momentum":      ["float",     0.7,  0.98],
  "weight_decay":  ["float",     0.0,  0.001],
  "warmup_epochs": ["float",     0.0,  5.0],
  "box":           ["float",     1.0,  20.0],
  "cls":           ["float",     0.1,  4.0],
  "dfl":           ["float",     0.4,  12.0]
}'

# Update me! These are the default augmentation search spaces for YOLO26.
AUG_SEARCH_SPACE='{
  "fliplr":     ["float", 0.0, 1.0],
  "flipud":     ["float", 0.0, 1.0],
  "degrees":    ["float", 0.0, 180.0],
  "hsv_h":      ["float", 0.0, 0.1],
  "hsv_s":      ["float", 0.0, 0.9],
  "hsv_v":      ["float", 0.0, 0.9],
  "translate":  ["float", 0.0, 0.3],
  "scale":      ["float", 0.0, 0.5],
  "mosaic":     ["float", 0.0, 1.0],
  "mixup":      ["float", 0.0, 0.3],
  "copy_paste": ["float", 0.0, 1.0]
}'

python -m src.models.yolo26.tune \
  --coco_dir "$COCO_DIR" \
  --output_dir "$JOB_OUTPUT_DIR/$RUN_NAME" \
  --model l \
  --search_alg optuna \
  --iterations 50 \
  --epochs 100 \
  --imgsz 1024 \
  --batch 16 \
  --workers 3 \
  --patience 10 \
  --device "$DEVICE" \
  --tune_aug \
  --search_space "$SEARCH_SPACE" \
  --aug_search_space "$AUG_SEARCH_SPACE"
