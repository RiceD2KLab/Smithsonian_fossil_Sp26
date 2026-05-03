#!/bin/bash
#
# Hyperparameter tuning for RF-DETR
#
# Before sbatch, set paths if defaults do not match your environment:
#   export SCRATCH_ROOT=/path/to/scratch          # large writable disk (default: ~/smithsonian_fossil_scratch)
#   export COCO_DIR=/path/to/coco                 # COCO-format dataset root
#   export VENV_DIR=/path/to/conda/prefix         # optional: conda env location
#   export JOB_OUTPUT_DIR=/path/to/tuning/outputs # optional: tune run artifacts
#
# Edit #SBATCH directives below for your site (account, partition, GPU request, walltime).

#SBATCH --job-name=rfdetr-tune
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-rfdetr-tune-%j.out
#SBATCH --error=slurm-rfdetr-tune-%j.err

module purge

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${SHARED_SCRATCH:-$HOME}/smithsonian_fossil_scratch}"
VENV_DIR="${VENV_DIR:-$SCRATCH_ROOT/venvs/smithsonian_fossil}"
JOB_OUTPUT_DIR="${JOB_OUTPUT_DIR:-$SCRATCH_ROOT/tuning_outputs/rfdetr}"
COCO_DIR="${COCO_DIR:-$SCRATCH_ROOT/data/coco}"
RUN_NAME="rfdetr_tune_${SLURM_JOB_ID}"

# Update me! These are the default search spaces for RF-DETR.
SEARCH_SPACE='{"lr": ["log_float", 5e-5, 5e-4], "weight_decay": ["log_float", 5e-3, 0.1],
       "drop_path": ["float", 0.20, 0.55], "warmup_epochs": ["int", 3, 8], "lr_min_factor": ["float", 0.08, 0.25]}'

# Update me! These are the default augmentation search spaces for RF-DETR.
AUG_SPACE='{"rotate_p": ["float", 0.3, 0.9], "brightness_p": ["float", 0.1, 0.5], "hsv_p": ["float", 0.1, 0.5]}'

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

N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | grep -c . || true)
[[ "$N_GPUS" -ge 1 ]] || N_GPUS=1

python -m src.models.rfdetr.tune \
  --coco_dir "$COCO_DIR" \
  --output_dir "$JOB_OUTPUT_DIR/$RUN_NAME" \
  --model 2xlarge \
  --method bayesian \
  --n_trials 20 \
  --epochs 30 \
  --imgsz 1000 \
  --workers 4 \
  --early_stopping_patience 10 \
  --nproc "$N_GPUS" \
  --tune_aug \
  --aug_search_space "$AUG_SPACE" \
  --batch_size 4 \
  --grad_accum 2 \
  --search_space "$SEARCH_SPACE"
