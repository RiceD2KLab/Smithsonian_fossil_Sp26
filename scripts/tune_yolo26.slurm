#!/bin/bash
#SBATCH --job-name=palynomorph-yolo26-tune
#SBATCH --account=commons
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-yolo26-tune-%j.out
#SBATCH --error=slurm-yolo26-tune-%j.err

module purge

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRATCH_ROOT="${SHARED_SCRATCH}/$USER"
VENV_DIR="${SCRATCH_ROOT}/miniconda_venv"
JOB_OUTPUT_DIR="$SCRATCH_ROOT/yolo26/tune_outputs"
COCO_DIR="$SCRATCH_ROOT/processed/new-export-script/coco_focus_stack"
RUN_NAME="yolo26_tune_run$SLURM_JOB_ID"

unset PYTHONHOME
unset PYTHONPATH

export PYTHONNOUSERSITE=1

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ ! -d "$VENV_DIR" || ! -d "$VENV_DIR/conda-meta" ]]; then
  echo "Conda env not found at $VENV_DIR. Creating and installing requirements..."
  conda create --yes --prefix "$VENV_DIR" python=3.13.12
  conda activate "$VENV_DIR"
  pip install --upgrade pip
  pip install -r "$REPO_DIR/requirements.txt"
else
  echo "Activating existing conda env at $VENV_DIR..."
  conda activate "$VENV_DIR"
fi

cd "$SLURM_SUBMIT_DIR"

# PyTorch JITerator kernel cache corruption causes "CUDA driver error: invalid argument"
# on torch.prod() (known bug pytorch/pytorch#132756). Use a fresh per-job cache dir to avoid corruption.
rm -rf "$PYTORCH_KERNEL_CACHE_PATH"

N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
DEVICE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | awk 'NR==1{printf "%d", NR-1} NR>1{printf ",%d", NR-1}')

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
  --iterations 20 \
  --epochs 30 \
  --imgsz 1024 \
  --batch 16 \
  --workers 4 \
  --patience 10 \
  --device "$DEVICE" \
  --tune_aug \
  --search_space "$SEARCH_SPACE" \
  --aug_search_space "$AUG_SEARCH_SPACE" \

SAVE_DIR="$HOME/yolo26/tune_outputs"
mkdir -p "$SAVE_DIR"
cp -r "$JOB_OUTPUT_DIR/$RUN_NAME/" "$SAVE_DIR"
echo "Results copied to $SAVE_DIR"
