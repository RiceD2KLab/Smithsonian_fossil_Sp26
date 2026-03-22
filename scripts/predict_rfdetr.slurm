#!/bin/bash
#SBATCH --job-name=palynomorph-rfdetr-predict
#SBATCH --account=commons
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-rfdetr-%j.out
#SBATCH --error=slurm-rfdetr-%j.err

module purge

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRATCH_ROOT="${SHARED_SCRATCH}/$USER"
VENV_DIR="${SCRATCH_ROOT}/miniconda_venv"
JOB_OUTPUT_DIR="$SCRATCH_ROOT/rfdetr/outputs"
COCO_DIR="$SCRATCH_ROOT/processed/coco_focus_stack"

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

# predictions only
# python -m src.models.rfdetr.predict \
#     --weights "$JOB_OUTPUT_DIR/rfdetr_focus_stack/checkpoint_best_ema.pth" \
#     --image_dir "$COCO_DIR/images/test/" \
#     --out_dir "$JOB_OUTPUT_DIR/predictions/" \
#     --conf_thresh 0.5 \
#     --resolution 1008 \
#     --iou_thresh 0.5 \
#     --max_images 0

# predictions + GT overlay + FP/FN breakdown
python -m src.models.rfdetr.predict \
    --weights "$JOB_OUTPUT_DIR/rfdetr_focus_stack/checkpoint_best_ema.pth" \
    --image_dir "$COCO_DIR/images/test/" \
    --out_dir "$JOB_OUTPUT_DIR/predictions/" \
    --coco_ann "$COCO_DIR/annotations/instances_test.json" \
    --show_gt \
    --show_fp_fn \
    --conf_thresh 0.5 \
    --resolution 1008 \
    --iou_thresh 0.5 \
    --max_images 0

SAVE_DIR="$HOME/rfdetr/outputs"
mkdir -p "$SAVE_DIR"
cp -r "$JOB_OUTPUT_DIR" "$SAVE_DIR"
echo "Results copied to $SAVE_DIR"