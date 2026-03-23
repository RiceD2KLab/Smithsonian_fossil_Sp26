#!/bin/bash
#SBATCH --job-name=palynomorph-yolo26-training
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-yolo26-%j.out
#SBATCH --error=slurm-yolo26-%j.err

module purge

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRATCH_ROOT="${SHARED_SCRATCH}/$USER"
VENV_DIR="${SCRATCH_ROOT}/miniconda_venv"
JOB_OUTPUT_DIR="$SCRATCH_ROOT/yolo26/outputs"
COCO_DIR="$SCRATCH_ROOT/processed/coco_focus_stack"
RUN_NAME="yolo26_palynomorphs_run$SLURM_JOB_ID"

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

python -m src.models.yolo26.train \
  --model yolo26l.pt \
  --coco_dir $COCO_DIR \
  --epochs 200 \
  --imgsz 1024 \
  --batch 16 \
  --workers 4 \
  --project "$JOB_OUTPUT_DIR" \
  --name $RUN_NAME \
  --optimizer adamw \
  --lr0 0.001 \
  --lrf 0.001 \
  --patience 10

SAVE_DIR="$HOME/yolo26/outputs"
mkdir -p "$SAVE_DIR"
cp -r "$JOB_OUTPUT_DIR" "$SAVE_DIR"
echo "Results copied to $SAVE_DIR"
