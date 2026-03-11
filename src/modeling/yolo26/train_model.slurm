#!/bin/bash
# NOTS (Rice CRC) SLURM script for YOLOv8l-style training on H5 tiles.
# Full run: commons 24h, batch 16, 4 workers (8 CPUs).
# For a short test, override partition/time/epochs/batch on the command line as needed.
#SBATCH --job-name=yolo26l
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-yolo26l-%j.out
#SBATCH --error=slurm-yolo26l-%j.err

module purge

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRATCH_ROOT="${SHARED_SCRATCH}/$USER"
VENV_DIR="${SCRATCH_ROOT}/smithsonian_venv"
JOB_OUTPUT_DIR="$SCRATCH_ROOT/yolo26/outputs"
SPLITS_JSON="$SCRATCH_ROOT/processed/train_val_test.json"

unset PYTHONHOME
unset PYTHONPATH

export PYTHONNOUSERSITE=1

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR and installing requirements..."
  python -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -r "$REPO_DIR/requirements.txt"
else
  source "$VENV_DIR/bin/activate"
fi

# Restrict PYTHONPATH to this repo
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

cd "$SLURM_SUBMIT_DIR"

python -m src.modeling.yolo26.run_train_nots \
  --model yolo26l.pt \
  --h5_root "$SCRATCH_ROOT/processed/tiles" \
  --splits_json "$SPLITS_JSON" \
  --epochs 200 \
  --imgsz 1024 \
  --batch 16 \
  --workers 4 \
  --project "$JOB_OUTPUT_DIR" \
  --name yolo26l_palynomorphs \
  --cache_labels \
  --single_cls

SAVE_DIR="$HOME/yolo26/outputs"
mkdir -p "$SAVE_DIR"
cp -r "$JOB_OUTPUT_DIR" "$SAVE_DIR"
echo "Results copied to $SAVE_DIR"
