#!/bin/bash
#SBATCH --job-name=rfdetr-base
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-rfdetr-%j.out
#SBATCH --error=slurm-rfdetr-%j.err

module purge

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRATCH_ROOT="${SHARED_SCRATCH}/$USER"
VENV_DIR="${SCRATCH_ROOT}/smithsonian_venv"
JOB_OUTPUT_DIR="$SCRATCH_ROOT/rfdetr/outputs"
COCO_DIR="$SCRATCH_ROOT/processed/coco_data_format"

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

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

cd "$SLURM_SUBMIT_DIR"

# PyTorch JITerator kernel cache corruption causes "CUDA driver error: invalid argument"
# on torch.prod() — known bug pytorch/pytorch#132756. Use a fresh per-job cache dir.
export PYTORCH_KERNEL_CACHE_PATH="/tmp/torch_kernels_${SLURM_JOB_ID}"

export MASTER_ADDR=localhost
export MASTER_PORT=29500

python -m src.models.rfdetr.train \
  --model base \
  --coco_dir "$COCO_DIR" \
  --epochs 200 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 1e-4 \
  --lr_scheduler cosine \
  --lr_min_factor 0.01 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --imgsz 1008 \
  --output_dir "$JOB_OUTPUT_DIR/rfdetr_base" \
  --workers 4

SAVE_DIR="$HOME/rfdetr/outputs"
mkdir -p "$SAVE_DIR"
cp -r "$JOB_OUTPUT_DIR" "$SAVE_DIR"
echo "Results copied to $SAVE_DIR"
