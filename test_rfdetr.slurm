#!/bin/bash
#SBATCH --job-name=rfdetr-base
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:lovelace:2
#SBATCH --time=05:00:00
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
  #source "$VENV_DIR/bin/activate"
  conda activate "/scratch/pom1/miniconda_venv"
fi

#export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

cd "$SLURM_SUBMIT_DIR"

# PyTorch JITerator kernel cache corruption causes "CUDA driver error: invalid argument"
# on torch.prod() — known bug pytorch/pytorch#132756. Use a fresh per-job cache dir.
# Cleaning out the cache directory every run to avoid corruption.
rm -rf "$PYTORCH_KERNEL_CACHE_PATH"

N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
# torchrun spawns one process per GPU; sets RANK/WORLD_SIZE/LOCAL_RANK for rfdetr DDP init.
torchrun --nproc_per_node="$N_GPUS" --master_addr=localhost --master_port=29500 \
  -m src.models.rfdetr.train \
  --model base \
  --coco_dir "$COCO_DIR" \
  --epochs 60 \
  --batch_size 4 \
  --grad_accum 2 \
  --lr 5e-5 \
  --lr_scheduler cosine \
  --lr_min_factor 0.1 \
  --warmup_epochs 2 \
  --weight_decay 0.01 \
  --imgsz 1008 \
  --output_dir "$JOB_OUTPUT_DIR/rfdetr_run1" \
  --workers 6 \
  --early_stopping_patience 8
  --drop_path 0.1
  --aug_config custom

SAVE_DIR="$HOME/rfdetr/outputs"
mkdir -p "$SAVE_DIR"
cp -r "$JOB_OUTPUT_DIR" "$SAVE_DIR"
echo "Results copied to $SAVE_DIR"