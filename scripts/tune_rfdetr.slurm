#!/bin/bash
#SBATCH --job-name=palynomorph-rfdetr-tune
#SBATCH --account=commons
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-rfdetr-tune-%j.out
#SBATCH --error=slurm-rfdetr-tune-%j.err

module purge

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRATCH_ROOT="${SHARED_SCRATCH}/$USER"
VENV_DIR="${SCRATCH_ROOT}/miniconda_venv"
JOB_OUTPUT_DIR="$SCRATCH_ROOT/rfdetr/tune_outputs"
COCO_DIR="$SCRATCH_ROOT/processed/coco_best_plane"
RUN_NAME="rfdetr_tune_run$SLURM_JOB_ID"

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
  --batch_size 4 \
  --grad_accum 2 \

SAVE_DIR="$HOME/rfdetr/tune_outputs"
mkdir -p "$SAVE_DIR"
cp -r "$JOB_OUTPUT_DIR/$RUN_NAME/" "$SAVE_DIR"
echo "Results copied to $SAVE_DIR"
