#!/bin/bash
#SBATCH --job-name=palynomorph-annotator
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-annotator-%j.out
#SBATCH --error=slurm-annotator-%j.err

NDPI_PATH="/rhf/allocations/dsci435/smithsonian_full_sp26/raw/USNMPAL_793473_D3917_L_2024-01-02_13_52_23_Alaska.ndpi"
OUTPUT_DIR="/scratch/${USER}/annotator"
MODEL_NAME="rfdetr"          # or "yolo"
CHECKPOINT_PATH="/scratch/${USER}/rfdetr/outputs/rfdetr_focus_stack/checkpoint_best_ema.pth"
OVERLAP=0.1
MAGNIFICATION=40
NPROC=$SLURM_CPUS_PER_TASK

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRATCH_ROOT="${SHARED_SCRATCH}/$USER"
VENV_DIR="${SCRATCH_ROOT}/miniconda_venv"

module purge
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

rm -rf "$PYTORCH_KERNEL_CACHE_PATH"

python -m src.annotator \
  --ndpi_path "$NDPI_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --overlap "$OVERLAP" \
  --magnification "$MAGNIFICATION" \
