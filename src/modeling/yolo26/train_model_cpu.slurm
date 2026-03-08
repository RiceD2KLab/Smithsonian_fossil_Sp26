#!/bin/bash
# NOTS (Rice CRC) SLURM script for YOLO26l training CPU only.
#SBATCH --job-name=yolo26l-cpu
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-yolo26l-cpu-%j.out
#SBATCH --error=slurm-yolo26l-cpu-%j.err

module purge
module load foss/2023a PyTorch/2.1.2-CUDA-12.1.1

REPO_DIR="$SLURM_SUBMIT_DIR"
VENV_DIR="${SHARED_SCRATCH:-$HOME}/$USER/smithsonian_venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR and installing requirements..."
  python -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -r "$REPO_DIR/requirements.txt"
else
  source "$VENV_DIR/bin/activate"
fi

# Directories (local scratch for fast I/O)
JOB_ROOT_DIR="$LOCAL_SCRATCH/$USER"
JOB_DATA_DIR="$JOB_ROOT_DIR/data"
JOB_OUTPUT_DIR="$JOB_ROOT_DIR/outputs"

export JOB_ROOT_DIR
export JOB_DATA_DIR
export JOB_OUTPUT_DIR

mkdir -p "$JOB_ROOT_DIR"
mkdir -p "$JOB_DATA_DIR"
mkdir -p "$JOB_OUTPUT_DIR"

# Copy training data from allocation
DATA_SRC_DIR="/rhf/allocations/dsci435/smithsonian_full_sp26/processed"
echo "Copying data from $DATA_SRC_DIR ..."
start_time=$(date +%s)
cp -r -T "$DATA_SRC_DIR" "$JOB_DATA_DIR"
echo "Data copy finished in $(($(date +%s) - start_time))s"

# If train_val_test.json is in processed root instead of tiles/, use: "$JOB_DATA_DIR/train_val_test.json"
SPLITS_JSON="$JOB_DATA_DIR/tiles/train_val_test.json"

cd "$SLURM_SUBMIT_DIR"

# Avoid torchvision NMS backend issues on NOTS (use Ultralytics TorchNMS instead)
export ULTRALYTICS_FORCE_TORCH_NMS=1

python -m src.modeling.yolo26.run_train_nots \
  --model yolo26l.pt \
  --h5_root "$JOB_DATA_DIR/tiles" \
  --splits_json "$SPLITS_JSON" \
  --epochs 2 \
  --imgsz 1024 \
  --batch 4 \
  --workers 4 \
  --project "$JOB_OUTPUT_DIR" \
  --name yolo26l_test_cpu \
  --device cpu \
  --cache_labels \
  --single_cls

# Save results to persistent storage so they survive node release
SAVE_DIR="${SHARED_SCRATCH:-$HOME}/$USER/yolo26l_cpu_$SLURM_JOB_ID"
mkdir -p "$(dirname "$SAVE_DIR")"
cp -r "$JOB_OUTPUT_DIR" "$SAVE_DIR"
echo "Results copied to $SAVE_DIR"
