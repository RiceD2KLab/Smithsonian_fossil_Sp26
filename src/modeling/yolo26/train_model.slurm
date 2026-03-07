#!/bin/bash
# NOTS (Rice CRC) SLURM script for YOLO26l training.
# Test run: partition=scavenge, 1h. Full run: use --partition=commons (24h) or long (72h).
# On NOTS run: module spider CUDA; module spider PyTorch — then set the module load below.
#SBATCH --job-name=yolo26l
#SBATCH --account=commons
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-yolo26l-%j.out
#SBATCH --error=slurm-yolo26l-%j.err

module purge
module load foss/2023a PyTorch/2.1.2-CUDA-12.1.1

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

SPLITS_JSON="$JOB_DATA_DIR/train_val_test.json"

cd "$SLURM_SUBMIT_DIR"

python -m src.modeling.yolo26.train \
  --model yolo26l.pt \
  --h5_root "$JOB_DATA_DIR/tiles" \
  --splits_json "$SPLITS_JSON" \
  --epochs 2 \
  --imgsz 1024 \
  --batch 4 \
  --workers 4 \
  --project "$JOB_OUTPUT_DIR" \
  --name yolo26l_test \
  --cache_labels \
  --single_cls

SAVE_DIR="${SHARED_SCRATCH:-$HOME}/$USER/yolo26l_$SLURM_JOB_ID"
mkdir -p "$(dirname "$SAVE_DIR")"
cp -r "$JOB_OUTPUT_DIR" "$SAVE_DIR"
echo "Results copied to $SAVE_DIR"
