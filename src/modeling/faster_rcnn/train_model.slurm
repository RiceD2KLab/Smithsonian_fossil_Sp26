#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --export=ALL

module purge
module load foss/2023a PyTorch/2.1.2-CUDA-12.1.1

# Set up environment variables
JOB_ROOT_DIR="$LOCAL_SCRATCH/$USER"
JOB_DATA_DIR="$JOB_ROOT_DIR/data"
JOB_OUTPUT_DIR="$JOB_ROOT_DIR/outputs"

export JOB_ROOT_DIR
export JOB_DATA_DIR
export JOB_OUTPUT_DIR

mkdir -p "$JOB_ROOT_DIR"
mkdir -p "$JOB_DATA_DIR"
mkdir -p "$JOB_JOB_OUTPUT_DIR"

# Copy over the training data
start_time=$(date +%s)
echo "STARTING DATA COPY: $start_time"
DATA_SRC_DIR="/rhf/allocations/dsci435/smithsonian_full_sp26/processed"
cp -r -T "$DATA_SRC_DIR" "$JOB_DATA_DIR"
end_time=$(date +%s)
echo "FINISHED DATA COPY: $end_time"

# Echo data files
echo "COPIED THE FOLLOWING DATA FILES"
echo $( ls "$JOB_DATA_DIR/tiles" )

#python run.py
exit 0