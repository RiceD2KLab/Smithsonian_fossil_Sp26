# Scripts

This folder contains Slurm batch scripts used to run the pipeline implementation on HPC resources. The scripts set up a conda environment (if needed), configure paths for datasets and outputs, and run the training, prediction, evaluation, or tuning entry points in src.

## What each script does

- annotate.slurm: Runs the end-to-end annotator on an NDPI slide and writes the output NDPA and artifacts (supports RF-DETR or YOLO).
- train_rfdetr.slurm: Trains RF-DETR and saves checkpoints and evaluation metrics.
- train_yolo.slurm: Trains YOLO26 and saves checkpoints.
- evaluate_yolo.slurm: Evaluates a trained YOLO26 checkpoint on the test split and saves plots/results.
- tune_rfdetr.slurm: Runs RF-DETR hyperparameter tuning with Optuna-style search spaces.
- tune_yolo26.slurm: Runs YOLO26 hyperparameter tuning with Optuna-style search spaces.
- predict_rfdetr.slurm: Generates RF-DETR predictions; can optionally include ground truth annotations and show false positives and false negatives.
- predict_yolo.slurm: Generates YOLO26 predictions; can optionally include ground truth annotations and show false positives and false negatives.

## Notes

- Update the SBATCH directives, paths, and variables at the top of each script before submitting with sbatch.
- Output artifacts are typically copied to a user-writable location under $HOME.
