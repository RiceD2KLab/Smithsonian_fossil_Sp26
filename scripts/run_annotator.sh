#!/bin/bash

python -m src.annotator.run \
    --ndpi_path /rhf/allocations/dsci435/smithsonian_full_sp26/raw/Images_with_annotations_for_CNN_training/USNMPAL_793367_D3702_L_2024_02_02_15_49_54_Tennessee.ndpi \
    --output_dir output \
    --model_name yolo \
    --checkpoint_path /scratch/pom1/yolo26/outputs/yolo26_train_run_focus_stacked/weights/best.pt \
    --overlap 0.1 \
    --magnification 40 \