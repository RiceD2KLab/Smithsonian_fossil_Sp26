#!/bin/bash

NDPI_PATH="/path/to/ndpi/file"
OUTPUT_PATH="/path/to/output/directory"
MODEL_NAME="rfdetr"  # or "yolo"
MODEL_PATH="/path/to/model/checkpoint.pt"
OVERLAP=0.1
MAGNIFICATION=40

python -m src.annotator \
    --ndpi_path $NDPI_PATH \
    --output_dir $OUTPUT_PATH \
    --model_name $MODEL_NAME \
    --checkpoint_path $MODEL_PATH \
    --overlap $OVERLAP \
    --magnification $MAGNIFICATION \
