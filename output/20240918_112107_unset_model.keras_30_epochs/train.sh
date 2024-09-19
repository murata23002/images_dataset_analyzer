#!/bin/bash

# Set environment variables
TRAIN_DATA_DIR="../test_dataset/dataset"

TRAIN_DIR="${TRAIN_DATA_DIR}/train/images/"
VAL_DIR="${TRAIN_DATA_DIR}/val/images/"
TEST_DIR="${TRAIN_DATA_DIR}/test/images/"
OUTPUT_BASE_DIR="../output"
MODEL_NAME="unset_model.keras"
IMG_HEIGHT=512
IMG_WIDTH=512
EPOCHS=30
BATCH_SIZE=8

# Generate experiment name
DATE=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="${DATE}_${MODEL_NAME}_${EPOCHS}_epochs"
EXPERIMENT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
LOGS_DIR="${EXPERIMENT_DIR}/logs"

# Create directories for the experiment
mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$LOGS_DIR"

# Copy this script to the experiment folder
cp "$0" "${EXPERIMENT_DIR}/"

# Set the log file
LOG_FILE="${LOGS_DIR}/training_log.txt"

# Run the Python script
python ../train.py \
    --train "$TRAIN_DIR" \
    --val "$VAL_DIR" \
    --test "$TEST_DIR" \
    --output "$EXPERIMENT_DIR" \
    --name "$MODEL_NAME" \
    --img_height "$IMG_HEIGHT" \
    --img_width "$IMG_WIDTH" \
    --epoch "$EPOCHS" \
    --batch "$BATCH_SIZE" 2>&1 | tee "$LOG_FILE"

# Split the log file into 1000KB parts
split -b 1000k -d --additional-suffix=".txt" "$LOG_FILE" "${LOG_FILE}_part_"

# Remove the original log file
rm "$LOG_FILE"

echo "Autoencoder training script executed successfully. Results are saved in $EXPERIMENT_DIR."
