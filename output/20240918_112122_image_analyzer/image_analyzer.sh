#!/bin/bash
export PYTHONPATH="$(dirname "$(pwd)"):$PYTHONPATH"

OUTPUT_BASE_DIR="../output"
INPUT_HDF5="../dist/features/extract_unet_autoencoder.h5"
MODEL_PATH="../dist/models/unset_model.keras"
INPUT_IMAGES="../test_dataset/dataset/train/images"

# Generate experiment name
DATE=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="${DATE}_image_analyzer"
EXPERIMENT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
LOGS_DIR="${EXPERIMENT_DIR}/logs"

# Create directories for the experiment
mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$LOGS_DIR"

# Copy this script to the experiment folder
cp "$0" "${EXPERIMENT_DIR}/"

# Set the log file
LOG_FILE="${LOGS_DIR}/analysis_log.txt"

# Run the Python script
python ../image_analyzer.py \
    --input_hdf5 "$INPUT_HDF5" \
    --model "$MODEL_PATH" \
    --output_dir "$EXPERIMENT_DIR" \
    --img_directory "$INPUT_IMAGES" 2>&1 | tee "$LOG_FILE"

# Split the log file into 1000KB parts
split -b 1000k -d "$LOG_FILE" "${LOG_FILE}_part_"

# Remove the original log file
rm "$LOG_FILE"

echo "Feature analysis script executed successfully. Results are saved in $EXPERIMENT_DIR."
