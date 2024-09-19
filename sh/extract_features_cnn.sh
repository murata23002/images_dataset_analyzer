#!/bin/bash
export PYTHONPATH="$(dirname "$(pwd)"):$PYTHONPATH"

OUTPUT_BASE_DIR="../output"
MODEL_PATH="../dist/models/unset_model.keras"
INPUT_IMAGES="../test_dataset//dataset//train/images"
OUTPUT_CSV="extract_unet_autoencoder.csv"
OUTPUT_HDF5="extract_unet_autoencoder.h5"

# Generate experiment name
DATE=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="${DATE}_extract_features_cnn"
EXPERIMENT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
LOGS_DIR="${EXPERIMENT_DIR}/logs"

# Create directories for the experiment
mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$LOGS_DIR"

# Copy this script to the experiment folder
cp "$0" "${EXPERIMENT_DIR}/"

# Set the log file
LOG_FILE="${LOGS_DIR}/extract_log.txt"

# Run the Python script
python ../features/extract_features_cnn.py \
    --img_directory "$INPUT_IMAGES" \
    --output_dir "$EXPERIMENT_DIR" \
    --output_csv "$OUTPUT_CSV" \
    --output_hdf5 "$OUTPUT_HDF5" \
    --model "$MODEL_PATH" 2>&1 | tee "$LOG_FILE"

# Split the log file into 1000KB parts
split -b 1000k -d "$LOG_FILE" "${LOG_FILE}_part_"

# Remove the original log file
rm "$LOG_FILE"

echo "Feature extract script executed successfully. Results are saved in $EXPERIMENT_DIR."