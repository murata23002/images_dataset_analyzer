#!/bin/bash
export PYTHONPATH="$(dirname "$(pwd)"):$PYTHONPATH"

OUTPUT_BASE_DIR="../output"
INPUT_HDF5="../dist/features/extract_unet_autoencoder.h5"

# Generate experiment name
DATE=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="${DATE}_feature_clustering"
EXPERIMENT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
LOGS_DIR="${EXPERIMENT_DIR}/logs"

# Create directories for the experiment
mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$LOGS_DIR"

# Copy this script to the experiment folder
cp "$0" "${EXPERIMENT_DIR}/"

# Set the log file
LOG_FILE="${LOGS_DIR}/training_log.txt"

METHOD="umap" # tsen pca
# Run the Python script
python ../features/cluster_features.py \
    --input_hdf5 "$INPUT_HDF5" \
    --output_dir "$EXPERIMENT_DIR" \
    --method "$METHOD" \
    --n_clusters 4 \
    --n_samples 100 2>&1 | tee "$LOG_FILE"

# Split the log file into 1000KB parts
split -b 1000k -d "$LOG_FILE" "${LOG_FILE}_part_"

# Remove the original log file
rm "$LOG_FILE"

echo "Feature clustering script executed successfully. Results are saved in $EXPERIMENT_DIR."
