
# Automated Script Execution

This repository contains several Python scripts for various data processing and machine learning tasks. This README provides instructions on how to set up the environment and execute all scripts using a provided shell script.

## Prerequisites

- Python 3.6+
- `venv` for creating a virtual environment
- Required Python packages listed in `requirements.txt`

## Setup Instructions

1. **Clone the Repository**: Clone this repository to your local machine.
    ```bash
    git clone https://github.com/your-repository.git
    cd your-repository
    ```

2. **Create a Virtual Environment**: Create and activate a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. **Install Required Packages**: Install the necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the Scripts**: Ensure all the Python scripts are in the repository.

5. **Make the Shell Script Executable**: Grant execute permissions to the shell script.
    ```bash
    chmod +x run_all.sh
    ```

6. **Execute the Shell Script**: Run the shell script to execute all the Python scripts sequentially.
    ```bash
    ./run_all.sh
    ```

## Scripts Overview

1. **train.py**
    - Trains and evaluates an autoencoder model for image feature extraction.
    - **Usage**:
        ```bash
        python train.py --train path/to/train_data --test path/to/test_data --val path/to/val_data --output path/to/output
        ```

2. **auto_en.py**
    - Contains the definition of the autoencoder model.
    - No direct execution required.

3. **unet_model.py**
    - Contains the definition of the UNet autoencoder model.
    - No direct execution required.

4. **cluster_features.py**
    - Clusters features and saves representative samples to HDF5 and CSV files.
    - **Usage**:
        ```bash
        python cluster_features.py --input_hdf5 path/to/input.hdf5 --output_hdf5 path/to/output.hdf5 --output_csv path/to/output.csv --method umap --n_clusters 4 --n_samples 100 --output_dir ./dist
        ```

5. **detect_anomalies.py**
    - Detects anomalies in new image data using existing features.
    - **Usage**:
        ```bash
        python detect_anomalies.py --input_hdf5 path/to/input.hdf5 --directory path/to/new_images --output_dir ./output
        ```

6. **extract_features_bovw.py**
    - Extracts and plots BoVW features from images.
    - **Usage**:
        ```bash
        python extract_features_bovw.py --image_dir path/to/images --output_csv path/to/features_bovw.csv
        ```

7. **extract_features_cnn.py**
    - Extracts and plots CNN features from images.
    - **Usage**:
        ```bash
        python extract_features_cnn.py --image_dir path/to/images --output_csv path/to/features_cnn.csv --output_hdf5 path/to/features_cnn.hdf5
        ```

8. **extract_features_hog.py**
    - Extracts and plots HOG features from images.
    - **Usage**:
        ```bash
        python extract_features_hog.py --image_dir path/to/images --output_csv path/to/features_hog.csv
        ```

9. **test.py**
    - Runs a test script to standardize data, perform PCA, and classify components.
    - **Usage**:
        ```bash
        python test.py
        ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
