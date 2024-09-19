import argparse
import os

import tensorflow as tf

from model.unet_model import UNetAutoEncoder
from util.hdf5_handler import HDF5Handler
from util.image_processor import ImageProcessor


class CNNFeatureExtractor:
    def __init__(self, image_dir, output_dir, output_csv, output_hdf5, model_path):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.output_csv = output_csv
        self.output_hdf5 = output_hdf5
        self.model_path = model_path

    def load_model(self):
        autoencoder = tf.keras.models.load_model(self.model_path, compile=False)
        input_shape = autoencoder.input_shape[1:]
        unet = UNetAutoEncoder(input_shape)
        unet.load_weights(self.model_path)
        bottleneck_model = unet.get_bottleneck_model()

        return bottleneck_model, input_shape

    def extract_and_save_features(self):
        os.makedirs(self.output_dir, exist_ok=True)
        model, input_shape = self.load_model()

        img_process = ImageProcessor(
            model=model,
            target_size=input_shape,
            batch_size=32,
            img_directory=self.image_dir,
        )

        features, image_paths = img_process.extract_features_from_folder()

        if len(features) == 0:
            print("No valid images found in the specified directory.")
            return

        output_hdf5 = os.path.join(self.output_dir, self.output_hdf5)
        HDF5Handler.save_features_to_hdf5(features, image_paths, output_hdf5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from image data.")
    parser.add_argument(
        "--img_directory", required=True, help="Directory containing new image data."
    )
    parser.add_argument(
        "--output_dir", default="./output", help="Directory to save output files"
    )
    parser.add_argument("--output_csv", required=True, help="Output CSV file name")
    parser.add_argument("--output_hdf5", required=True, help="Output HDF5 file name")
    parser.add_argument(
        "--model", required=True, help="The path to the model for feature extraction"
    )

    args = parser.parse_args()

    cnn = CNNFeatureExtractor(
        image_dir=args.img_directory,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        output_hdf5=args.output_hdf5,
        model_path=args.model,
    )

    cnn.extract_and_save_features()
