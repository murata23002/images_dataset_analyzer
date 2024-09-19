import argparse
import csv
import os
from multiprocessing import Process

import tensorflow as tf

from convex_hull_analysis.convex_hull_analysis_manager import ConvexHullAnalysisManager
from model.unet_model import UNetAutoEncoder
from util.hdf5_handler import HDF5Handler
from util.image_processor import ImageProcessor


class ImageAnalyzer:
    def __init__(
        self, input_hdf5, img_directory, output_dir, model_path="autoencoder.h5"
    ):
        self.input_hdf5 = input_hdf5
        self.img_directory = img_directory
        self.output_dir = output_dir
        self.model_path = model_path

    def load_model(self):
        autoencoder = tf.keras.models.load_model(self.model_path, compile=False)
        input_shape = autoencoder.input_shape[1:]
        unet = UNetAutoEncoder(input_shape)
        unet.load_weights(self.model_path)
        bottleneck_model = unet.get_bottleneck_model()

        return bottleneck_model, input_shape

    def write_results_to_csv(self, results, filename, method):
        output_file = os.path.join(self.output_dir, f"{filename}_{method}_results.csv")
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Index", "Status", "Image Path"])
            for row in results:
                writer.writerow(row)

    def detect(self):
        os.makedirs(self.output_dir, exist_ok=True)
        model, input_shape = self.load_model()
        img_process = ImageProcessor(
            model=model,
            target_size=input_shape,
            batch_size=32,
            img_directory=self.img_directory,
        )

        features, image_paths_original = HDF5Handler.load_features_from_hdf5(
            self.input_hdf5
        )
        new_features, new_image_paths = img_process.extract_features_from_folder()
        filename = os.path.basename(self.img_directory)
        con_hull_manager = ConvexHullAnalysisManager(self.output_dir)

        for method in ["pca", "tsne", "umap"]:
            results = con_hull_manager.run_analysis(
                features,
                new_features,
                image_paths_original,
                new_image_paths,
                method=method,
            )
            self.write_results_to_csv(results, filename, method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect anomalies in new image data.")
    parser.add_argument(
        "--input_hdf5", required=True, help="Input HDF5 file containing features."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="The path to the model for feature extraction",
    )
    parser.add_argument(
        "--img_directory", required=True, help="Directory containing new image data."
    )
    parser.add_argument(
        "--output_dir", default="./output", help="Directory to save output files"
    )

    args = parser.parse_args()

    detector = ImageAnalyzer(
        input_hdf5=args.input_hdf5,
        img_directory=args.img_directory,
        output_dir=args.output_dir,
        model_path=args.model,
    )
    p = Process(target=detector.detect)
    p.start()
    p.join()
