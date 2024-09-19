import argparse

import tensorflow as tf

from features.cluster_features import FeatureClusterer
from features.extract_features_bovw import BoVWExtractor
from features.extract_features_cnn import CNNFeatureExtractor
from features.extract_features_hog import HOGFeatureExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Feature Extraction and Clustering"
    )
    parser.add_argument(
        "--method",
        choices=["bovw", "cnn", "hog", "cluster"],
        required=True,
        help="Feature extraction method to use.",
    )
    parser.add_argument(
        "--image_dir", help="Directory containing images for feature extraction."
    )
    parser.add_argument(
        "--output_csv", help="Output CSV file to save extracted features."
    )
    parser.add_argument(
        "--output_hdf5", help="Output HDF5 file to save extracted features."
    )
    parser.add_argument(
        "--input_hdf5", help="Input HDF5 file containing features for clustering."
    )
    parser.add_argument("--output_dir", help="Directory to save outputs.")
    parser.add_argument(
        "--n_clusters", type=int, help="Number of clusters for feature clustering."
    )
    parser.add_argument(
        "--n_samples", type=int, help="Number of representative samples to save."
    )
    parser.add_argument(
        "--model_path",
        help="Path to the model file. If not provided, the model must be passed in the constructor.",
    )

    args = parser.parse_args()

    if args.method == "bovw":
        extractor = BoVWExtractor(
            image_dir=args.image_dir,
            output_csv=args.output_csv,
            n_clusters=args.n_clusters,
        )
        images, image_paths = extractor.load_images_from_folder()
        features = extractor.extract_features_bovw(images)
        extractor.save_features_to_csv(features, image_paths)

    elif args.method == "cnn":
        if not args.model_path:
            raise ValueError("Model path must be provided for CNN method.")
        model = tf.keras.models.load_model(args.model_path)
        extractor = CNNFeatureExtractor(
            image_dir=args.image_dir,
            output_csv=args.output_csv,
            output_hdf5=args.output_hdf5,
            model=model,
        )
        extractor.extract_and_save_features()

    elif args.method == "hog":
        extractor = HOGFeatureExtractor(
            image_dir=args.image_dir, output_csv=args.output_csv
        )
        extractor.run()

    elif args.method == "cluster":
        clusterer = FeatureClusterer(
            input_hdf5=args.input_hdf5,
            output_hdf5=args.output_hdf5,
            output_csv=args.output_csv,
            method=args.method,
            n_clusters=args.n_clusters,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
        )
        clusterer.cluster_features()


if __name__ == "__main__":
    main()
