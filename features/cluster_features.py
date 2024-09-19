import argparse
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from util.cluster_analyzer import ClusterAnalyzer
from util.hdf5_handler import HDF5Handler


class FeaturePlotter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_clusters_with_images(
        self, features, labels, cluster_centers, image_paths, method
    ):
        print("Plotting clusters with images...")
        start_time = time.time()

        # Plot the clusters in a 2D space
        plt.figure(figsize=(10, 8))

        # Assuming the features are already reduced to 2D for plotting (e.g., using UMAP or t-SNE)
        for cluster_idx in np.unique(labels):
            cluster_points = features[labels == cluster_idx]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f"Cluster {cluster_idx}",
            )

        # Plot cluster centers
        plt.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            color="black",
            marker="x",
            s=100,
            label="Centers",
        )

        plt.title(f"Clusters visualization using {method}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "clusters.png"))
        plt.close()

        print(
            f"Clusters plotted and saved to {os.path.join(self.output_dir, 'clusters.png')}"
        )
        print(f"Plotting completed in {time.time() - start_time:.2f} seconds.")

    def visualize_clusters(
        self, representative_samples, representative_paths, labels, method
    ):
        print("Visualizing clusters...")
        start_time = time.time()

        # Create a directory for representative samples
        rep_dir = os.path.join(self.output_dir, "representative_samples")
        if not os.path.exists(rep_dir):
            os.makedirs(rep_dir)

        for i, (sample, path) in tqdm(
            enumerate(zip(representative_samples, representative_paths)),
            total=len(representative_samples),
            desc="Saving representative samples",
        ):
            cluster_idx = labels[i]
            img = Image.open(path)
            img.save(os.path.join(rep_dir, f"cluster_{cluster_idx}_sample_{i}.png"))

        print(f"Representative samples have been saved to {rep_dir}")
        print(f"Visualization completed in {time.time() - start_time:.2f} seconds.")


class FeatureClusterer:
    def __init__(self, input_hdf5, method, n_clusters, n_samples, output_dir):
        self.input_hdf5 = input_hdf5
        self.method = method
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.output_dir = output_dir

    def cluster_features(self):
        print(f"Loading features and image paths from HDF5 file: {self.input_hdf5}")
        start_time = time.time()

        # Load features and image paths from HDF5
        features, image_paths = HDF5Handler.load_features_from_hdf5(self.input_hdf5)
        print("Features and image paths loaded successfully.")

        print(f"Performing clustering with {self.n_clusters} clusters.")
        # Perform clustering
        analyzer = ClusterAnalyzer(self.n_clusters)
        labels, cluster_centers = analyzer.cluster_features(features)
        print("Clustering completed.")
        print(
            f"Cluster labels and centers computed in {time.time() - start_time:.2f} seconds."
        )

        # Plot clusters with images
        plotter = FeaturePlotter(self.output_dir)
        plotter.plot_clusters_with_images(
            features, labels, cluster_centers, image_paths, self.method
        )

        print("Finding and visualizing representative samples.")
        start_time = time.time()

        # Find and visualize representative samples
        representative_samples, representative_paths = (
            analyzer.find_representative_samples(
                features, labels, cluster_centers, image_paths, n_samples=self.n_samples
            )
        )

        plotter.visualize_clusters(
            representative_samples, representative_paths, labels, self.method
        )

        print(
            f"Representative samples found and visualized in {time.time() - start_time:.2f} seconds."
        )
        print("Representative samples have been saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster features and save representative samples to HDF5."
    )
    parser.add_argument(
        "--input_hdf5", required=True, help="Input HDF5 file containing features."
    )
    parser.add_argument(
        "--output_dir", default="./dist", help="Directory to save output files"
    )
    parser.add_argument("--method", default="umap", help="Extract features method")
    parser.add_argument(
        "--n_clusters", type=int, default=4, help="Number of clusters for k-means."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of representative samples to save for each cluster.",
    )

    args = parser.parse_args()

    clusterer = FeatureClusterer(
        args.input_hdf5, args.method, args.n_clusters, args.n_samples, args.output_dir
    )
    clusterer.cluster_features()
