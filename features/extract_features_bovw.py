import argparse
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from plot.feature_plotting import plot_pca, plot_tsne, plot_umap


class BoVWExtractor:
    def __init__(self, image_dir, output_csv, n_clusters=100):
        self.image_dir = image_dir
        self.output_csv = output_csv
        self.n_clusters = n_clusters

    def load_images_from_folder(self):
        images = []
        image_paths = []
        for filename in tqdm(os.listdir(self.image_dir), desc="Loading images"):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(self.image_dir, filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray)
                image_paths.append(img_path)
        return images, image_paths

    def extract_features_bovw(self, images):
        sift = cv2.SIFT_create()
        descriptors_list = []

        for image in tqdm(images, desc="Extracting SIFT features"):
            keypoints, descriptors = sift.detectAndCompute(image, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)

        all_descriptors = np.vstack(descriptors_list)
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(all_descriptors)

        all_histograms = []
        for image in tqdm(images, desc="Computing histograms"):
            keypoints, descriptors = sift.detectAndCompute(image, None)
            histogram = np.zeros(self.n_clusters)
            if descriptors is not None:
                labels = kmeans.predict(descriptors)
                for label in labels:
                    histogram[label] += 1
            all_histograms.append(histogram)

        scaler = StandardScaler()
        all_histograms = scaler.fit_transform(all_histograms)
        return np.array(all_histograms)

    def plot_features(self, features, task, output_dir):
        plot_pca(features, output_dir, task, title="PCA of BoVW Features")
        plot_tsne(features, output_dir, task, title="t-SNE of BoVW Features")
        plot_umap(features, output_dir, task, title="UMAP of BoVW Features")

    def run(self):
        images, image_paths = self.load_images_from_folder()
        if len(images) == 0:
            print("No valid images found in the specified directory.")
            return

        bovw_features = self.extract_features_bovw(images)
        self.save_features_to_csv(bovw_features, image_paths)
        self.plot_features(bovw_features, "BoVW", "./dist")


# 実行スクリプトとして使用する場合
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and plot BoVW features from images."
    )
    parser.add_argument(
        "--image_dir", required=True, help="Directory containing the images."
    )
    parser.add_argument(
        "--output_csv",
        default="features_bovw.csv",
        help="Output CSV file for features.",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=100, help="Number of clusters for KMeans."
    )

    args = parser.parse_args()

    extractor = BoVWExtractor(args.image_dir, args.output_csv, args.n_clusters)
    extractor.run()
