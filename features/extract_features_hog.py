import argparse
import os

import cv2
import numpy as np
import pandas as pd
from skimage import feature
from tqdm import tqdm
from utils.feature_plotting import plot_pca, plot_tsne, plot_umap


class HOGFeatureExtractor:
    def __init__(self, image_dir, output_csv, target_size=(416, 416)):
        self.image_dir = image_dir
        self.output_csv = output_csv
        self.target_size = target_size

    def load_images_from_folder(self):
        images = []
        image_paths = []
        for filename in tqdm(os.listdir(self.image_dir), desc="Loading images"):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(self.image_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, self.target_size)
                    img_resized = img_resized.astype("float32") / 255.0
                    images.append(img_resized)
                    image_paths.append(img_path)
        return np.array(images), image_paths

    def extract_features_hog(self, images):
        all_hog_features = []
        for image in tqdm(images, desc="Extracting HOG features"):
            hog_features, hog_image = feature.hog(image, visualize=True)
            all_hog_features.append(hog_features)
        return np.array(all_hog_features)

    def save_features_to_csv(self, features, image_paths):
        df = pd.DataFrame(features)
        df["image_path"] = image_paths
        df.to_csv(self.output_csv, index=False)

    def plot_features(self, features, task, output_dir):
        plot_pca(features, output_dir, task, title="PCA of HOG Features")
        plot_tsne(features, output_dir, task, title="t-SNE of HOG Features")
        plot_umap(features, output_dir, task, title="UMAP of HOG Features")

    def run(self):
        images, image_paths = self.load_images_from_folder()
        if len(images) == 0:
            print("No valid images found in the specified directory.")
            return

        hog_features = self.extract_features_hog(images)
        self.save_features_to_csv(hog_features, image_paths)
        self.plot_features(hog_features, "HOG", "./dist")


# 実行スクリプトとして使用する場合
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and plot HOG features from images."
    )
    parser.add_argument(
        "--image_dir", required=True, help="Directory containing the images."
    )
    parser.add_argument(
        "--output_csv", default="features_hog.csv", help="Output CSV file for features."
    )

    args = parser.parse_args()

    extractor = HOGFeatureExtractor(args.image_dir, args.output_csv)
    extractor.run()
