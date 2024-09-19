import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


class FeatureVisualizer:
    def __init__(self, model, layer_name):
        """
        Initialize the FeatureVisualizer with the model and the layer name to visualize.

        :param model: The trained UNet autoencoder model.
        :param layer_name: The name of the layer to visualize features from.
        """
        self.model = model
        self.layer_name = layer_name
        self.feature_extractor = Model(
            inputs=model.input, outputs=model.get_layer(layer_name).output
        )

    def get_features(self, img):
        """
        Extract features from the specified layer for the given image.

        :param img: The input image to extract features from.
        :return: The extracted feature map.
        """
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        features = self.feature_extractor.predict(img)
        return features[0]  # Remove batch dimension

    def visualize_features(
        self, img, output_dir, img_name, target_size=(256, 256), cmap="viridis"
    ):
        """
        Visualize features for the given image and save the result.

        :param img: The input image to extract features from.
        :param output_dir: The directory to save the visualized feature images.
        :param img_name: The name of the image file.
        :param target_size: The target size for resizing the input image.
        :param cmap: The colormap to use for visualizing the feature map.
        """
        # Extract features
        features = self.get_features(img)

        # Overlay feature maps on the original image
        feature_map_mean = np.mean(features, axis=-1)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.imshow(feature_map_mean, cmap=cmap, alpha=0.5)  # Adjust alpha for overlay
        plt.axis("off")

        # Save the overlay visualization
        overlay_output_path = f"{output_dir}/overlay_{self.layer_name}_{img_name}.png"
        plt.savefig(overlay_output_path, bbox_inches="tight")
        plt.close()

        print(f"Overlay visualization saved to {overlay_output_path}")
        return overlay_output_path


class UNetAutoEncoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        # Define your UNet model here
        # This function should return a compiled Keras model
        pass

    def load_weights(self, model_path):
        self.model.load_weights(model_path)

    def get_bottleneck_model(self):
        # This function should return the bottleneck model
        pass


class FeatureAnalysisManager:
    def __init__(self, model_path, image_dir, output_dir):
        self.model_path = model_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model, self.input_shape = self.load_model()

    def load_model(self):
        autoencoder = tf.keras.models.load_model(self.model_path, compile=False)
        input_shape = autoencoder.input_shape[1:]
        unet = UNetAutoEncoder(input_shape)
        unet.load_weights(self.model_path)
        bottleneck_model = unet.get_bottleneck_model()

        return bottleneck_model, input_shape

    def process_and_visualize(
        self, layer_name="conv2d", target_size=(256, 256), cmap="viridis", max_images=5
    ):
        visualizer = FeatureVisualizer(self.model, layer_name)

        image_paths = [
            os.path.join(self.image_dir, fname)
            for fname in os.listdir(self.image_dir)
            if fname.endswith((".png", ".jpg", ".jpeg"))
        ]

        processed_images = []
        for img_path in image_paths[:max_images]:
            img_name = os.path.basename(img_path).split(".")[0]
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img) / 255.0
            overlay_path = visualizer.visualize_features(
                img_array, self.output_dir, img_name, target_size, cmap
            )
            processed_images.append(overlay_path)

        # Create a tile of images
        n_images = len(processed_images)
        n_cols = 2  # Number of columns for the tile
        n_rows = np.ceil(n_images / n_cols).astype(int)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))

        for i, overlay_path in enumerate(processed_images):
            img = image.load_img(overlay_path)
            ax = axes[i // n_cols, i % n_cols]
            ax.imshow(img)
            ax.axis("off")

        # Remove empty subplots
        for i in range(n_images, n_rows * n_cols):
            fig.delaxes(axes.flatten()[i])

        # Save the tile of images
        tile_output_path = os.path.join(self.output_dir, "tile_visualization.png")
        plt.savefig(tile_output_path, bbox_inches="tight")
        plt.close()
        print(f"Tile visualization saved to {tile_output_path}")


# Example usage:
# feature_analysis_manager = FeatureAnalysisManager('path_to_model.h5', 'path_to_image_directory', 'path_to_output_directory')
# feature_analysis_manager.process_and_visualize('layer_name')
