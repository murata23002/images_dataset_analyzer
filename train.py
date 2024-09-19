import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.optimizers import Adam

from model.autoencoder import AutoEncoderModel
from model.unet_model import UNetAutoEncoder
from util.send_notification import send_notification


class ImageProcessor:
    @staticmethod
    def set_random_seed(seed_value):
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

    @staticmethod
    def get_image_file_paths(folder):
        return [
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.endswith(".jpg") or fname.endswith(".png")
        ]

    @staticmethod
    def load_and_preprocess_image(file_path, img_height, img_width):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.cast(img, tf.float32) / 255.0
        return img

    @staticmethod
    def load_and_preprocess_image_with_target(file_path, img_height, img_width):
        img = ImageProcessor.load_and_preprocess_image(file_path, img_height, img_width)
        return img, img


class MetricsCalculator:
    @staticmethod
    def calculate_metrics(true_images, reconstructed_images):
        true_images_flat = tf.reshape(true_images, [-1]).numpy()
        reconstructed_images_flat = tf.reshape(reconstructed_images, [-1]).numpy()
        f1 = f1_score(true_images_flat > 0.5, reconstructed_images_flat > 0.5)
        precision = precision_score(
            true_images_flat > 0.5, reconstructed_images_flat > 0.5
        )
        recall = recall_score(true_images_flat > 0.5, reconstructed_images_flat > 0.5)
        return f1, precision, recall

    @staticmethod
    def save_comparison_images(original_images, reconstructed_images, save_path):
        n = min(5, len(original_images))
        plt.figure(figsize=(15, 5))
        for i in range(n):
            plt.subplot(2, n, i + 1)
            plt.imshow(original_images[i])
            plt.axis("off")
            plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed_images[i])
            plt.axis("off")
        plt.savefig(save_path)
        plt.close()


class AutoEncoderTrainer:
    def __init__(
        self,
        train_dir,
        val_dir,
        test_dir,
        output_dir,
        model_name,
        img_height,
        img_width,
        epochs,
        batch_size,
    ):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.img_height = img_height
        self.img_width = img_width
        self.epochs = epochs
        self.batch_size = batch_size
        self.strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        print(f"Number of devices: {self.strategy.num_replicas_in_sync}")

    def create_datasets(self):
        train_paths = ImageProcessor.get_image_file_paths(self.train_dir)
        val_paths = ImageProcessor.get_image_file_paths(self.val_dir)
        test_paths = ImageProcessor.get_image_file_paths(self.test_dir)

        train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_paths)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)

        train_dataset = train_dataset.map(
            lambda x: ImageProcessor.load_and_preprocess_image_with_target(
                x, self.img_height, self.img_width
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        val_dataset = val_dataset.map(
            lambda x: ImageProcessor.load_and_preprocess_image_with_target(
                x, self.img_height, self.img_width
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        test_dataset = test_dataset.map(
            lambda x: ImageProcessor.load_and_preprocess_image_with_target(
                x, self.img_height, self.img_width
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        train_dataset = train_dataset.batch(self.batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        val_dataset = val_dataset.batch(self.batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        test_dataset = test_dataset.batch(self.batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        return train_dataset, val_dataset, test_dataset

    def train_and_evaluate(self):
        ImageProcessor.set_random_seed(42)
        train_dataset, val_dataset, test_dataset = self.create_datasets()

        try:
            send_notification("Training started.")
            with self.strategy.scope():
                input_shape = (self.img_height, self.img_width, 3)
                model_builder = UNetAutoEncoder(input_shape)
                autoencoder = model_builder.get_model()
                autoencoder.compile(
                    optimizer=Adam(learning_rate=0.001), loss="mean_squared_error"
                )

            autoencoder.fit(
                train_dataset, validation_data=val_dataset, epochs=self.epochs
            )
            autoencoder.save(os.path.join(self.output_dir, self.model_name))

            test_loss = autoencoder.evaluate(test_dataset)
            print(f"Test Loss: {test_loss}")

            all_true_images = []
            all_reconstructed_images = []
            batch_count = 0

            for test_images, _ in test_dataset:
                batch_count += 1
                reconstructed_images = autoencoder.predict(test_images)
                all_true_images.append(test_images)
                all_reconstructed_images.append(reconstructed_images)
                comparison_image_path = os.path.join(
                    self.output_dir, f"{batch_count}_comparison_images.png"
                )
                MetricsCalculator.save_comparison_images(
                    test_images, reconstructed_images, comparison_image_path
                )
                break

            all_true_images = tf.concat(all_true_images, axis=0)
            all_reconstructed_images = tf.concat(all_reconstructed_images, axis=0)
            f1, precision, recall = MetricsCalculator.calculate_metrics(
                all_true_images, all_reconstructed_images
            )
            print(f"Overall F1 Score: {f1}, Precision: {precision}, Recall: {recall}")

            send_notification(
                f"Training completed successfully. Test Loss: {test_loss}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}"
            )

        except Exception as e:
            send_notification(f"Training failed: {str(e)}")
            print(f"Training failed: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate an autoencoder model for image feature extraction."
    )
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to the folder containing the training images.",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to the folder containing the test images.",
    )
    parser.add_argument(
        "--val",
        type=str,
        required=True,
        help="Path to the folder containing the validation images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the folder where comparison images will be saved.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="autoencoder_model.keras",
        help="Name of the output model file.",
    )
    parser.add_argument(
        "--img_height", type=int, default=512, help="Height of the input images."
    )
    parser.add_argument(
        "--img_width", type=int, default=512, help="Width of the input images."
    )
    parser.add_argument(
        "--epoch", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="Batch size for training."
    )

    args = parser.parse_args()
    trainer = AutoEncoderTrainer(
        train_dir=args.train,
        val_dir=args.val,
        test_dir=args.test,
        output_dir=args.output,
        model_name=args.name,
        img_height=args.img_height,
        img_width=args.img_width,
        epochs=args.epoch,
        batch_size=args.batch,
    )
    trainer.train_and_evaluate()
