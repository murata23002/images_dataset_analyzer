import os

import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm


class ImageProcessor:
    def __init__(self, model, target_size=(224, 224), batch_size=32, img_directory=""):
        self.model = model
        self.target_size = target_size
        self.batch_size = batch_size
        self.img_directory = img_directory

    def preprocess_image(self, img_path):
        img = load_img(img_path, target_size=self.target_size)
        img_data = img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        return preprocess_input(img_data)

    def extract_features_from_folder(self):
        all_features = []
        image_paths = []
        dir = self.img_directory

        for filename in tqdm(os.listdir(dir), desc="Processing images"):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(dir, filename)
                image_paths.append(img_path)
                img_array = self.preprocess_image(img_path)
                features = self.model.predict(img_array)
                all_features.append(features.flatten())

        all_features = np.array(all_features)
        return all_features, image_paths

    # def process_batch(self, batch_images):
    #     batch_images = np.vstack(batch_images)
    #     features = self.model.predict(batch_images)
    #     return features.reshape(len(batch_images), -1)

    # def extract_features_from_folder(self):
    #     all_features = []
    #     image_paths = []
    #     batch_images = []
    #     dir = self.img_directory

    #     for filename in tqdm(os.listdir(dir), desc="Processing images"):
    #         if filename.lower().endswith((".jpg", ".jpeg", ".png")):
    #             img_path = os.path.join(dir, filename)
    #             image_paths.append(img_path)
    #             img_array = self.preprocess_image(img_path)
    #             batch_images.append(img_array)

    #             if len(batch_images) == self.batch_size:
    #                 features = self.process_batch(batch_images)
    #                 all_features.extend(features)
    #                 batch_images = []

    #     if batch_images:
    #         features = self.process_batch(batch_images)
    #         all_features.extend(features)

    #     all_features = np.array(all_features)
    #     return all_features, image_paths
