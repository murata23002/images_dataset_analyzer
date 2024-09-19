import h5py
import numpy as np


class HDF5Handler:
    @classmethod
    def load_features_from_hdf5(cls, input_hdf5):
        with h5py.File(input_hdf5, "r") as f:
            features = f["features"][:]
            image_paths = [path.decode("utf-8") for path in f["image_paths"][:]]
        return features, image_paths

    @classmethod
    def save_features_to_hdf5(cls, features, image_paths, output_hdf5):
        with h5py.File(output_hdf5, "w") as f:
            f.create_dataset("features", data=features)
            f.create_dataset("image_paths", data=np.string_(image_paths))
