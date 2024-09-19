import h5py


class HDF5Inspector:
    @staticmethod
    def inspect_file(hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            print("Datasets in the file:")

            def print_attrs(name, obj):
                print(name)

            f.visititems(print_attrs)


# 実行スクリプトとして使用する場合
if __name__ == "__main__":
    input_hdf5 = "features_cnn.hdf5"
    HDF5Inspector.inspect_file(input_hdf5)
