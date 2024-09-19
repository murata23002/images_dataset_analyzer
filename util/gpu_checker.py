import tensorflow as tf


class GPUChecker:
    @staticmethod
    def check_gpus():
        num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
        print("Num GPUs Available: ", num_gpus)
        if num_gpus > 0:
            print("GPU Details: ", tf.config.experimental.list_physical_devices("GPU"))
        else:
            print("No GPUs found.")


# 実行スクリプトとして使用する場合
if __name__ == "__main__":
    GPUChecker.check_gpus()
