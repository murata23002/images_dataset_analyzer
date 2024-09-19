import gc

import tensorflow as tf


class SessionManager:
    @staticmethod
    def clear_session():
        tf.keras.backend.clear_session()
        gc.collect()


if __name__ == "__main__":
    SessionManager.clear_session()
