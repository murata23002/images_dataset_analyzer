import tensorflow as tf


class AutoencoderModelLoader:
    def __init__(self, model_path, model_class, *model_args, **model_kwargs):
        """
        :param model_path: Path to the saved autoencoder model.
        :param model_class: The class of the model to load.
        :param model_args: Positional arguments to initialize the model.
        :param model_kwargs: Keyword arguments to initialize the model.
        """
        self.model_path = model_path
        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def load_model(self):
        """
        Loads the autoencoder model and its weights.

        :return: Tuple of (model, input_shape)
        """
        autoencoder = tf.keras.models.load_model(self.model_path)
        input_shape = autoencoder.input_shape[1:]
        model = self.model_class(
            input_shape, *self.model_args, **self.model_kwargs
        ).get_model()
        model.set_weights(autoencoder.get_weights())
        return model, input_shape


# Example usage:
# from some_module import UNetAutoEncoder
# model_loader = AutoencoderModelLoader('path/to/autoencoder', UNetAutoEncoder)
# model, input_shape = model_loader.load_model()
