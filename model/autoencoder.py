from tensorflow.keras.layers import Dense, Flatten, Input, Reshape
from tensorflow.keras.models import Model


class AutoEncoderModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Encoder
        x = Flatten()(inputs)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        latent = Dense(32, activation="relu")(x)

        # Decoder
        x = Dense(64, activation="relu")(latent)
        x = Dense(128, activation="relu")(x)
        x = Dense(
            self.input_shape[0] * self.input_shape[1] * self.input_shape[2],
            activation="sigmoid",
        )(x)
        outputs = Reshape(self.input_shape)(x)

        model = Model(inputs, outputs)
        return model

    def get_model(self):
        return self.model
