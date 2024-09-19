import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)
from tensorflow.keras.models import Model


class UNetAutoEncoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model, self.bottleneck_model = self.build_model()

    def conv_block(self, inputs, filters, kernel_size=(3, 3), padding="same"):
        conv = Conv2D(filters, kernel_size, padding=padding)(inputs)
        conv = Activation("relu")(conv)
        conv = Conv2D(filters, (1, 1), padding=padding)(conv)  # 1x1 convolution
        conv = Activation("relu")(conv)
        return conv

    def encoder_block(self, inputs, filters):
        conv = self.conv_block(inputs, filters)
        pool = MaxPooling2D((2, 2))(conv)
        return conv, pool

    def decoder_block(self, inputs, skip_features, filters):
        up = UpSampling2D((2, 2))(inputs)
        concat = concatenate([up, skip_features])
        conv = self.conv_block(concat, filters)
        return conv

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Encoder (reduced layers and filters)
        c1, p1 = self.encoder_block(inputs, 16)
        c2, p2 = self.encoder_block(p1, 32)
        c3, p3 = self.encoder_block(p2, 64)
        c4, p4 = self.encoder_block(p3, 128)

        # Bottleneck (reduced channels)
        bottleneck = self.conv_block(p4, 8)

        # Decoder (reduced layers and filters)
        d1 = self.decoder_block(bottleneck, c4, 128)
        d2 = self.decoder_block(d1, c3, 64)
        d3 = self.decoder_block(d2, c2, 32)
        d4 = self.decoder_block(d3, c1, 16)

        outputs = Conv2D(3, (1, 1), activation="sigmoid")(d4)

        model = Model(inputs, outputs)

        # Bottleneck model
        bottleneck_model = Model(inputs, bottleneck)

        return model, bottleneck_model

    def get_model(self):
        return self.model

    def get_bottleneck_model(self):
        return self.bottleneck_model

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

        # Bottleneck model の重みを設定
        bottleneck_layers = [
            layer for layer in self.model.layers if layer.name.startswith("bottleneck")
        ]
        bottleneck_weights = [
            self.model.get_layer(layer.name).get_weights()
            for layer in bottleneck_layers
        ]

        # Bottleneck model の重みを設定
        for layer, weights in zip(self.bottleneck_model.layers, bottleneck_weights):
            layer.set_weights(weights)


# 使用例
if __name__ == "__main__":
    input_shape = (128, 128, 3)
    unet = UNetAutoEncoder(input_shape)
    model = unet.get_model()

    bottleneck_model = unet.get_bottleneck_model()
    sample_input = tf.random.normal([1, 128, 128, 3])
    model.save_weights("unet_weights.h5")
    unet.load_weights("unet_weights.h5")

    bottleneck_output = bottleneck_model(sample_input)
    print("Bottleneck output shape:", bottleneck_output.shape)
