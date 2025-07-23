import tensorflow as tf
from tensorflow.keras import layers, models
from transformer_dna_classifier import PositionalEncoding  # reuse encoding layer
import keras

# Register the custom layer
keras.saving.register_keras_serializable()(PositionalEncoding)

# Configuration
TEMPERATURE = 2.0  # used for temperature scaling
NUM_CLASSES = 3


def build_uncertain_transformer(input_shape=(180, 1), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = layers.Dense(64)(inputs)
    x = PositionalEncoding()(x)

    for _ in range(2):  # Add 2 attention blocks with dropout inside
        attn_input = x
        x = layers.LayerNormalization()(x)
        x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Dropout(0.3)(x)
        x = layers.Add()([x, attn_input])

        ffn_input = x
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64)(x)
        x = layers.Add()([x, ffn_input])

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes)(x)
    outputs = tf.keras.layers.Activation(lambda x: tf.nn.softmax(x / TEMPERATURE))(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="transformer_dna_uncertain")
    return model


# To use in training script
if __name__ == "__main__":
    model = build_uncertain_transformer()
    model.summary()
