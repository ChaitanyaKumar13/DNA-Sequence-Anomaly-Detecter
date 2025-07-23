import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers, models
import numpy as np
from keras.saving import register_keras_serializable

@register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "embed_dim": self.embed_dim
        })
        return config

    def call(self, x):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        pe = np.zeros((self.max_len, self.embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = tf.convert_to_tensor(pe, dtype=tf.float32)
        return x + pe

def build_transformer_model(input_shape=(180, 1), num_classes=3):
    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    x = layers.Dense(64)(inputs)
    x = PositionalEncoding(max_len=input_shape[0], embed_dim=64)(x)

    # Encoder block 1
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)

    ffn = layers.Dense(128, activation='relu')(x)
    ffn = layers.Dense(64)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.Dropout(0.1)(x)

    # Encoder block 2
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)

    ffn = layers.Dense(128, activation='relu')(x)
    ffn = layers.Dense(64)(ffn)
    x = layers.Add()([x, ffn])

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
