"""
Custom Keras layers needed to deserialize the EffViT-Hybrid model.

These layer definitions MUST match the ones used during training in
brain_tumor_classification_FIXED.ipynb, otherwise model.load_model()
will fail with "Unknown layer" errors.
"""
import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    """Standard pre-norm ViT encoder block."""

    def __init__(self, embed_dim=192, num_heads=6, mlp_dim=768, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        h = self.norm1(x)
        h = self.attn(h, h, training=training)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h, training=training)
        x = x + h
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate,
        })
        return config


class AddPositionalEmbedding(layers.Layer):
    """Prepends a CLS token and adds learnable positional embedding."""

    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="cls_token",
        )
        self.pos_embed = self.add_weight(
            shape=(1, self.num_patches + 1, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embed",
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.embed_dim])
        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim,
        })
        return config


class ExtractCLSToken(layers.Layer):
    """Extracts the CLS token (index 0) from a sequence of tokens.

    Replaces the Lambda layer used during training. This is needed because
    Keras 3 cannot reliably deserialize Lambda layers across versions, and
    this explicit layer provides a compute_output_shape() that Keras can use.
    """

    def call(self, x):
        return x[:, 0]

    def compute_output_shape(self, input_shape):
        # (batch, num_patches+1, embed_dim) -> (batch, embed_dim)
        return (input_shape[0], input_shape[-1])


CUSTOM_OBJECTS = {
    "TransformerBlock": TransformerBlock,
    "AddPositionalEmbedding": AddPositionalEmbedding,
    "ExtractCLSToken": ExtractCLSToken,
}
