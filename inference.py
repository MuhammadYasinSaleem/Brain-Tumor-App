"""
Inference for EffViT-Hybrid — weights-only loading approach.

The .keras file was saved with a Lambda layer whose bytecode is tied to
the Python version used during training. Loading it on a different Python
version causes "unknown opcode" errors. We work around this by rebuilding
the exact same architecture in pure Python (using ExtractCLSToken instead
of Lambda) and then loading ONLY the weights from the saved file.
"""
import io
import os
import zipfile
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image
import cv2

from custom_layers import TransformerBlock, AddPositionalEmbedding, ExtractCLSToken

IMG_SIZE = 224
EMBED_DIM = 192
NUM_HEADS = 6
NUM_LAYERS = 6
MLP_DIM = 768
NUM_CLASSES = 4

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

_model = None
_stem = None      # cached reference to the effnet_stem sub-model
_head_layers = None  # cached list of layers after the stem


def _build_effvit_hybrid():
    """Rebuild the exact EffViT-Hybrid architecture used during training,
    but with ExtractCLSToken instead of Lambda."""
    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,
    )
    base.trainable = False
    cnn_features = base.get_layer("block5c_add").output
    cnn_backbone = Model(base.input, cnn_features, name="effnet_stem")

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = cnn_backbone(inputs, training=False)
    x = layers.Conv2D(EMBED_DIM, kernel_size=1, padding="same",
                      name="patch_proj")(x)
    num_patches = 14 * 14
    x = layers.Reshape((num_patches, EMBED_DIM))(x)
    x = AddPositionalEmbedding(num_patches, EMBED_DIM,
                               name="pos_embed_layer")(x)
    for i in range(NUM_LAYERS):
        x = TransformerBlock(embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                             mlp_dim=MLP_DIM, dropout=0.1,
                             name=f"transformer_{i}")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = ExtractCLSToken(name="cls_extract")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="EffViT_Hybrid")
    return model


def _load_weights_from_keras_file(model, keras_path):
    """Extract weights h5 from .keras zip archive and load by name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(keras_path, "r") as z:
            z.extractall(tmpdir)
        weights_file = os.path.join(tmpdir, "model.weights.h5")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"model.weights.h5 not found inside {keras_path}."
            )
        model.load_weights(weights_file, skip_mismatch=True)


def load_model(model_path: str):
    """Build architecture + load trained weights. Cached after first call."""
    global _model, _stem, _head_layers
    if _model is None:
        print("Building EffViT-Hybrid architecture ...")
        _model = _build_effvit_hybrid()
        print(f"Architecture built. Params: {_model.count_params():,}")

        print(f"Loading weights from {model_path} ...")
        _load_weights_from_keras_file(_model, model_path)
        print("Weights loaded successfully.")

        # Cache the stem and head layers for Grad-CAM
        _stem = _model.get_layer("effnet_stem")
        _head_layers = []
        found_stem = False
        for layer in _model.layers:
            if layer.name == "effnet_stem":
                found_stem = True
                continue
            if found_stem and "input" not in layer.name.lower():
                _head_layers.append(layer)

        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        _ = _model.predict(dummy, verbose=0)
        print("Model warmed up.")
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(image_bytes: bytes, model_path: str):
    model = load_model(model_path)
    x = preprocess_image(image_bytes)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    sorted_probs = sorted(
        [
            {
                "class": CLASS_NAMES[i],
                "name": DISPLAY_NAMES[CLASS_NAMES[i]],
                "prob": float(probs[i]),
            }
            for i in range(len(CLASS_NAMES))
        ],
        key=lambda d: d["prob"],
        reverse=True,
    )
    return {
        "predicted_class": DISPLAY_NAMES[CLASS_NAMES[idx]],
        "predicted_key": CLASS_NAMES[idx],
        "predicted_idx": idx,
        "confidence": float(probs[idx]),
        "probabilities": sorted_probs,
    }


def compute_gradcam(image_bytes: bytes, model_path: str, pred_index=None):
    """
    Grad-CAM on the EfficientNetB0 stem output (block5c_add, 14x14x112).

    Instead of tracing gradients through the full transformer (which is
    extremely slow on CPU), we:
      1. Call stem(input) and tape.watch the output
      2. Manually propagate through the head layers
      3. Compute gradients of the predicted class w.r.t. stem output
    This gives a 14x14 attention map showing which spatial regions the CNN
    stem found most relevant, upsampled to 224x224 for overlay.
    """
    model = load_model(model_path)

    x = preprocess_image(image_bytes)
    x_tensor = tf.constant(x)

    with tf.GradientTape() as tape:
        # Forward through stem
        conv_features = _stem(x_tensor, training=False)
        tape.watch(conv_features)

        # Forward through head layers manually
        h = conv_features
        for layer in _head_layers:
            try:
                h = layer(h, training=False)
            except TypeError:
                h = layer(h)
        preds = h

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]).numpy())
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_features)
    if grads is None:
        return None, None, pred_index

    # Pool gradients over spatial dims -> channel importance weights
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv0 = conv_features[0]
    heatmap = conv0 @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    # Resize and colorize
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_jet_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    orig_arr = np.asarray(orig)
    overlay = (orig_arr.astype(np.float32) * 0.55 + heatmap_jet_rgb.astype(np.float32) * 0.45)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay, heatmap_jet_rgb, pred_index
