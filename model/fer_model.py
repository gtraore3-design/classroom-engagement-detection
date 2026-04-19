"""
fer_model.py — MobileNetV2-based CNN for FER2013 emotion classification.

Architecture rationale
----------------------
We use **MobileNetV2** as the backbone for three reasons:
  1. It is lightweight enough to run on a standard laptop CPU during inference
     (< 50 ms per face crop on modern hardware).
  2. Its ImageNet pre-trained weights provide a strong feature extractor that
     transfers well to the small FER2013 images (Sandler et al., 2018).
  3. It achieves near state-of-the-art accuracy on FER2013 while being
     significantly smaller than VGG-16 or ResNet-50.

Transfer learning strategy (two-phase)
---------------------------------------
Phase 1 — feature extraction only:
    The MobileNetV2 backbone is frozen.  Only the classification head
    (Dense 256 → Dense 7) is trained.  This prevents overwriting the
    rich ImageNet feature representations before the head has converged.

Phase 2 — fine-tuning:
    The top 30 layers of MobileNetV2 are unfrozen and trained jointly with
    the head at a lower learning rate (1e-4 vs. 1e-3 in phase 1).  This
    adapts higher-level features to the FER2013 domain (close-up faces,
    48 × 48 grayscale).

Input processing
----------------
FER2013 images are 48 × 48 single-channel (grayscale).  MobileNetV2 expects
3-channel RGB input at minimum 32 × 32.  We therefore:
  1. Replicate the grayscale channel → 3 identical channels.
  2. Upsample to 96 × 96 (larger than the 32 × 32 minimum improves accuracy).
  3. Apply MobileNetV2's built-in preprocessing (scale to [-1, 1]).

⚠️  Secondary signal only
    This model's predictions are NEVER used as the sole determinant of
    engagement.  Emotion classification from static face images is an
    unreliable and ethically sensitive proxy for engagement (see About page).

References
----------
Sandler, M. et al. (2018). MobileNetV2: Inverted residuals and linear
    bottlenecks. *CVPR 2018*, 4510–4520.
Goodfellow, I. et al. (2013). Challenges in representation learning: A report
    on three machine learning contests. *ICML Workshop on Challenges in
    Representation Learning*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import EMOTION_CLASSES, FER_IMG_SIZE as IMG_SIZE

# Number of output classes (7 FER2013 emotions)
NUM_CLASSES = len(EMOTION_CLASSES)

# Default path where the trained model is saved / loaded
MODEL_PATH = Path(__file__).parent / "fer_mobilenetv2.keras"


def build_fer_model(dropout: float = 0.4) -> keras.Model:
    """
    Construct the MobileNetV2 transfer-learning model for FER2013.

    Parameters
    ----------
    dropout : float
        Dropout rate applied before the final Dense layer.
        0.4 is the default — higher values help regularise on the small
        FER2013 dataset (28 709 training samples).

    Returns
    -------
    keras.Model
        Compiled model with input shape (batch, 48, 48, 1) and output
        shape (batch, 7) — softmax probabilities over emotion classes.

    Notes
    -----
    The model is *not* compiled here; compilation is done in train.py
    so that the learning rate and loss can be adjusted between phases.
    """
    # ---- Input: 48 × 48 grayscale face crop ----
    inp = keras.Input(shape=(*IMG_SIZE, 1), name="face_crop")

    # ---- Channel replication: 1 → 3 (MobileNetV2 requirement) ----
    # tf.repeat along the channel axis avoids any learnable parameters here.
    x = layers.Lambda(
        lambda t: tf.repeat(t, 3, axis=-1),
        name="gray_to_rgb",
    )(inp)

    # ---- Spatial upsampling: 48 × 48 → 96 × 96 ----
    # Larger input improves gradient flow through the MobileNetV2 inverted
    # residuals.  96 is a reasonable trade-off between accuracy and speed.
    x = layers.Resizing(96, 96, name="resize")(x)

    # ---- MobileNetV2 preprocessing: scale pixel values to [-1, 1] ----
    x = layers.Lambda(
        lambda t: tf.keras.applications.mobilenet_v2.preprocess_input(t),
        name="preprocess",
    )(x)

    # ---- MobileNetV2 backbone (ImageNet pre-trained, phase-1 frozen) ----
    base = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,      # drop ImageNet classification head
        weights="imagenet",     # start from ImageNet features
    )
    base.trainable = False      # freeze during phase 1

    x = base(x, training=False)   # training=False keeps BN layers in inference mode

    # ---- Classification head ----
    x   = layers.GlobalAveragePooling2D(name="gap")(x)   # (batch, 1280)
    x   = layers.BatchNormalization()(x)                  # stabilises training
    x   = layers.Dense(256, activation="relu")(x)         # intermediate representation
    x   = layers.Dropout(dropout)(x)                      # regularisation
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="emotions")(x)

    model = keras.Model(inp, out, name="FER_MobileNetV2")
    return model


def unfreeze_top_layers(model: keras.Model, n_layers: int = 30) -> None:
    """
    Unfreeze the top *n_layers* of the MobileNetV2 backbone for phase-2
    fine-tuning.

    Lower layers (early conv filters) remain frozen because they capture
    generic low-level features (edges, textures) that transfer well from
    ImageNet.  Higher layers capture more domain-specific features that
    benefit from adaptation to FER2013 face images.

    Parameters
    ----------
    model    : The model returned by build_fer_model().
    n_layers : Number of backbone layers to unfreeze from the top (default 30).
    """
    base = model.get_layer("mobilenetv2_1.00_96")
    base.trainable = True
    # Freeze all but the last n_layers layers
    for layer in base.layers[:-n_layers]:
        layer.trainable = False


def load_fer_model(path: Path | str = MODEL_PATH) -> keras.Model | None:
    """
    Load a previously trained model from disk.

    Parameters
    ----------
    path : Path or str to the saved .keras file.

    Returns
    -------
    keras.Model if the file exists; None otherwise.
    """
    p = Path(path)
    if not p.exists():
        return None
    return keras.models.load_model(str(p))


def predict_emotion(
    model: keras.Model,
    face_gray: np.ndarray,
) -> dict[str, float]:
    """
    Predict emotion class probabilities for a single face crop.

    Parameters
    ----------
    model     : Trained Keras model returned by load_fer_model().
    face_gray : np.ndarray, shape (48, 48) or (48, 48, 1), dtype uint8.
                Grayscale face crop, pixel values in [0, 255].

    Returns
    -------
    dict mapping each emotion label to its predicted probability (sum ≈ 1.0).

    Example
    -------
    >>> probs = predict_emotion(model, face_crop)
    >>> print(max(probs, key=probs.get))   # most likely emotion
    'Happy'
    """
    # Normalise to [0, 1] float32
    img = face_gray.astype(np.float32) / 255.0

    # Ensure shape is (48, 48, 1)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    # Add batch dimension → (1, 48, 48, 1)
    img = img[np.newaxis]

    # Run forward pass
    probs = model.predict(img, verbose=0)[0]   # shape: (7,)

    return {cls: float(p) for cls, p in zip(EMOTION_CLASSES, probs)}
