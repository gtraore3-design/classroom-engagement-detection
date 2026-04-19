"""
behavior_model.py — MobileNetV2-based CNN for SCB-Dataset behavior classification.

Primary dataset
---------------
SCB-Dataset (Student Classroom Behavior), Mohanta et al. (2023).
arXiv: https://arxiv.org/abs/2304.02488
Kaggle: kaggle.com/datasets/asthalochanmohanta/class-room-student-behaviour

Six behavior classes (mapped directly to engagement signals):
    hand_raising      → highest engagement (active participation)
    paying_attention  → high engagement (attentional orientation)
    writing           → moderate engagement (on-task, head bowed)
    distracted        → low engagement (off-task)
    bored             → very low engagement (passive disengagement)
    phone_use         → minimal engagement (off-task device use)

Architecture rationale
----------------------
We reuse the same MobileNetV2 backbone as the FER2013 baseline model, but
with three key differences:

  1. **Input size**: 96 × 96 RGB (vs. 48 × 48 grayscale for FER2013).
     SCB images are real classroom crops with much more spatial context
     (body posture, arm position, surrounding desk items).  Larger input
     captures this context better.

  2. **Colour**: RGB input (vs. grayscale).
     Colour provides additional discriminative cues: phone screens are often
     brighter / differently coloured than notebooks; raised arms against a
     background wall have colour contrast that aids detection.

  3. **6 output classes** (vs. 7 for FER2013).
     The SCB behavior classes are semantically richer and map directly to the
     engagement scoring rubric — no post-hoc emotion-to-engagement conversion
     is needed.

Transfer learning strategy
---------------------------
Identical two-phase approach to fer_model.py:
  Phase 1 — freeze backbone, train head (Adam lr=1e-3, ~20 epochs)
  Phase 2 — unfreeze top 30 layers, fine-tune (Adam lr=1e-4, ~20 epochs)

Fusion with MediaPipe
---------------------
When this model is loaded, behavioral_detection.py blends CNN-derived proxy
scores with MediaPipe landmark scores (see config.SCB_CNN_WEIGHT).
This hybrid approach is more robust than either source alone:
  - CNN excels at recognising whole-body behavioral patterns.
  - MediaPipe excels at precise landmark geometry (head angle, iris offset).

References
----------
Mohanta, A. et al. (2023). SCB-Dataset: A Dataset for Detecting Student
    Classroom Behavior. arXiv:2304.02488.
Sandler, M. et al. (2018). MobileNetV2: Inverted residuals and linear
    bottlenecks. CVPR 2018, 4510–4520.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import SCB_BEHAVIOR_CLASSES, SCB_IMG_SIZE, SCB_MODEL_FILENAME, SCB_ENGAGEMENT_SCORES

NUM_CLASSES = len(SCB_BEHAVIOR_CLASSES)    # 6
MODEL_PATH  = Path(__file__).parent / SCB_MODEL_FILENAME


def build_scb_model(dropout: float = 0.4) -> keras.Model:
    """
    Build the SCB-Dataset behavior classification model.

    Parameters
    ----------
    dropout : float
        Dropout rate before the output layer.  0.4 balances regularisation
        with the relatively small SCB training set.

    Returns
    -------
    keras.Model
        Input shape : (batch, 96, 96, 3)  — RGB student crop
        Output shape: (batch, 6)           — softmax over SCB behavior classes

    Notes
    -----
    The model is returned *uncompiled* so that train.py can set the optimiser
    and learning rate per training phase.
    """
    # ---- Input: 96×96 RGB student crop ----
    inp = keras.Input(shape=(*SCB_IMG_SIZE, 3), name="student_crop_rgb")

    # ---- MobileNetV2 preprocessing (scale pixels to [-1, 1]) ----
    # Done inside the model graph so it applies automatically during inference.
    x = layers.Lambda(
        lambda t: tf.keras.applications.mobilenet_v2.preprocess_input(t),
        name="preprocess",
    )(inp)

    # ---- MobileNetV2 backbone (ImageNet weights, phase-1 frozen) ----
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*SCB_IMG_SIZE, 3),
        include_top=False,      # remove ImageNet classification head
        weights="imagenet",     # start from rich ImageNet feature representations
    )
    base.trainable = False      # frozen during phase 1

    x = base(x, training=False)   # training=False → BN layers use running stats

    # ---- Classification head ----
    x   = layers.GlobalAveragePooling2D(name="gap")(x)    # (batch, 1280)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="behaviors")(x)

    return keras.Model(inp, out, name="SCB_MobileNetV2")


def unfreeze_top_layers(model: keras.Model, n_layers: int = 30) -> None:
    """
    Unfreeze the top *n_layers* of the MobileNetV2 backbone for phase-2
    fine-tuning.

    The frozen lower layers retain generic edge/texture features from
    ImageNet; the unfrozen top layers are adapted to SCB classroom images.

    Parameters
    ----------
    model    : The model returned by build_scb_model().
    n_layers : Number of backbone layers to unfreeze (default 30).
    """
    base = model.get_layer("mobilenetv2_1.00_96")
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False


def load_scb_model(path: Path | str = MODEL_PATH) -> keras.Model | None:
    """
    Load a previously saved SCB behavior model from disk.

    Returns None (without raising) if no checkpoint exists, so callers can
    gracefully fall back to MediaPipe-only scoring.

    Parameters
    ----------
    path : path to the .keras checkpoint file.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return keras.models.load_model(str(p))
    except Exception:
        return None


def predict_behavior(
    model: keras.Model,
    rgb_crop: np.ndarray,
) -> dict[str, float]:
    """
    Predict behavior class probabilities for a single student crop.

    Parameters
    ----------
    model    : Trained Keras model from load_scb_model().
    rgb_crop : np.ndarray, shape (H, W, 3), dtype uint8, pixel values [0, 255].
               The crop should be centred on one student; aspect ratio need not
               match 96×96 — the model's internal Resizing layer handles that.

    Returns
    -------
    dict mapping each SCB class label → its predicted probability (sum ≈ 1.0).

    Example
    -------
    >>> probs = predict_behavior(model, student_crop)
    >>> print(max(probs, key=probs.get))
    'paying_attention'
    """
    # Resize to model input size (96×96) and normalise to float32 [0, 255]
    # (the model's internal preprocess Lambda scales to [-1, 1])
    resized = tf.image.resize(rgb_crop, SCB_IMG_SIZE).numpy().astype(np.float32)
    batch   = resized[np.newaxis]   # add batch dim: (1, 96, 96, 3)

    probs = model.predict(batch, verbose=0)[0]   # shape: (6,)
    return {cls: float(p) for cls, p in zip(SCB_BEHAVIOR_CLASSES, probs)}


def scb_probs_to_engagement(probs: dict[str, float]) -> float:
    """
    Convert SCB class probability distribution → scalar engagement score [0,1].

    The engagement score is the expected value of the per-class engagement
    scores (defined in config.SCB_ENGAGEMENT_SCORES), weighted by the
    predicted class probabilities:

        E[engagement] = Σ P(class_i) × engagement_score(class_i)

    Parameters
    ----------
    probs : dict returned by predict_behavior()

    Returns
    -------
    float in [0, 1] — 0 = fully disengaged, 1 = fully engaged
    """
    return float(sum(
        prob * SCB_ENGAGEMENT_SCORES.get(cls, 0.5)
        for cls, prob in probs.items()
    ))


def scb_probs_to_proxy_scores(probs: dict[str, float]) -> dict[str, float]:
    """
    Convert SCB class probabilities → per-proxy signal scores.

    Each proxy score is the probability-weighted sum of that proxy's expected
    value across all SCB classes (from config.SCB_TO_PROXY).

    This allows the SCB CNN output to update individual proxy scores (e.g.,
    if the CNN is confident the student is phone_use, the phone_absent_score
    is driven toward 0.0), while preserving MediaPipe's contribution for
    proxies the CNN is uncertain about.

    Parameters
    ----------
    probs : dict returned by predict_behavior()

    Returns
    -------
    dict mapping proxy name → CNN-derived score [0, 1]
    """
    from config import SCB_TO_PROXY, ENGAGEMENT_WEIGHTS

    # Initialise with neutral scores
    proxy_names = list(ENGAGEMENT_WEIGHTS.keys())
    # Convert proxy weight keys (e.g. "hand_raised") to PersonResult field names
    field_names = {k: f"{k}_score" for k in proxy_names}

    result: dict[str, float] = {f: 0.0 for f in field_names.values()}

    for cls, p in probs.items():
        cls_proxy = SCB_TO_PROXY.get(cls, {})
        for field, expected in cls_proxy.items():
            result[field] = result.get(field, 0.0) + p * expected

    # Clip all to [0, 1]
    return {k: float(min(max(v, 0.0), 1.0)) for k, v in result.items()}
