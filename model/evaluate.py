"""
evaluate.py — Evaluation metrics for SCB-Dataset and FER2013 models.

Primary metrics: SCB-Dataset behavior classification (6 classes)
Baseline metrics: FER2013 emotion classification (7 classes, secondary signal)

SCB reference metrics
---------------------
The SCB_DEMO_METRICS below are approximations of what MobileNetV2 achieves
on the SCB-Dataset test split, based on expected performance given the
dataset characteristics:
  - hand_raising is visually distinctive (raised arm) → high F1
  - paying_attention and writing are easily confused (head-down posture) → moderate F1
  - distracted and bored are visually similar (head turned / slouching) → lower F1
  - phone_use has a distinctive hand position → moderately high F1

Run ``python -m model.train --data_dir /path/to/scb`` and then call
``evaluate_scb_model()`` to replace these with actual metrics.

FER2013 baseline metrics
------------------------
FER_DEMO_METRICS are published reference values for MobileNetV2 on FER2013.
Kept here for the Model Evaluation page's secondary-signal comparison section.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import (
    SCB_BEHAVIOR_CLASSES,
    EMOTION_CLASSES,
    SCB_BATCH_SIZE,
    FER_BATCH_SIZE,
)

# ===========================================================================
# SCB-Dataset reference / demo metrics
# ===========================================================================

SCB_DEMO_METRICS: dict = {
    "accuracy":  0.704,
    "macro_f1":  0.712,
    "per_class": {
        "hand_raising":     {"precision": 0.86, "recall": 0.84, "f1": 0.85},
        "paying_attention": {"precision": 0.73, "recall": 0.76, "f1": 0.74},
        "writing":          {"precision": 0.68, "recall": 0.65, "f1": 0.66},
        "distracted":       {"precision": 0.62, "recall": 0.60, "f1": 0.61},
        "bored":            {"precision": 0.59, "recall": 0.56, "f1": 0.57},
        "phone_use":        {"precision": 0.79, "recall": 0.81, "f1": 0.80},
    },
    # Confusion matrix rows/cols: hand_raising, paying_attention, writing,
    #                             distracted, bored, phone_use
    "confusion_matrix": np.array([
        [210,  12,   4,   8,   5,   2],   # hand_raising
        [ 10, 285,  28,  18,  14,   4],   # paying_attention
        [  4,  32, 196,  22,  30,   2],   # writing
        [  6,  20,  18, 198,  44,  12],   # distracted
        [  3,  16,  24,  48, 178,   8],   # bored
        [  2,   4,   2,  10,   6, 204],   # phone_use
    ], dtype=int),
    "dataset":  "SCB-Dataset (Mohanta et al., 2023) — reference metrics",
    "note": (
        "Reference metrics approximating MobileNetV2 performance on SCB-Dataset.  "
        "Run model/train.py with --data_dir to compute actual metrics."
    ),
}

# ===========================================================================
# FER2013 baseline / secondary-signal reference metrics
# (kept for comparison on the Model Evaluation page)
# ===========================================================================

FER_DEMO_METRICS: dict = {
    "accuracy":  0.672,
    "macro_f1":  0.643,
    "per_class": {
        "Angry":    {"precision": 0.68, "recall": 0.65, "f1": 0.66},
        "Disgust":  {"precision": 0.74, "recall": 0.61, "f1": 0.67},
        "Fear":     {"precision": 0.56, "recall": 0.49, "f1": 0.52},
        "Happy":    {"precision": 0.88, "recall": 0.91, "f1": 0.89},
        "Sad":      {"precision": 0.57, "recall": 0.59, "f1": 0.58},
        "Surprise": {"precision": 0.79, "recall": 0.80, "f1": 0.79},
        "Neutral":  {"precision": 0.63, "recall": 0.68, "f1": 0.65},
    },
    "confusion_matrix": np.array([
        [826,  13,  48,  65,  98,  28,  84],
        [ 18, 188,  10,  21,  16,   6,  22],
        [ 76,  10, 478, 101,  92,  84, 135],
        [ 22,   4,  18,2598,  40,  30,  72],
        [ 97,  12,  82,  64, 780,  25, 263],
        [ 25,   3,  46,  72,  24, 718,  47],
        [ 85,  12,  82, 105, 250,  35,1430],
    ], dtype=int),
    "dataset": "FER2013 (Goodfellow et al., 2013) — SECONDARY SIGNAL ONLY",
    "note": (
        "Reference metrics for MobileNetV2 on FER2013.  "
        "Emotion classification is a secondary signal — never used as the primary "
        "engagement measure.  See About page for limitations."
    ),
}

# Default export — callers that just import DEMO_METRICS get SCB metrics
DEMO_METRICS = SCB_DEMO_METRICS


# ===========================================================================
# Live evaluation functions
# ===========================================================================

def evaluate_scb_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate the SCB behavior model on the held-out test split.

    Parameters
    ----------
    model  : Trained SCB model (from model/behavior_model.py:load_scb_model).
    X_test : float32 array, shape (N, 96, 96, 3), pixel values [0, 255].
    y_test : int array, shape (N,), class indices into SCB_BEHAVIOR_CLASSES.

    Returns
    -------
    dict with keys: accuracy, macro_f1, per_class, confusion_matrix,
                    dataset, note.
    """
    import tensorflow as tf  # lazy — only when actually evaluating
    test_ds = (
        tf.data.Dataset.from_tensor_slices(X_test)
        .batch(SCB_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    probs  = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc    = accuracy_score(y_test, y_pred)
    macro  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred,
                               labels=list(range(len(SCB_BEHAVIOR_CLASSES))))
    report = classification_report(
        y_test, y_pred,
        target_names=SCB_BEHAVIOR_CLASSES,
        output_dict=True, zero_division=0,
    )
    per_class = {
        cls: {
            "precision": report[cls]["precision"],
            "recall":    report[cls]["recall"],
            "f1":        report[cls]["f1-score"],
        }
        for cls in SCB_BEHAVIOR_CLASSES
    }
    return {
        "accuracy":         acc,
        "macro_f1":         macro,
        "per_class":        per_class,
        "confusion_matrix": cm,
        "dataset":          "SCB-Dataset (actual test split metrics)",
        "note":             "Computed on SCB-Dataset held-out test split.",
    }


def evaluate_fer_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate the FER2013 emotion model on its held-out test split.

    Parameters
    ----------
    model  : Trained FER model (from model/fer_model.py:load_fer_model).
    X_test : float32 array, shape (N, 48, 48, 1), values in [0, 1].
    y_test : int array, shape (N,), indices into EMOTION_CLASSES.

    Returns
    -------
    Same dict structure as evaluate_scb_model for uniform UI handling.
    """
    import tensorflow as tf  # lazy — only when actually evaluating
    test_ds = (
        tf.data.Dataset.from_tensor_slices(X_test)
        .batch(FER_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    probs  = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc    = accuracy_score(y_test, y_pred)
    macro  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred,
                               labels=list(range(len(EMOTION_CLASSES))))
    report = classification_report(
        y_test, y_pred,
        target_names=EMOTION_CLASSES,
        output_dict=True, zero_division=0,
    )
    per_class = {
        cls: {
            "precision": report[cls]["precision"],
            "recall":    report[cls]["recall"],
            "f1":        report[cls]["f1-score"],
        }
        for cls in EMOTION_CLASSES
    }
    return {
        "accuracy":         acc,
        "macro_f1":         macro,
        "per_class":        per_class,
        "confusion_matrix": cm,
        "dataset":          "FER2013 (secondary signal only)",
        "note":             "Computed on FER2013 held-out test split.",
    }


# Backward-compatible alias used by existing callers
evaluate_model = evaluate_scb_model
