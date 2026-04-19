"""
model/ — CNN models for the Classroom Engagement Detection System.

Primary model (SCB-Dataset)
----------------------------
behavior_model.py  — MobileNetV2 trained on SCB behavior classes:
                     hand_raising, paying_attention, writing,
                     distracted, bored, phone_use
train.py           — Two-phase transfer-learning training script (SCB)
evaluate.py        — Metrics for SCB (primary) and FER2013 (baseline)

Baseline / secondary model (FER2013)
--------------------------------------
fer_model.py       — MobileNetV2 trained on FER2013 emotion classes
                     (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
                     Kept for secondary-signal comparison on the
                     Model Evaluation page; NOT used for primary scoring.
"""

from .behavior_model import (
    build_scb_model,
    load_scb_model,
    predict_behavior,
    scb_probs_to_engagement,
    scb_probs_to_proxy_scores,
)
from .fer_model import build_fer_model, load_fer_model
from .evaluate import evaluate_scb_model, evaluate_fer_model, evaluate_model

__all__ = [
    # SCB (primary)
    "build_scb_model",
    "load_scb_model",
    "predict_behavior",
    "scb_probs_to_engagement",
    "scb_probs_to_proxy_scores",
    # FER2013 (secondary)
    "build_fer_model",
    "load_fer_model",
    # Evaluation
    "evaluate_scb_model",
    "evaluate_fer_model",
    "evaluate_model",
]
