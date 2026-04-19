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

# Intentionally empty — import directly from submodules.
# TensorFlow is optional; top-level re-exports here would force it to be
# imported at app startup even when it is not installed.
