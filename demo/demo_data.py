"""
demo_data.py — Synthetic analysis results for grader / demo mode.

The 20 synthetic students now carry SCB behavior class labels in addition
to the MediaPipe proxy scores, reflecting the new primary dataset.

Each synthetic student has:
  - MediaPipe-style proxy scores (hand_raised, head_forward, etc.)
  - An SCB class label (the behavior the CNN would have predicted)
  - A pre-computed scb_engagement score derived from SCB_ENGAGEMENT_SCORES

This gives the Dashboard and per-person table a realistic preview of the
fully-integrated pipeline (MediaPipe + SCB CNN) without requiring a trained
model or a real classroom image.
"""

from __future__ import annotations

import numpy as np

from config import (
    ENGAGEMENT_WEIGHTS,
    ENGAGED_THRESHOLD,
    LOW_ATTENDANCE_THRESHOLD,
    SCB_BEHAVIOR_CLASSES,
    SCB_ENGAGEMENT_SCORES,
    SCB_CNN_WEIGHT,
    MEDIAPIPE_WEIGHT,
)
from pipeline.behavioral_detection import PersonResult


# ---------------------------------------------------------------------------
# Helper: replicate score_person() locally to avoid circular imports
# ---------------------------------------------------------------------------

def _score(r: PersonResult) -> float:
    """Engagement score using SCB fusion when scb_engagement is available."""
    proxy = float(np.clip(
        ENGAGEMENT_WEIGHTS["hand_raised"]  * r.hand_raised_score   +
        ENGAGEMENT_WEIGHTS["head_forward"] * r.head_forward_score  +
        ENGAGEMENT_WEIGHTS["gaze_forward"] * r.gaze_forward_score  +
        ENGAGEMENT_WEIGHTS["good_posture"] * r.good_posture_score  +
        ENGAGEMENT_WEIGHTS["phone_absent"] * r.phone_absent_score,
        0.0, 1.0,
    ))
    if r.scb_engagement >= 0:
        return float(np.clip(SCB_CNN_WEIGHT*r.scb_engagement + MEDIAPIPE_WEIGHT*proxy, 0.0, 1.0))
    return proxy


# ---------------------------------------------------------------------------
# Seat positions (4 rows × 5 cols, 800 × 600 image)
# ---------------------------------------------------------------------------

def _seat_centres() -> list[tuple[float, float]]:
    margin, board_y2  = 60, 60 + 52
    usable_w = 800 - 2*margin - 40
    usable_h = 600 - 2*margin - 140
    col_step = usable_w // 6
    row_step = usable_h // 5
    return [
        (float(margin + 20 + col_step*(col+1)),
         float(board_y2 + 60 + row_step*(row+1)))
        for row in range(4) for col in range(5)
    ]


# ---------------------------------------------------------------------------
# Synthetic student data
# Format: (hand_raised, head_forward, gaze_forward, good_posture, phone_absent,
#           head_pitch, head_yaw, scb_class)
# ---------------------------------------------------------------------------

_STUDENT_DATA = [
    # ---- Row 0: FRONT — high engagement ----
    (1.00, 0.95, 0.90, 0.92, 1.00,  -3.0,   2.5, "hand_raising"),
    (0.00, 0.88, 0.85, 0.80, 1.00,   5.1,   4.2, "paying_attention"),
    (1.00, 0.92, 0.95, 0.95, 1.00,  -1.5,  -3.0, "hand_raising"),
    (0.00, 0.85, 0.80, 0.75, 1.00,   7.2,   8.0, "paying_attention"),
    (0.50, 0.78, 0.76, 0.72, 1.00,  10.5,   5.5, "hand_raising"),

    # ---- Row 1: SECOND ROW — good engagement ----
    (0.00, 0.75, 0.72, 0.70, 1.00,  12.0,  10.0, "paying_attention"),
    (0.00, 0.80, 0.78, 0.65, 1.00,   8.5,  -6.0, "writing"),
    (0.00, 0.70, 0.65, 0.60, 1.00,  14.0,  12.5, "writing"),
    (0.00, 0.72, 0.70, 0.68, 1.00,  11.0,  -9.0, "paying_attention"),
    (0.00, 0.68, 0.60, 0.62, 1.00,  16.0,   8.0, "writing"),

    # ---- Row 2: MIDDLE — mixed engagement ----
    (0.00, 0.55, 0.52, 0.48, 1.00,  18.5,  15.0, "distracted"),
    (0.00, 0.62, 0.58, 0.55, 1.00,  14.5,  18.0, "writing"),
    (0.00, 0.50, 0.45, 0.42, 0.60,  20.0,  16.5, "phone_use"),
    (0.00, 0.48, 0.40, 0.38, 1.00,  22.0,  20.0, "distracted"),
    (0.00, 0.60, 0.55, 0.50, 1.00,  17.0,  14.0, "distracted"),

    # ---- Row 3: BACK — low engagement ----
    (0.00, 0.35, 0.30, 0.28, 0.10,  28.0,  22.0, "phone_use"),
    (0.00, 0.28, 0.25, 0.22, 0.20,  30.5,  25.0, "bored"),
    (0.00, 0.48, 0.42, 0.40, 1.00,  20.0,  18.5, "distracted"),
    (0.00, 0.25, 0.20, 0.18, 0.00,  32.0,  28.0, "phone_use"),
    (0.00, 0.38, 0.32, 0.30, 0.30,  25.0,  20.0, "bored"),
]


def _build_person_results() -> list[PersonResult]:
    """
    Construct 20 PersonResult objects from the synthetic data table.

    Each result includes:
    - MediaPipe proxy scores (as if from Face Mesh + Pose)
    - SCB class label and probability vector (as if from the CNN)
    - scb_engagement derived from SCB_ENGAGEMENT_SCORES
    """
    centres = _seat_centres()
    results = []

    for pid, (row, centre) in enumerate(zip(_STUDENT_DATA, centres)):
        hr, hf, gf, gp, pa, pitch, yaw, scb_cls = row

        # Build a simple one-hot-ish probability vector centred on the true class
        # (adds small noise to neighbouring classes for realism)
        probs = {c: 0.02 for c in SCB_BEHAVIOR_CLASSES}
        probs[scb_cls] = 0.85
        # Redistribute remaining probability among the rest
        others = [c for c in SCB_BEHAVIOR_CLASSES if c != scb_cls]
        remainder = (1.0 - 0.85 - 0.02*(len(SCB_BEHAVIOR_CLASSES)-1))
        for i, c in enumerate(others):
            probs[c] += remainder / len(others)

        scb_engagement = SCB_ENGAGEMENT_SCORES.get(scb_cls, 0.5)

        results.append(PersonResult(
            person_id=pid,
            hand_raised_score=float(hr),
            head_forward_score=float(hf),
            gaze_forward_score=float(gf),
            good_posture_score=float(gp),
            phone_absent_score=float(pa),
            signals_available={"pose": True, "face_mesh": True, "scb_cnn": True},
            center=centre,
            head_pitch_deg=float(pitch),
            head_yaw_deg=float(yaw),
            scb_class=scb_cls,
            scb_probs=probs,
            scb_engagement=float(scb_engagement),
        ))

    return results


def _build_engagement_info(
    person_results: list[PersonResult],
    expected_size: int = 25,
) -> dict:
    """Build the engagement_info dict matching the live pipeline schema."""
    per_person     = [_score(r) for r in person_results]
    avg_behavioral = float(np.mean(per_person))
    engaged_count  = sum(1 for s in per_person if s >= ENGAGED_THRESHOLD)
    n              = len(person_results)
    attendance_rate = min(n / max(expected_size, 1), 1.0)
    classroom_score = float(np.clip(avg_behavioral * attendance_rate, 0.0, 1.0))

    proxy_breakdown = {
        "hand_raised":  float(np.mean([r.hand_raised_score  for r in person_results])),
        "head_forward": float(np.mean([r.head_forward_score for r in person_results])),
        "gaze_forward": float(np.mean([r.gaze_forward_score for r in person_results])),
        "good_posture": float(np.mean([r.good_posture_score for r in person_results])),
        "phone_absent": float(np.mean([r.phone_absent_score for r in person_results])),
    }

    # SCB class histogram
    scb_dist: dict[str, int] = {}
    for r in person_results:
        if r.scb_class:
            scb_dist[r.scb_class] = scb_dist.get(r.scb_class, 0) + 1

    return {
        "per_person_scores":       per_person,
        "avg_behavioral_score":    avg_behavioral,
        "engaged_count":           engaged_count,
        "detected_count":          n,
        "expected_size":           expected_size,
        "attendance_rate":         attendance_rate,
        "engagement_rate":         engaged_count / max(expected_size, 1),
        "classroom_score":         classroom_score,
        "low_attendance_warning":  attendance_rate < LOW_ATTENDANCE_THRESHOLD,
        "proxy_breakdown":         proxy_breakdown,
        "scb_class_distribution":  scb_dist,
        "scb_model_active":        True,
    }


def _build_attendance_info(detected: int = 20) -> dict:
    return {
        "face_count":    detected,
        "body_count":    detected,
        "best_estimate": detected,
        "face_boxes":    [],
        "body_boxes":    [],
    }


def get_demo_results(expected_size: int = 25) -> dict:
    """
    Return a complete ``results`` dict populated with synthetic data.

    Includes SCB class labels and probability distributions so that all
    Dashboard components (including the SCB class histogram) are exercised.
    """
    import cv2
    from demo.sample_image import get_demo_image_bgr
    from utils.visualization import draw_engagement_annotations, engagement_heatmap_overlay

    person_results  = _build_person_results()
    engagement_info = _build_engagement_info(person_results, expected_size)
    attendance_info = _build_attendance_info(len(person_results))

    base_bgr    = get_demo_image_bgr()
    annotated   = draw_engagement_annotations(base_bgr, person_results, show_score=True)
    heatmap_bgr = engagement_heatmap_overlay(base_bgr, person_results, radius=40)

    return {
        "engagement_info": engagement_info,
        "person_results":  person_results,
        "attendance_info": attendance_info,
        "expected_size":   expected_size,
        "annotated_bgr":   annotated,
        "heatmap_bgr":     heatmap_bgr,
        "blurred_bgr":     base_bgr,
        "elapsed_s":       0.0,
        "is_demo":         True,
    }
