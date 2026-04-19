"""
engagement_scorer.py — Weighted engagement scoring rubric with SCB CNN fusion.

Scoring architecture
--------------------
When the SCB behavior model IS loaded:
    per_person_score = SCB_CNN_WEIGHT  × scb_engagement_score
                     + MEDIAPIPE_WEIGHT × proxy_engagement_score

When the SCB model is NOT loaded (MediaPipe-only fallback):
    per_person_score = proxy_engagement_score

The proxy_engagement_score is the weighted dot product of five landmark-based
behavioral signals (same rubric as before):

┌──────────────────┬────────┬────────────────────────────────────────────────┐
│ Behavioral proxy │ Weight │ Rationale                                      │
├──────────────────┼────────┼────────────────────────────────────────────────┤
│ Hand raised      │  0.30  │ Gold-standard active-participation indicator    │
│ Head forward     │  0.25  │ Attentional orientation (Whitehill et al. 2014) │
│ Gaze forward     │  0.20  │ Visual attention proxy (iris offset)           │
│ Good posture     │  0.15  │ Alertness correlate (Mota & Picard 2003)       │
│ Phone absent     │  0.10  │ Off-task device use (Burak 2012)               │
└──────────────────┴────────┴────────────────────────────────────────────────┘

Classroom-level metric (attendance-adjusted)
--------------------------------------------
    classroom_score = mean(per_person_scores) × (detected / expected)

The attendance-adjustment penalises low turnout so that a highly engaged
but very small group never masks poor overall attendance.

References
----------
Mohanta, A. et al. (2023). SCB-Dataset. arXiv:2304.02488.
Whitehill, J. et al. (2014). Faces of engagement. IEEE TAC 5(1).
Mota, S. & Picard, R. (2003). Automated posture analysis. CVPR Workshop.
Burak, L.L. (2012). Multitasking in the university classroom. IJSS.
"""

from __future__ import annotations

import numpy as np

from config import (
    ENGAGEMENT_WEIGHTS,
    ENGAGED_THRESHOLD,
    LOW_ATTENDANCE_THRESHOLD,
    SCB_CNN_WEIGHT,
    MEDIAPIPE_WEIGHT,
)
from pipeline.behavioral_detection import PersonResult


def score_person(result: PersonResult) -> float:
    """
    Compute the final engagement score in [0, 1] for one person.

    Fuses SCB CNN engagement (when available) with the MediaPipe proxy score
    using the weights from config.  When the SCB model has not been trained
    (scb_engagement == -1), the score is the pure proxy dot product.

    Parameters
    ----------
    result : PersonResult from BehavioralDetector.analyse()

    Returns
    -------
    float in [0, 1]
    """
    # ---- MediaPipe behavioral proxy score (always available) ----
    proxy_score = float(np.clip(
        ENGAGEMENT_WEIGHTS["hand_raised"]  * result.hand_raised_score   +
        ENGAGEMENT_WEIGHTS["head_forward"] * result.head_forward_score  +
        ENGAGEMENT_WEIGHTS["gaze_forward"] * result.gaze_forward_score  +
        ENGAGEMENT_WEIGHTS["good_posture"] * result.good_posture_score  +
        ENGAGEMENT_WEIGHTS["phone_absent"] * result.phone_absent_score,
        0.0, 1.0,
    ))

    # ---- SCB CNN engagement score (fuse when available) ----
    # result.scb_engagement == -1 signals "no CNN prediction"
    if result.scb_engagement >= 0:
        # Weighted blend: CNN carries SCB_CNN_WEIGHT (default 0.55)
        fused = SCB_CNN_WEIGHT * result.scb_engagement + MEDIAPIPE_WEIGHT * proxy_score
        return float(np.clip(fused, 0.0, 1.0))

    return proxy_score


def classroom_engagement(
    person_results: list[PersonResult],
    expected_class_size: int,
) -> dict:
    """
    Aggregate individual engagement scores into classroom-level metrics.

    Parameters
    ----------
    person_results      : List of PersonResult from BehavioralDetector.analyse().
    expected_class_size : Instructor-supplied expected number of students.

    Returns
    -------
    dict with keys:
        per_person_scores      (list[float])
        avg_behavioral_score   (float)
        engaged_count          (int)
        detected_count         (int)
        expected_size          (int)
        attendance_rate        (float ∈ [0,1])
        engagement_rate        (float)
        classroom_score        (float ∈ [0,1])
        low_attendance_warning (bool)
        proxy_breakdown        (dict[str, float])
        scb_class_distribution (dict[str, int])  ← new: SCB class histogram
        scb_model_active       (bool)             ← new: whether CNN contributed
    """
    n = len(person_results)
    if n == 0:
        return _empty_result(expected_class_size)

    per_person     = [score_person(r) for r in person_results]
    avg_behavioral = float(np.mean(per_person))
    engaged_count  = sum(1 for s in per_person if s >= ENGAGED_THRESHOLD)

    # ---- Class-average proxy breakdown ----
    proxy_breakdown = {
        "hand_raised":  float(np.mean([r.hand_raised_score   for r in person_results])),
        "head_forward": float(np.mean([r.head_forward_score  for r in person_results])),
        "gaze_forward": float(np.mean([r.gaze_forward_score  for r in person_results])),
        "good_posture": float(np.mean([r.good_posture_score  for r in person_results])),
        "phone_absent": float(np.mean([r.phone_absent_score  for r in person_results])),
    }

    # ---- SCB class distribution histogram ----
    scb_class_dist: dict[str, int] = {}
    scb_active = any(r.scb_engagement >= 0 for r in person_results)
    for r in person_results:
        if r.scb_class:
            scb_class_dist[r.scb_class] = scb_class_dist.get(r.scb_class, 0) + 1

    # ---- Attendance-adjusted classroom score ----
    expected        = max(expected_class_size, 1)
    attendance_rate = min(n / expected, 1.0)
    engagement_rate = engaged_count / expected
    classroom_score = float(np.clip(avg_behavioral * attendance_rate, 0.0, 1.0))

    return {
        "per_person_scores":       per_person,
        "avg_behavioral_score":    avg_behavioral,
        "engaged_count":           engaged_count,
        "detected_count":          n,
        "expected_size":           expected_class_size,
        "attendance_rate":         attendance_rate,
        "engagement_rate":         engagement_rate,
        "classroom_score":         classroom_score,
        "low_attendance_warning":  attendance_rate < LOW_ATTENDANCE_THRESHOLD,
        "proxy_breakdown":         proxy_breakdown,
        "scb_class_distribution":  scb_class_dist,
        "scb_model_active":        scb_active,
    }


def _empty_result(expected_class_size: int) -> dict:
    """Return a fully-zero result dict when no people are detected."""
    return {
        "per_person_scores":       [],
        "avg_behavioral_score":    0.0,
        "engaged_count":           0,
        "detected_count":          0,
        "expected_size":           expected_class_size,
        "attendance_rate":         0.0,
        "engagement_rate":         0.0,
        "classroom_score":         0.0,
        "low_attendance_warning":  True,
        "proxy_breakdown":         {k: 0.0 for k in ENGAGEMENT_WEIGHTS},
        "scb_class_distribution":  {},
        "scb_model_active":        False,
    }
