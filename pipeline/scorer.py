"""
pipeline/scorer.py
==================
Weighted engagement scoring and attendance-aware class-pulse computation.

Signal weights (sum of positive weights = 90 %)
------------------------------------------------
head_pose    30 %  — attentional direction; strongest individual predictor
posture      25 %  — upright body correlates with active attention
hand_raise   25 %  — direct active-participation signal (bonus above baseline)
phone_use   −20 %  — off-task device use (penalty, applied last)
talking      10 %  — context-dependent; 0.5 = not observed (neutral)

Baseline score (attentive student, no hand raised, not talking, no phone):
  = 0.30×1.0 + 0.25×1.0 + 0.25×0.0 + 0.10×0.5 = 0.60  → "engaged"

Class pulse
-----------
  class_score = Σ(individual scores) / expected_class_size

Absent students count as 0.0, so low attendance penalises the class score
naturally without a separate penalty factor.

References
----------
Raca et al. (2015) — head orientation predicts on-task behaviour.
Dewan et al. (2019) — multimodal engagement detection review.
"""

from __future__ import annotations

from pipeline.detector import PersonDetection

# ---------------------------------------------------------------------------
# Weights and thresholds
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "head_pose":  0.30,
    "posture":    0.25,
    "hand_raise": 0.25,
    "talking":    0.10,
}
PHONE_PENALTY = 0.20

# Score cutoffs for engagement labels
ENGAGED_THRESH    = 0.60
NEUTRAL_THRESH    = 0.35

# Attendance warning trigger (< 50 % of expected → flag)
LOW_ATT_RATIO = 0.50

# Class-pulse thresholds
PULSE_HIGH     = 0.70
PULSE_MODERATE = 0.40

PULSE_COLOUR: dict[str, str] = {
    "High":     "#2ecc71",
    "Moderate": "#f39c12",
    "Low":      "#e74c3c",
}


# ---------------------------------------------------------------------------
# Per-person scoring
# ---------------------------------------------------------------------------

def score_person(p: PersonDetection) -> float:
    """
    Compute a single engagement score ∈ [0, 1] for one detected student.

    HOG-only detections (no face) are capped at 0.45 — we have less
    information and should not over-estimate their engagement.
    """
    raw = (
        WEIGHTS["head_pose"]  * p.head_pose_score  +
        WEIGHTS["posture"]    * p.posture_score     +
        WEIGHTS["hand_raise"] * p.hand_raise_score  +
        WEIGHTS["talking"]    * p.talking_score
    )
    if p.phone_detected:
        raw -= PHONE_PENALTY

    score = max(0.0, min(1.0, raw))

    # Partial credit only for detections without a visible face
    if not p.has_face:
        score = min(score, 0.45)

    return score


def engagement_label(score: float) -> str:
    """Map a score to a three-level engagement label."""
    if score >= ENGAGED_THRESH:
        return "engaged"
    elif score >= NEUTRAL_THRESH:
        return "neutral"
    return "disengaged"


# ---------------------------------------------------------------------------
# Classroom-level aggregation
# ---------------------------------------------------------------------------

def compute_scores(persons: list[PersonDetection], expected_size: int) -> dict:
    """
    Aggregate engagement statistics and compute the attendance-adjusted
    class-pulse score.

    Parameters
    ----------
    persons       : Output of ``detect_persons()``.
    expected_size : Instructor-supplied expected class enrolment (≥ 1).

    Returns
    -------
    dict with keys:
        detected          int    — persons found
        expected          int    — as supplied
        per_scores        list   — individual score per person [0, 1]
        engaged_count     int
        neutral_count     int
        disengaged_count  int
        class_score       float  — Σ scores / expected  ∈ [0, 1]
        class_score_pct   float  — × 100
        pulse_label       str    — 'High' | 'Moderate' | 'Low'
        attendance_rate   float  — detected / expected (capped at 1.0)
        low_attendance    bool
        proxy_avgs        dict   — per-signal class averages for the chart
        persons           list   — original PersonDetection list
    """
    expected_size = max(1, int(expected_size))
    detected      = len(persons)

    per_scores = [score_person(p) for p in persons]

    counts: dict[str, int] = {"engaged": 0, "neutral": 0, "disengaged": 0}
    for s in per_scores:
        counts[engagement_label(s)] += 1

    # Absent students implicitly score 0 — sum / expected penalises absences
    class_score = min(1.0, sum(per_scores) / expected_size)

    if class_score >= PULSE_HIGH:
        pulse_label = "High"
    elif class_score >= PULSE_MODERATE:
        pulse_label = "Moderate"
    else:
        pulse_label = "Low"

    n = max(detected, 1)
    proxy_avgs: dict[str, float] = {
        "Head pose":  sum(p.head_pose_score  for p in persons) / n,
        "Posture":    sum(p.posture_score     for p in persons) / n,
        "Hand raise": sum(p.hand_raise_score  for p in persons) / n,
        "Talking":    sum(p.talking_score     for p in persons) / n,
        "Phone use":  sum(p.phone_score       for p in persons) / n,
    }

    return {
        "detected":         detected,
        "expected":         expected_size,
        "per_scores":       per_scores,
        "engaged_count":    counts["engaged"],
        "neutral_count":    counts["neutral"],
        "disengaged_count": counts["disengaged"],
        "class_score":      class_score,
        "class_score_pct":  round(class_score * 100, 1),
        "pulse_label":      pulse_label,
        "attendance_rate":  min(1.0, detected / expected_size),
        "low_attendance":   detected < LOW_ATT_RATIO * expected_size,
        "proxy_avgs":       proxy_avgs,
        "persons":          persons,
    }
