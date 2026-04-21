"""
pipeline/scorer.py — Engagement scoring, image annotation, and chart generation
              for the Gradio classroom analysis POC.

No MediaPipe or TensorFlow required — consumes only the output of detector.py.

Public API
----------
compute_scores(detections, expected_size)  →  dict
    Aggregate statistics: counts, rates, classroom score, low-attendance flag.

annotate_image(bgr, detections)  →  np.ndarray
    Returns a copy of *bgr* with:
      • All detected face regions Gaussian-blurred (privacy).
      • Coloured bounding boxes per person:
          green  = engaged      (head forward, two symmetric eyes)
          yellow = neutral      (tilted head or profile face)
          red    = disengaged   (head bowed, no eyes)
          grey   = no_face      (HOG-only body detection)
      • Small text label in the top-left corner of each box.

make_bar_chart(scores)  →  plt.Figure
    Horizontal bar chart of Engaged / Neutral / Disengaged / No-face counts.

Design notes
------------
Face blurring uses a kernel proportional to the face size so that large
close-up faces are blurred as thoroughly as small distant ones.

Box colours follow the traffic-light convention widely used in educational
dashboards (green → good, amber → caution, red → concern, grey → unknown).
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pipeline.detector import StudentDetection

# ---------------------------------------------------------------------------
# Colour palette (BGR for OpenCV)
# ---------------------------------------------------------------------------

_COLOUR = {
    "engaged":    (57,  197,  73),   # green
    "neutral":    (29,  195, 255),   # amber/gold (BGR: low B, high G, high R)
    "disengaged": (60,   60, 233),   # red
    "no_face":    (130, 130, 130),   # grey
}

# Text drawn on boxes (kept short so it fits even on small face crops)
_LABEL = {
    "engaged":    "engaged",
    "neutral":    "neutral",
    "disengaged": "disengaged",
    "no_face":    "no face",
}

# Engagement scores (used to compute the numeric classroom score)
_ENGAGEMENT_VALUE = {
    "engaged":    1.0,
    "neutral":    0.5,
    "disengaged": 0.0,
    "no_face":    0.3,   # present but unknown pose → small partial credit
}

# Threshold below which we flag a low-attendance warning
_LOW_ATTENDANCE_RATIO = 0.50


# ---------------------------------------------------------------------------
# Public: compute_scores
# ---------------------------------------------------------------------------

def compute_scores(
    detections: list[StudentDetection],
    expected_size: int,
) -> dict:
    """
    Aggregate engagement and attendance statistics from a list of detections.

    Parameters
    ----------
    detections    : Output of ``detect_students()``.
    expected_size : Instructor-supplied expected class size (≥ 1).

    Returns
    -------
    dict with keys:
        detected_count        int   — total persons found
        engaged_count         int
        neutral_count         int
        disengaged_count      int
        no_face_count         int
        expected_size         int   — as supplied
        attendance_rate       float — detected / expected  (capped at 1.0)
        avg_engagement        float — mean _ENGAGEMENT_VALUE across all detections
        classroom_score       float — avg_engagement × attendance_rate  (∈ [0, 1])
        classroom_score_pct   float — classroom_score × 100
        low_attendance_warning bool — True when detected < 50 % of expected
        label                 str   — 'High' | 'Moderate' | 'Low'
    """
    expected_size = max(1, int(expected_size))

    counts = {"engaged": 0, "neutral": 0, "disengaged": 0, "no_face": 0}
    total_engagement = 0.0

    for d in detections:
        eng = d.engagement if d.engagement in counts else "no_face"
        counts[eng] += 1
        total_engagement += _ENGAGEMENT_VALUE[eng]

    detected = len(detections)
    avg_eng  = total_engagement / max(detected, 1)
    att_rate = min(1.0, detected / expected_size)
    cls_score = avg_eng * att_rate

    if cls_score >= 0.70:
        label = "High"
    elif cls_score >= 0.40:
        label = "Moderate"
    else:
        label = "Low"

    return {
        "detected_count":         detected,
        "engaged_count":          counts["engaged"],
        "neutral_count":          counts["neutral"],
        "disengaged_count":       counts["disengaged"],
        "no_face_count":          counts["no_face"],
        "expected_size":          expected_size,
        "attendance_rate":        att_rate,
        "avg_engagement":         avg_eng,
        "classroom_score":        cls_score,
        "classroom_score_pct":    round(cls_score * 100, 1),
        "low_attendance_warning": detected < _LOW_ATTENDANCE_RATIO * expected_size,
        "label":                  label,
    }


# ---------------------------------------------------------------------------
# Public: annotate_image
# ---------------------------------------------------------------------------

def annotate_image(
    bgr: np.ndarray,
    detections: list[StudentDetection],
) -> np.ndarray:
    """
    Return a privacy-safe annotated copy of *bgr*.

    Processing order (important for privacy):
    1. Blur every detected face region **before** drawing any overlay.
    2. Draw coloured bounding boxes and engagement labels.

    Parameters
    ----------
    bgr        : Original classroom frame in BGR colour order.
    detections : Output of ``detect_students()``.

    Returns
    -------
    Annotated BGR image (same resolution as input).
    """
    out = bgr.copy()
    img_h, img_w = out.shape[:2]

    # --- Step 1: blur all face regions (privacy first) ---
    for d in detections:
        if d.face_w > 0 and d.face_h > 0:
            _blur_region(out, d.face_x, d.face_y, d.face_w, d.face_h, img_w, img_h)

    # --- Step 2: draw bounding boxes and labels ---
    for d in detections:
        eng    = d.engagement if d.engagement in _COLOUR else "no_face"
        colour = _COLOUR[eng]
        label  = _LABEL[eng]

        # Box thickness scales slightly with face size for readability
        thickness = max(1, min(3, d.h // 40))

        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), colour, thickness)

        # Label background + text
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.30, min(0.55, d.w / 130))
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

        lx = max(0, d.x)
        ly = max(th + baseline + 2, d.y - 2)

        # Semi-transparent label pill
        cv2.rectangle(
            out,
            (lx, ly - th - baseline - 2),
            (lx + tw + 4, ly + 2),
            colour, -1,
        )
        # Dark text for contrast
        cv2.putText(
            out, label,
            (lx + 2, ly - baseline),
            font, font_scale, (20, 20, 20), 1, cv2.LINE_AA,
        )

        # Eye-count badge (only for face detections)
        if not d.is_hog_only:
            badge = f"eyes:{d.eye_count}"
            bscale = font_scale * 0.75
            (bw, bh), _ = cv2.getTextSize(badge, font, bscale, 1)
            bx = min(img_w - bw - 4, d.x + d.w - bw - 2)
            by = d.y + bh + 4
            cv2.putText(out, badge, (bx, by), font, bscale,
                        colour, 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Public: make_bar_chart
# ---------------------------------------------------------------------------

def make_bar_chart(scores: dict) -> plt.Figure:
    """
    Horizontal bar chart of engagement category counts.

    Parameters
    ----------
    scores : dict returned by ``compute_scores()``.

    Returns
    -------
    matplotlib Figure (caller is responsible for closing it).
    """
    categories = ["Engaged", "Neutral", "Disengaged", "No face"]
    counts = [
        scores["engaged_count"],
        scores["neutral_count"],
        scores["disengaged_count"],
        scores["no_face_count"],
    ]
    colours = [
        "#39c549",   # green
        "#ffc31d",   # amber
        "#e93c3c",   # red
        "#828282",   # grey
    ]

    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars = ax.barh(categories[::-1], counts[::-1],
                   color=colours[::-1], height=0.55, edgecolor="none")

    # Count labels
    for bar, count in zip(bars, counts[::-1]):
        if count > 0:
            ax.text(
                bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", ha="left",
                color="#c9d1d9", fontsize=10,
            )

    ax.set_xlim(0, max(counts + [1]) * 1.25)
    ax.set_xlabel("Number of students", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.set_title("Engagement breakdown", color="#c9d1d9", fontsize=10, pad=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _blur_region(
    img: np.ndarray,
    x: int, y: int, w: int, h: int,
    img_w: int, img_h: int,
) -> None:
    """
    In-place Gaussian blur of a rectangular region, clamped to image bounds.

    Kernel size is proportional to the region's diagonal so that large
    close-up faces are blurred as strongly as small distant ones.
    """
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return

    # Odd kernel — at least 15 × 15, at most 101 × 101
    diag   = math.hypot(x2 - x1, y2 - y1)
    k_size = int(diag * 0.35) | 1   # bitwise OR with 1 forces odd
    k_size = max(15, min(101, k_size))

    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k_size, k_size), 0)
