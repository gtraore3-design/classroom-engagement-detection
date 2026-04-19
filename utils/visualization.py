"""
Visualization utilities: heatmap overlay and annotation drawing.

All displayed images are pre-blurred by pipeline/face_blur.py.
These functions annotate the blurred image only.
"""

from __future__ import annotations

import colorsys
import math

import cv2
import numpy as np

from config import HEATMAP_ALPHA, ENGAGED_THRESHOLD
from pipeline.behavioral_detection import PersonResult
from pipeline.engagement_scorer import score_person


def _engagement_color_bgr(score: float) -> tuple[int, int, int]:
    """Map score [0,1] to BGR: red (0) → yellow (0.5) → green (1)."""
    # Hue: 0 = red, 0.33 = green
    hue = score * 0.33
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    return int(b * 255), int(g * 255), int(r * 255)


def engagement_heatmap_overlay(
    bgr_image: np.ndarray,
    person_results: list[PersonResult],
    radius: int = 60,
) -> np.ndarray:
    """
    Render a semi-transparent Gaussian blob for each person, coloured by
    their engagement score.  Returns a blended copy of bgr_image.
    """
    h, w = bgr_image.shape[:2]
    heatmap = np.zeros((h, w, 3), dtype=np.float32)

    for res in person_results:
        score = score_person(res)
        cx, cy = int(res.center[0]), int(res.center[1])
        color  = np.array(_engagement_color_bgr(score), dtype=np.float32)

        # Draw a filled ellipse scaled by image size
        ry = min(int(radius * 1.5), h // 4)
        rx = min(radius, w // 8)

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        # Soft-edge via Gaussian blur
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rx // 2, sigmaY=ry // 2)
        mask = mask / (mask.max() + 1e-9)

        heatmap += mask[:, :, np.newaxis] * color[np.newaxis, np.newaxis, :]

    # Clip and convert
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)

    # Alpha blend
    base = bgr_image.astype(np.float32)
    heat = heatmap.astype(np.float32)
    blended = (1 - HEATMAP_ALPHA) * base + HEATMAP_ALPHA * heat
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_engagement_annotations(
    bgr_image: np.ndarray,
    person_results: list[PersonResult],
    show_score: bool = True,
) -> np.ndarray:
    """
    Draw a coloured circle and optional score label near each person centre
    on the already-blurred image.
    """
    out = bgr_image.copy()
    for res in person_results:
        score = score_person(res)
        color = _engagement_color_bgr(score)
        cx, cy = int(res.center[0]), int(res.center[1])

        cv2.circle(out, (cx, cy), 18, color, -1)
        cv2.circle(out, (cx, cy), 18, (255, 255, 255), 2)

        if show_score:
            label = f"{score:.0%}"
            font_scale = 0.6
            thickness  = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            tx = cx - tw // 2
            ty = cy - 25
            cv2.putText(out, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 1)
            cv2.putText(out, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return out
