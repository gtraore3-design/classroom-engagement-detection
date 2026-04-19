"""
sample_image.py — Generate a synthetic top-down classroom diagram using Pillow.

The generated image is a visual representation of a 4-row × 5-column classroom:
  - A whiteboard at the top (front of room)
  - Grey circles for student seats arranged in rows
  - Instructor podium at the front

This image is used exclusively in demo mode.  Because it contains no real
faces, MediaPipe cannot process it — the demo mode therefore bypasses the
MediaPipe pipeline entirely and uses pre-computed synthetic results instead.

The image is generated once and cached in memory via functools.lru_cache.
"""

from __future__ import annotations

import functools
import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_W, IMG_H = 800, 600          # output image size (pixels)
MARGIN        = 60               # border margin

# Classroom grid: 4 rows × 5 columns
ROWS, COLS    = 4, 5
SEAT_RADIUS   = 22               # radius of each seat circle

# Per-seat engagement colours (green=high, yellow=medium, red=low, grey=empty)
# Laid out row-by-row, front (row 0) → back (row 3).
# Scores roughly match the synthetic PersonResult objects in demo_data.py.
_SEAT_SCORES = [
    # Row 0 (front — closest to board)
    [0.88, 0.76, 0.92, 0.81, 0.73],
    # Row 1
    [0.65, 0.71, 0.58, 0.68, 0.60],
    # Row 2
    [0.42, 0.55, 0.48, 0.38, 0.51],
    # Row 3 (back — furthest from board)
    [0.31, 0.25, 0.44, 0.22, 0.35],
]


def _score_to_rgb(score: float) -> tuple[int, int, int]:
    """
    Map an engagement score in [0, 1] to an RGB colour tuple.
    Uses a red–yellow–green gradient (same palette as the heatmap overlay).
    """
    # hue: 0 = red (HSV 0°), 0.33 = green (HSV 120°)
    import colorsys
    h = score * 0.33
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.90)
    return int(r * 255), int(g * 255), int(b * 255)


def _generate_classroom_png() -> np.ndarray:
    """
    Render the synthetic classroom diagram and return it as a BGR NumPy array.
    Called once; result is cached by get_demo_image_bgr().
    """
    # ---- Background ----
    img = Image.new("RGB", (IMG_W, IMG_H), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)

    # ---- Classroom border ----
    draw.rectangle(
        [MARGIN, MARGIN, IMG_W - MARGIN, IMG_H - MARGIN],
        outline=(160, 160, 155), width=3,
    )

    # ---- Whiteboard (front of room) ----
    board_y1, board_y2 = MARGIN + 8, MARGIN + 52
    draw.rectangle(
        [MARGIN + 20, board_y1, IMG_W - MARGIN - 20, board_y2],
        fill=(230, 235, 255), outline=(90, 100, 180), width=2,
    )
    draw.text(
        ((IMG_W) // 2, (board_y1 + board_y2) // 2),
        "WHITEBOARD / PROJECTION SCREEN",
        fill=(60, 70, 160), anchor="mm",
    )

    # ---- Instructor podium ----
    pod_cx = IMG_W // 2
    pod_cy = board_y2 + 28
    draw.rectangle(
        [pod_cx - 30, pod_cy - 12, pod_cx + 30, pod_cy + 12],
        fill=(200, 180, 140), outline=(100, 80, 40), width=2,
    )
    draw.text((pod_cx, pod_cy), "Instructor", fill=(60, 40, 0), anchor="mm")

    # ---- Seat grid ----
    usable_w = IMG_W - 2 * MARGIN - 40
    usable_h = IMG_H - 2 * MARGIN - 140     # leave space for board + legend

    col_step = usable_w // (COLS + 1)
    row_step = usable_h // (ROWS + 1)

    # Store seat centres for legend placement
    seat_centres: list[tuple[int, int]] = []

    for row in range(ROWS):
        for col in range(COLS):
            cx = MARGIN + 20 + col_step * (col + 1)
            cy = board_y2 + 60 + row_step * (row + 1)
            seat_centres.append((cx, cy))

            score  = _SEAT_SCORES[row][col]
            colour = _score_to_rgb(score)

            # Seat shadow
            draw.ellipse(
                [cx - SEAT_RADIUS + 2, cy - SEAT_RADIUS + 2,
                 cx + SEAT_RADIUS + 2, cy + SEAT_RADIUS + 2],
                fill=(180, 180, 175),
            )
            # Seat fill
            draw.ellipse(
                [cx - SEAT_RADIUS, cy - SEAT_RADIUS,
                 cx + SEAT_RADIUS, cy + SEAT_RADIUS],
                fill=colour, outline=(255, 255, 255), width=2,
            )
            # Score label inside circle
            draw.text(
                (cx, cy), f"{score:.0%}", fill=(255, 255, 255), anchor="mm",
            )

        # Row label on the left
        label_y = board_y2 + 60 + row_step * (row + 1)
        row_name = ["Front", "Row 2", "Row 3", "Back"][row]
        draw.text(
            (MARGIN + 5, label_y), row_name,
            fill=(100, 100, 100), anchor="lm",
        )

    # ---- Legend ----
    legend_y  = IMG_H - MARGIN + 5
    legend_items = [
        ((80, 180, 60),  "High engagement (≥70%)"),
        ((220, 190, 30), "Medium (40–70%)"),
        ((200, 60, 40),  "Low engagement (<40%)"),
    ]
    lx = MARGIN + 20
    for colour, label in legend_items:
        draw.ellipse([lx, legend_y, lx + 16, legend_y + 16], fill=colour)
        draw.text((lx + 22, legend_y + 8), label, fill=(80, 80, 80), anchor="lm")
        lx += 200

    # ---- Title ----
    draw.text(
        (IMG_W // 2, MARGIN - 22),
        "DEMO — Synthetic Classroom Engagement Overview",
        fill=(40, 40, 40), anchor="mm",
    )

    # ---- Convert PIL → BGR NumPy array ----
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr


@functools.lru_cache(maxsize=1)
def get_demo_image_bgr() -> np.ndarray:
    """
    Return the synthetic classroom diagram as a BGR NumPy array.
    Result is cached after the first call so the image is only generated once.
    """
    return _generate_classroom_png()
