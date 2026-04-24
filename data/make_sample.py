"""
data/make_sample.py
===================
Generate a synthetic classroom image for the demo mode of gradio_app.py.

The image is designed to look like a front-of-class camera view of a
small lecture room.  Students are drawn as simple geometric figures so
the UI layout can be demonstrated without a real photograph.

NOTE: Haar face cascades require realistic face textures to trigger
      detections reliably.  The synthetic faces here use ellipses and
      circles — they may or may not fire the cascade depending on the
      OpenCV version.  Upload a real classroom photo for accurate results.
"""

import math
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

_SKIN = [
    (180, 150, 120), (160, 120,  90), (200, 175, 145),
    (140, 105,  75), (215, 190, 165), (165, 125,  95),
]
_SHIRT = [
    ( 80, 120, 180), (120, 160, 100), ( 80,  80, 160),
    (160,  80,  80), ( 80, 160, 160), (140, 140,  80),
    (100, 100, 200), (180, 120,  80), (100, 160, 130),
]
_HAIR = [
    ( 35,  25,  18), ( 55,  45,  30), ( 75,  55,  25),
    (175, 135,  75), ( 25,  25,  25), ( 90,  60,  20),
]


def _darken(c: tuple, d: int = 25) -> tuple:
    return tuple(max(0, v - d) for v in c)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def create_sample_classroom(width: int = 960, height: int = 640) -> np.ndarray:
    """
    Render a synthetic classroom and return a BGR NumPy array.

    Layout
    ------
    * Light-grey wall with a green chalkboard strip at top.
    * 4 rows × 5 columns of stylised student figures.
    * Each figure has a coloured torso, an oval head with eyes, and a desk.
    * Two students at the front-left are drawn with one arm raised to
      simulate a hand-raise signal.
    """
    img = np.full((height, width, 3), (230, 225, 215), dtype=np.uint8)

    # ── Chalkboard ────────────────────────────────────────────────────────────
    cv2.rectangle(img, (60, 15), (900, 105), (55, 80, 55), -1)
    cv2.rectangle(img, (60, 15), (900, 105), (35, 55, 35), 2)
    cv2.putText(
        img,
        "Demo classroom  —  upload a real photo for live analysis",
        (90, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (195, 215, 195), 1, cv2.LINE_AA,
    )

    # ── Chalk tray ────────────────────────────────────────────────────────────
    cv2.rectangle(img, (60, 103), (900, 115), (180, 175, 165), -1)

    # ── Floor line ───────────────────────────────────────────────────────────
    cv2.line(img, (0, height - 60), (width, height - 60), (180, 175, 165), 2)

    # ── Students (4 rows × 5 columns) ────────────────────────────────────────
    rows, cols = 4, 5
    xs = np.linspace(110, 850, cols, dtype=int)
    ys = np.linspace(175, 555, rows, dtype=int)

    for ri, cy in enumerate(ys):
        for ci, cx in enumerate(xs):
            idx   = ri * cols + ci
            skin  = _SKIN[idx % len(_SKIN)]
            shirt = _SHIRT[idx % len(_SHIRT)]
            hair  = _HAIR[idx % len(_HAIR)]

            # ── Desk ─────────────────────────────────────────────────────────
            dy = cy + 28
            cv2.rectangle(img, (cx - 44, dy), (cx + 44, dy + 38),
                          (195, 185, 165), -1)
            cv2.rectangle(img, (cx - 44, dy), (cx + 44, dy + 38),
                          (160, 150, 135), 1)

            # ── Torso ────────────────────────────────────────────────────────
            cv2.rectangle(img, (cx - 23, cy - 4), (cx + 23, cy + 28),
                          shirt, -1)
            cv2.rectangle(img, (cx - 23, cy - 4), (cx + 23, cy + 28),
                          _darken(shirt), 1)

            # ── Neck ─────────────────────────────────────────────────────────
            cv2.rectangle(img, (cx - 5, cy - 14), (cx + 5, cy),
                          skin, -1)

            # ── Head ─────────────────────────────────────────────────────────
            hcy = cy - 33
            cv2.ellipse(img, (cx, hcy), (20, 26), 0, 0, 360, skin, -1)
            cv2.ellipse(img, (cx, hcy), (20, 26), 0, 0, 360, _darken(skin, 18), 1)

            # ── Hair ─────────────────────────────────────────────────────────
            cv2.ellipse(img, (cx, hcy - 10), (21, 16), 0, 180, 360, hair, -1)

            # ── Eyes ─────────────────────────────────────────────────────────
            eye_y = hcy - 6
            for ex in (cx - 7, cx + 7):
                cv2.circle(img, (ex, eye_y), 4, (255, 255, 255), -1)
                cv2.circle(img, (ex, eye_y), 2, (50, 35, 25), -1)

            # ── Raised arm — first student in rows 0 & 1 ─────────────────────
            if ci == 0 and ri in (0, 1):
                # Upper arm
                ax1, ay1 = cx - 23, cy - 4
                ax2, ay2 = cx - 50, cy - 55
                cv2.line(img, (ax1, ay1), (ax2, ay2), skin, 8)
                # Lower arm + hand
                ax3, ay3 = cx - 55, cy - 85
                cv2.line(img, (ax2, ay2), (ax3, ay3), skin, 7)
                cv2.circle(img, (ax3, ay3), 9, skin, -1)

    # ── Light blur to soften harsh edges ─────────────────────────────────────
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    out = os.path.join(os.path.dirname(__file__), "sample_classroom.jpg")
    img = create_sample_classroom()
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved {out}  ({img.shape[1]}×{img.shape[0]})")
