"""
pipeline/visualizer.py
======================
Privacy-safe image annotation and chart generation for the Gradio UI.

annotate_frame(bgr, persons)       → RGB numpy array
    1. Gaussian-blur every detected face region  (privacy first)
    2. Draw colour-coded body boxes (green / amber / red)
    3. Overlay score label and signal icons (H✋ P📱 T💬)

build_signal_chart(scores)         → matplotlib Figure
    Horizontal bar chart of per-signal class averages.

build_gauge(class_score, label)    → matplotlib Figure
    Semicircular gauge displaying the class-pulse score.
"""

from __future__ import annotations

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pipeline.detector import PersonDetection
from pipeline.scorer   import engagement_label, score_person, PULSE_COLOUR

# ---------------------------------------------------------------------------
# Colour palette — BGR for OpenCV, hex for matplotlib
# ---------------------------------------------------------------------------

_BGR: dict[str, tuple[int, int, int]] = {
    "engaged":    ( 39, 174,  96),   # green
    "neutral":    ( 18, 156, 243),   # amber  (low B, high G/R in BGR)
    "disengaged": ( 60,  76, 231),   # red
}

_HEX: dict[str, str] = {
    "engaged":    "#27ae60",
    "neutral":    "#f39c12",
    "disengaged": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _blur_face(img: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> None:
    """Gaussian-blur a face ROI in place.  Kernel scales with face size."""
    ih, iw = img.shape[:2]
    x1, y1 = max(0, fx), max(0, fy)
    x2, y2 = min(iw, fx + fw), min(ih, fy + fh)
    if x2 <= x1 or y2 <= y1:
        return
    diag = math.hypot(x2 - x1, y2 - y1)
    k    = max(15, int(diag * 0.40) | 1)   # force odd; proportional blur
    k    = min(k, 99)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)


# ---------------------------------------------------------------------------
# Public: annotate_frame
# ---------------------------------------------------------------------------

def annotate_frame(bgr: np.ndarray, persons: list[PersonDetection]) -> np.ndarray:
    """
    Return a privacy-safe annotated copy of *bgr* in RGB colour order.

    Faces are blurred before any box or label is drawn.
    Box colour encodes engagement level:
        green  → engaged    (score ≥ 0.60)
        amber  → neutral    (0.35 ≤ score < 0.60)
        red    → disengaged (score < 0.35)
    """
    out  = bgr.copy()
    ih, iw = out.shape[:2]

    # ── Privacy: blur all faces first ────────────────────────────────────────
    for p in persons:
        if p.fx >= 0 and p.fw > 0:
            _blur_face(out, p.fx, p.fy, p.fw, p.fh)

    # ── Draw annotation per person ────────────────────────────────────────────
    font    = cv2.FONT_HERSHEY_SIMPLEX
    for p in persons:
        score  = score_person(p)
        label  = engagement_label(score)
        colour = _BGR[label]
        thick  = max(2, p.bh // 55)

        # Body bounding box
        cv2.rectangle(out,
                      (p.bx, p.by), (p.bx + p.bw, p.by + p.bh),
                      colour, thick)

        # Score pill (top-left of box)
        score_txt            = f"{score:.0%}"
        fs                   = max(0.38, min(0.65, p.bw / 110))
        (tw, th), baseline   = cv2.getTextSize(score_txt, font, fs, 1)
        lx = max(0, p.bx)
        ly = max(th + baseline + 3, p.by - 2)
        cv2.rectangle(out,
                      (lx, ly - th - baseline - 2), (lx + tw + 6, ly + 2),
                      colour, -1)
        cv2.putText(out, score_txt,
                    (lx + 3, ly - baseline),
                    font, fs, (255, 255, 255), 1, cv2.LINE_AA)

        # Head-pose label (below score pill)
        if p.has_face and p.head_pose_label not in ("unknown",):
            pose_txt           = p.head_pose_label
            pfs                = fs * 0.72
            (ptw, pth), pbl    = cv2.getTextSize(pose_txt, font, pfs, 1)
            px = lx
            py = ly + pth + pbl + 4
            if py + pbl < ih:
                cv2.putText(out, pose_txt,
                            (px + 2, py),
                            font, pfs, colour, 1, cv2.LINE_AA)

        # Signal icons (top-right corner of box)
        icons: list[str] = []
        if p.hand_raised:    icons.append("H")
        if p.phone_detected: icons.append("P")
        if p.talking:        icons.append("T")
        if icons:
            icon_s = " ".join(icons)
            ifs    = max(0.35, min(0.55, p.bw / 150))
            (iw_, _), _ = cv2.getTextSize(icon_s, font, ifs, 1)
            ix = max(0, min(iw - iw_ - 4, p.bx + p.bw - iw_ - 4))
            iy = p.by + 18
            cv2.putText(out, icon_s, (ix, iy),
                        font, ifs, colour, 1, cv2.LINE_AA)

    # Convert BGR → RGB for Gradio
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Public: build_signal_chart
# ---------------------------------------------------------------------------

def build_signal_chart(scores: dict) -> plt.Figure:
    """
    Horizontal bar chart of per-signal class averages.

    Phone use is shown in red; other signals use the traffic-light scale.
    """
    avgs  = scores["proxy_avgs"]
    keys  = list(avgs.keys())
    vals  = [avgs[k] for k in keys]
    clrs  = []
    for k, v in avgs.items():
        if k == "Phone use":
            clrs.append("#e74c3c")
        elif v >= 0.65:
            clrs.append("#27ae60")
        elif v >= 0.40:
            clrs.append("#f39c12")
        else:
            clrs.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    bars = ax.barh(keys[::-1], [v * 100 for v in vals[::-1]],
                   color=clrs[::-1], height=0.55, edgecolor="white")
    ax.set_xlim(0, 112)
    ax.set_xlabel("Class average (%)", fontsize=9, color="#555")
    ax.set_title("Behavioral signal breakdown", fontsize=10, color="#333", pad=8)
    ax.axvline(60, color="#aaa", ls="--", lw=0.8)

    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=9, color="#444")

    ax.tick_params(colors="#555", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#ddd")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public: build_gauge
# ---------------------------------------------------------------------------

def build_gauge(class_score: float, pulse_label: str) -> plt.Figure:
    """
    Render a semicircular gauge displaying the class-pulse score.

    The arc spans 180° (π rad).  The filled portion is proportional to
    *class_score*; colour is keyed to *pulse_label*.
    """
    colour = PULSE_COLOUR.get(pulse_label, "#555")

    fig, ax = plt.subplots(figsize=(3.8, 2.6),
                           subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    # Background track
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg),
            lw=14, color="#e0e0e0", solid_capstyle="round", zorder=1)

    # Filled arc (progress)
    if class_score > 0:
        end_angle = np.pi * (1.0 - class_score)   # maps 0→π, 1→0
        theta_fill = np.linspace(np.pi, end_angle, 200)
        ax.plot(np.cos(theta_fill), np.sin(theta_fill),
                lw=14, color=colour, solid_capstyle="round", zorder=2)

    # Score text
    ax.text(0,  0.05, f"{class_score:.0%}",
            ha="center", va="center",
            fontsize=26, fontweight="bold", color=colour, zorder=3)
    ax.text(0, -0.38, f"{pulse_label} Engagement",
            ha="center", va="center",
            fontsize=10, color="#666", zorder=3)

    # Tick marks at 0 %, 50 %, 100 %
    for pct, ang in [(0, np.pi), (0.5, np.pi / 2), (1.0, 0)]:
        cx, cy = math.cos(ang) * 1.12, math.sin(ang) * 1.12
        ax.text(cx, cy, f"{int(pct*100)}%",
                ha="center", va="center", fontsize=7, color="#999")

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.60, 1.35)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig
