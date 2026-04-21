"""
pipeline/visualizer.py
======================
Privacy-safe image annotation and chart generation for the Gradio UI.

annotate_frame(bgr, persons)   → RGB numpy array
    1. Gaussian-blur every detected face region  (privacy first)
    2. Draw colour-coded body boxes (green / amber / red)
    3. Overlay score label and signal icons

build_signal_chart(scores)     → matplotlib Figure
    Colour-coded horizontal bar chart: green = positive, red = penalty,
    blue = neutral.  Value labels at the end of each bar.

build_gauge(class_score, label) → matplotlib Figure
    Semicircular gauge with three coloured zone arcs:
        green  70 – 100 %
        yellow 40 –  70 %
        red     0 –  40 %
"""

from __future__ import annotations

import math

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pipeline.detector import PersonDetection
from pipeline.scorer   import engagement_label, score_person, PULSE_COLOUR

# ---------------------------------------------------------------------------
# Colour palette — BGR for OpenCV
# ---------------------------------------------------------------------------

_BGR: dict[str, tuple[int, int, int]] = {
    "engaged":    ( 39, 174,  96),
    "neutral":    ( 18, 156, 243),
    "disengaged": ( 60,  76, 231),
}

# Per-signal colours for bar chart (hex, matplotlib)
_SIG_COLOUR: dict[str, str] = {
    "Head pose":  "#3498db",   # blue   – attentional direction
    "Posture":    "#3498db",   # blue   – body geometry
    "Hand raise": "#27ae60",   # green  – active participation bonus
    "Talking":    "#9b59b6",   # purple – social / context-dependent
    "Phone use":  "#e74c3c",   # red    – penalty signal
}

# Zone colours for gauge background arcs (light fills)
_ZONE = [
    (0.00, 0.40, "#ffcccc", "#e74c3c"),   # red zone   0 – 40 %
    (0.40, 0.70, "#fff3cc", "#f39c12"),   # amber zone 40 – 70 %
    (0.70, 1.00, "#ccf2db", "#27ae60"),   # green zone 70 – 100 %
]


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
    k    = max(15, int(diag * 0.40) | 1)
    k    = min(k, 99)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)


def _pct_to_angle(p: float) -> float:
    """Map percentage [0, 1] to gauge angle in radians (π → 0, left → right)."""
    return math.pi * (1.0 - p)


# ---------------------------------------------------------------------------
# Public: annotate_frame
# ---------------------------------------------------------------------------

def annotate_frame(bgr: np.ndarray, persons: list[PersonDetection]) -> np.ndarray:
    """
    Return a privacy-safe annotated copy of *bgr* in RGB colour order.

    Box colour encodes engagement level:
        green  → engaged    (score ≥ 0.60)
        amber  → neutral    (0.35 ≤ score < 0.60)
        red    → disengaged (score < 0.35)
    """
    out    = bgr.copy()
    ih, iw = out.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX

    # ── Privacy: blur all faces first ────────────────────────────────────────
    for p in persons:
        if p.fx >= 0 and p.fw > 0:
            _blur_face(out, p.fx, p.fy, p.fw, p.fh)

    # ── Draw annotation per person ────────────────────────────────────────────
    for p in persons:
        score  = score_person(p)
        label  = engagement_label(score)
        colour = _BGR[label]
        thick  = max(2, p.bh // 55)

        # Body bounding box
        cv2.rectangle(out,
                      (p.bx, p.by), (p.bx + p.bw, p.by + p.bh),
                      colour, thick)

        # Score pill (top-left of box) — slightly larger text
        score_txt          = f"{score:.0%}"
        fs                 = max(0.42, min(0.72, p.bw / 100))
        (tw, th), baseline = cv2.getTextSize(score_txt, font, fs, 1)
        lx = max(0, p.bx)
        ly = max(th + baseline + 3, p.by - 2)
        cv2.rectangle(out,
                      (lx, ly - th - baseline - 2), (lx + tw + 6, ly + 2),
                      colour, -1)
        cv2.putText(out, score_txt,
                    (lx + 3, ly - baseline),
                    font, fs, (255, 255, 255), 1, cv2.LINE_AA)

        # Head-pose sub-label
        if p.has_face and p.head_pose_label not in ("unknown",):
            pfs = fs * 0.70
            (ptw, pth), pbl = cv2.getTextSize(p.head_pose_label, font, pfs, 1)
            py = ly + pth + pbl + 4
            if py < ih:
                cv2.putText(out, p.head_pose_label,
                            (lx + 2, py), font, pfs, colour, 1, cv2.LINE_AA)

        # Signal icons (top-right of box)
        icons: list[str] = []
        if p.hand_raised:    icons.append("H")
        if p.phone_detected: icons.append("P")
        if p.talking:        icons.append("T")
        if icons:
            icon_s = " ".join(icons)
            ifs    = max(0.36, min(0.55, p.bw / 150))
            (iw_, _), _ = cv2.getTextSize(icon_s, font, ifs, 1)
            ix = max(0, min(iw - iw_ - 4, p.bx + p.bw - iw_ - 4))
            cv2.putText(out, icon_s, (ix, p.by + 18),
                        font, ifs, colour, 1, cv2.LINE_AA)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Public: build_signal_chart
# ---------------------------------------------------------------------------

def build_signal_chart(scores: dict) -> plt.Figure:
    """
    Colour-coded horizontal bar chart of per-signal class averages.

    Colours
    -------
    Blue   — head pose, posture (neutral body-geometry signals)
    Green  — hand raise          (active-participation bonus)
    Purple — talking             (social / context-dependent)
    Red    — phone use           (penalty signal)
    """
    avgs  = scores["proxy_avgs"]
    keys  = list(avgs.keys())
    vals  = [avgs[k] for k in keys]
    clrs  = [_SIG_COLOUR[k] for k in keys]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    y_pos = np.arange(len(keys))
    bars  = ax.barh(
        y_pos[::-1],
        [v * 100 for v in vals[::-1]],
        color=clrs[::-1],
        height=0.52,
        edgecolor="none",
        zorder=2,
    )

    # Grid lines
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#ffffff14", linewidth=0.6, zorder=0)

    # 60 % reference line
    ax.axvline(60, color="#ffffff40", ls="--", lw=1.0, zorder=3)
    ax.text(61, len(keys) - 0.5, "60%", color="#ffffff50",
            fontsize=7.5, va="top")

    # Value labels at end of each bar
    for bar, val in zip(bars, vals[::-1]):
        ax.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0%}",
            va="center", ha="left",
            fontsize=9, color="#e0e0e0", fontweight="600",
        )

    ax.set_xlim(0, 118)
    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(keys[::-1], fontsize=9, color="#c9d1d9")
    ax.set_xlabel("Class average (%)", fontsize=8.5, color="#8b949e", labelpad=6)
    ax.set_title("Behavioral signal breakdown",
                 fontsize=10.5, color="#e6edf3", pad=10, fontweight="700")

    ax.tick_params(axis="x", colors="#8b949e", labelsize=8)
    ax.tick_params(axis="y", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Legend
    legend_items = [
        mpatches.Patch(color="#3498db", label="Body geometry"),
        mpatches.Patch(color="#27ae60", label="Positive signal"),
        mpatches.Patch(color="#9b59b6", label="Social signal"),
        mpatches.Patch(color="#e74c3c", label="Penalty"),
    ]
    ax.legend(
        handles=legend_items, loc="lower right",
        fontsize=7.5, framealpha=0.15,
        labelcolor="#c9d1d9", edgecolor="#30363d",
    )

    fig.tight_layout(pad=0.8)
    return fig


# ---------------------------------------------------------------------------
# Public: build_gauge
# ---------------------------------------------------------------------------

def build_gauge(class_score: float, pulse_label: str) -> plt.Figure:
    """
    Semicircular gauge with three coloured zone arcs.

    Zone arcs (background):
        red    0 – 40 %
        amber 40 – 70 %
        green 70 – 100 %

    The active progress arc is drawn on top in the zone colour.
    """
    # Determine active colour from zone
    if class_score >= 0.70:
        active_colour = "#27ae60"
    elif class_score >= 0.40:
        active_colour = "#f39c12"
    else:
        active_colour = "#e74c3c"

    fig, ax = plt.subplots(figsize=(4.0, 2.8),
                           subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    # ── Zone background arcs ──────────────────────────────────────────────────
    for (s, e, fill_c, _) in _ZONE:
        a_start = _pct_to_angle(s)
        a_end   = _pct_to_angle(e)
        # linspace from high angle (low pct) to low angle (high pct)
        t = np.linspace(a_start, a_end, 200)
        ax.plot(np.cos(t), np.sin(t),
                lw=20, color=fill_c, solid_capstyle="butt",
                alpha=0.25, zorder=1)

    # ── Darker track outline ──────────────────────────────────────────────────
    t_all = np.linspace(math.pi, 0, 400)
    ax.plot(np.cos(t_all), np.sin(t_all),
            lw=20, color="#1e2a3a", solid_capstyle="butt", zorder=2)

    # ── Zone fills (brighter, narrower band on top of track) ─────────────────
    for (s, e, fill_c, _) in _ZONE:
        a_start = _pct_to_angle(s)
        a_end   = _pct_to_angle(e)
        t = np.linspace(a_start, a_end, 200)
        ax.plot(np.cos(t), np.sin(t),
                lw=10, color=fill_c, solid_capstyle="butt",
                alpha=0.35, zorder=3)

    # ── Active progress arc ───────────────────────────────────────────────────
    if class_score > 0.005:
        t_fill = np.linspace(math.pi, _pct_to_angle(class_score), 400)
        ax.plot(np.cos(t_fill), np.sin(t_fill),
                lw=14, color=active_colour,
                solid_capstyle="round", zorder=4)

    # ── Zone tick marks ───────────────────────────────────────────────────────
    for pct, label_txt in [(0.0, "0"), (0.40, "40"), (0.70, "70"), (1.0, "100")]:
        ang = _pct_to_angle(pct)
        tx  = math.cos(ang) * 1.22
        ty  = math.sin(ang) * 1.22
        ax.text(tx, ty, label_txt, ha="center", va="center",
                fontsize=7, color="#8b949e")

    # ── Score text (centre) ───────────────────────────────────────────────────
    ax.text(0,  0.08, f"{class_score:.0%}",
            ha="center", va="center",
            fontsize=28, fontweight="bold", color=active_colour, zorder=5)
    ax.text(0, -0.30, f"{pulse_label} Engagement",
            ha="center", va="center",
            fontsize=9.5, color="#8b949e", zorder=5)

    # ── Endpoint dots ─────────────────────────────────────────────────────────
    ax.scatter([-1.0, 1.0], [0.0, 0.0], s=28, color="#2d3748",
               zorder=6, edgecolors="none")

    ax.set_xlim(-1.40, 1.40)
    ax.set_ylim(-0.55, 1.40)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig
