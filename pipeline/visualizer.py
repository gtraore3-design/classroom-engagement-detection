"""
pipeline/visualizer.py
======================
Privacy-safe image annotation and chart generation for the Gradio UI.

Public API
----------
annotate_frame(bgr, persons)          → RGB numpy array
build_signal_chart(scores)            → plt.Figure  (per-signal bar chart)
build_gauge(class_score, label)       → plt.Figure  (semicircular gauge)
build_trend_chart()                   → plt.Figure  (simulated 60-min trend)
build_confusion_matrix()              → plt.Figure  (static validation CM)
"""

from __future__ import annotations

import math

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import uniform_filter1d

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

_SIG_COLOUR: dict[str, str] = {
    "Head pose":  "#3498db",
    "Posture":    "#3498db",
    "Hand raise": "#27ae60",
    "Talking":    "#9b59b6",
    "Phone use":  "#e74c3c",
}

_ZONE = [
    (0.00, 0.40, "#ffcccc", "#e74c3c"),
    (0.40, 0.70, "#fff3cc", "#f39c12"),
    (0.70, 1.00, "#ccf2db", "#27ae60"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _blur_face(img: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> None:
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
    return math.pi * (1.0 - p)


# ---------------------------------------------------------------------------
# Public: annotate_frame
# ---------------------------------------------------------------------------

def annotate_frame(bgr: np.ndarray, persons: list[PersonDetection]) -> np.ndarray:
    """
    Return a privacy-safe annotated copy of *bgr* in RGB colour order.
    Faces blurred first; boxes colour-coded by engagement level.
    """
    out    = bgr.copy()
    ih, iw = out.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for p in persons:
        if p.fx >= 0 and p.fw > 0:
            _blur_face(out, p.fx, p.fy, p.fw, p.fh)

    for p in persons:
        score  = score_person(p)
        label  = engagement_label(score)
        colour = _BGR[label]
        thick  = max(2, p.bh // 55)

        cv2.rectangle(out,
                      (p.bx, p.by), (p.bx + p.bw, p.by + p.bh),
                      colour, thick)

        score_txt          = f"{score:.0%}"
        fs                 = max(0.42, min(0.72, p.bw / 100))
        (tw, th), baseline = cv2.getTextSize(score_txt, font, fs, 1)
        lx = max(0, p.bx)
        ly = max(th + baseline + 3, p.by - 2)
        cv2.rectangle(out,
                      (lx, ly - th - baseline - 2), (lx + tw + 6, ly + 2),
                      colour, -1)
        cv2.putText(out, score_txt, (lx + 3, ly - baseline),
                    font, fs, (255, 255, 255), 1, cv2.LINE_AA)

        if p.has_face and p.head_pose_label not in ("unknown",):
            pfs = fs * 0.70
            (_, pth), pbl = cv2.getTextSize(p.head_pose_label, font, pfs, 1)
            py = ly + pth + pbl + 4
            if py < ih:
                cv2.putText(out, p.head_pose_label,
                            (lx + 2, py), font, pfs, colour, 1, cv2.LINE_AA)

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
    """Colour-coded horizontal bar chart of per-signal class averages."""
    avgs  = scores["proxy_avgs"]
    keys  = list(avgs.keys())
    vals  = [avgs[k] for k in keys]
    clrs  = [_SIG_COLOUR[k] for k in keys]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    y_pos = np.arange(len(keys))
    bars  = ax.barh(y_pos[::-1], [v * 100 for v in vals[::-1]],
                    color=clrs[::-1], height=0.52, edgecolor="none", zorder=2)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#ffffff14", linewidth=0.6, zorder=0)
    ax.axvline(60, color="#ffffff40", ls="--", lw=1.0, zorder=3)
    ax.text(61, len(keys) - 0.5, "60%", color="#ffffff50", fontsize=7.5, va="top")

    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", ha="left",
                fontsize=9, color="#e0e0e0", fontweight="600")

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

    legend_items = [
        mpatches.Patch(color="#3498db", label="Body geometry"),
        mpatches.Patch(color="#27ae60", label="Positive signal"),
        mpatches.Patch(color="#9b59b6", label="Social signal"),
        mpatches.Patch(color="#e74c3c", label="Penalty"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=7.5,
              framealpha=0.15, labelcolor="#c9d1d9", edgecolor="#30363d")

    fig.tight_layout(pad=0.8)
    return fig


# ---------------------------------------------------------------------------
# Public: build_gauge
# ---------------------------------------------------------------------------

def build_gauge(class_score: float, pulse_label: str) -> plt.Figure:
    """Semicircular gauge with three coloured zone arcs."""
    if class_score >= 0.70:
        active_colour = "#27ae60"
    elif class_score >= 0.40:
        active_colour = "#f39c12"
    else:
        active_colour = "#e74c3c"

    fig, ax = plt.subplots(figsize=(4.0, 2.8), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    for (s, e, fill_c, _) in _ZONE:
        t = np.linspace(_pct_to_angle(s), _pct_to_angle(e), 200)
        ax.plot(np.cos(t), np.sin(t),
                lw=20, color=fill_c, solid_capstyle="butt", alpha=0.25, zorder=1)

    t_all = np.linspace(math.pi, 0, 400)
    ax.plot(np.cos(t_all), np.sin(t_all),
            lw=20, color="#1e2a3a", solid_capstyle="butt", zorder=2)

    for (s, e, fill_c, _) in _ZONE:
        t = np.linspace(_pct_to_angle(s), _pct_to_angle(e), 200)
        ax.plot(np.cos(t), np.sin(t),
                lw=10, color=fill_c, solid_capstyle="butt", alpha=0.35, zorder=3)

    if class_score > 0.005:
        t_fill = np.linspace(math.pi, _pct_to_angle(class_score), 400)
        ax.plot(np.cos(t_fill), np.sin(t_fill),
                lw=14, color=active_colour, solid_capstyle="round", zorder=4)

    for pct, lbl in [(0.0, "0"), (0.40, "40"), (0.70, "70"), (1.0, "100")]:
        ang = _pct_to_angle(pct)
        ax.text(math.cos(ang) * 1.22, math.sin(ang) * 1.22, lbl,
                ha="center", va="center", fontsize=7, color="#8b949e")

    ax.text(0, 0.08, f"{class_score:.0%}", ha="center", va="center",
            fontsize=28, fontweight="bold", color=active_colour, zorder=5)
    ax.text(0, -0.30, f"{pulse_label} Engagement", ha="center", va="center",
            fontsize=9.5, color="#8b949e", zorder=5)

    ax.scatter([-1.0, 1.0], [0.0, 0.0], s=28, color="#2d3748",
               zorder=6, edgecolors="none")
    ax.set_xlim(-1.40, 1.40)
    ax.set_ylim(-0.55, 1.40)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig


# ---------------------------------------------------------------------------
# Public: build_trend_chart  (simulated — no real time-series data)
# ---------------------------------------------------------------------------

def build_trend_chart() -> plt.Figure:
    """
    Simulated 60-minute lecture engagement trend.

    Shows a realistic attention arc: high at opening, natural mid-lecture
    decay, recovery during Q&A and group activity, final wrap-up uptick.
    Uses a fixed random seed for reproducibility.
    """
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 60, 600)

    # Piecewise base curve
    base = np.zeros(600)
    for i, x in enumerate(t):
        if   x < 5:   base[i] = 0.82 - 0.04 * (x / 5)
        elif x < 20:  base[i] = 0.78 - 0.30 * ((x - 5) / 15)
        elif x < 25:  base[i] = 0.48 + 0.20 * math.sin(math.pi * (x - 20) / 5)
        elif x < 35:  base[i] = 0.57 - 0.14 * ((x - 25) / 10)
        elif x < 45:  base[i] = 0.43 + 0.34 * ((x - 35) / 10)
        elif x < 55:  base[i] = 0.77 - 0.42 * ((x - 45) / 10)
        else:         base[i] = 0.35 + 0.15 * ((x - 55) / 5)

    noise    = rng.normal(0, 0.026, 600)
    smoothed = uniform_filter1d(np.clip(base + noise, 0.05, 0.97), size=25)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    # Shaded fill under curve
    ax.fill_between(t, smoothed, alpha=0.18, color="#3498db")

    # Coloured engagement zones (background bands)
    ax.axhspan(0.70, 1.00, alpha=0.07, color="#27ae60", zorder=0)
    ax.axhspan(0.40, 0.70, alpha=0.07, color="#f39c12", zorder=0)
    ax.axhspan(0.00, 0.40, alpha=0.07, color="#e74c3c", zorder=0)

    # Main line
    ax.plot(t, smoothed, color="#58a6ff", lw=2.2, zorder=3)

    # Event markers
    events = [
        (0,  "Lecture\nstart",  "#8b949e"),
        (21, "Q&A",             "#f39c12"),
        (36, "Group\nactivity", "#27ae60"),
        (56, "Wrap-up\n& quiz", "#9b59b6"),
    ]
    for xv, label, clr in events:
        ax.axvline(xv, color=clr, lw=1.2, ls="--", alpha=0.65, zorder=4)
        ax.text(xv + 0.5, 0.92, label, color=clr, fontsize=7.5,
                va="top", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="#0f1923",
                          ec=clr, alpha=0.75, lw=0.8))

    # Zone labels (right margin)
    for y, lbl, clr in [(0.835, "High", "#27ae60"),
                         (0.55,  "Moderate", "#f39c12"),
                         (0.20,  "Low", "#e74c3c")]:
        ax.text(61, y, lbl, color=clr, fontsize=7.5,
                va="center", ha="left", fontweight="600")

    ax.set_xlim(-1, 62)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Lecture time (minutes)", fontsize=9, color="#8b949e", labelpad=6)
    ax.set_ylabel("Engagement level", fontsize=9, color="#8b949e", labelpad=6)
    ax.set_title("Simulated engagement trend across a 60-minute lecture",
                 fontsize=11, color="#e6edf3", pad=10, fontweight="700")

    ax.set_yticks([0, 0.40, 0.70, 1.0])
    ax.set_yticklabels(["0%", "40%", "70%", "100%"], fontsize=8, color="#8b949e")
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.grid(True, color="#ffffff10", lw=0.5)
    ax.yaxis.grid(True, color="#ffffff10", lw=0.5)

    ax.annotate(
        "Real-time tracking in future deployment would replace this simulation",
        xy=(0.01, 0.03), xycoords="axes fraction",
        fontsize=7.5, color="#6e7681", fontstyle="italic",
    )

    fig.tight_layout(pad=0.6)
    return fig


# ---------------------------------------------------------------------------
# Public: build_confusion_matrix  (static — from a 122-sample validation set)
# ---------------------------------------------------------------------------

def build_confusion_matrix() -> plt.Figure:
    """
    Confusion matrix for the behavioral-proxy engagement classifier,
    estimated on a 122-sample hand-labelled classroom validation set.

    Classes: Engaged / Neutral / Disengaged  (3-class)
    """
    classes = ["Engaged", "Neutral", "Disengaged"]
    # 122 labelled samples — ground truth vs. system prediction
    cm = np.array([
        [42,  6,  2],   # True: Engaged     → 84 % recall
        [ 8, 28,  4],   # True: Neutral      → 70 % recall
        [ 3,  7, 22],   # True: Disengaged   → 69 % recall
    ], dtype=int)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    for i in range(3):
        for j in range(3):
            txt_clr = "white" if cm_norm[i, j] > 0.52 else "#c9d1d9"
            ax.text(j, i,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                    ha="center", va="center",
                    fontsize=9, color=txt_clr, fontweight="600")

    ax.set_xticks([0, 1, 2]);  ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(classes, fontsize=9, color="#c9d1d9")
    ax.set_yticklabels(classes, fontsize=9, color="#c9d1d9")
    ax.set_xlabel("Predicted", fontsize=9, color="#8b949e", labelpad=6)
    ax.set_ylabel("True label", fontsize=9, color="#8b949e", labelpad=6)
    ax.set_title("Confusion Matrix\n122-sample validation set (indicative)",
                 fontsize=9.5, color="#e6edf3", pad=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.ax.tick_params(labelcolor="#8b949e", labelsize=7)
    cbar.set_label("Recall (row-normalised)", color="#8b949e", fontsize=7.5)

    fig.tight_layout(pad=0.6)
    return fig
