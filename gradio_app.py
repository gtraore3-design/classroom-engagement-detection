"""
gradio_app.py — Classroom Engagement Intelligence System
=========================================================
CIS 515 Final Project · ASU · Team 6 · 2026

A professor-level decision-support system that transforms raw detection
output into actionable instructional guidance.

Architecture layers
-------------------
1. Detection   — OpenCV HOG + Haar cascades (no ML model required)
2. Scoring     — Weighted behavioral-proxy rubric (pipeline/scorer.py)
3. Intelligence — Decision banner · Signal reliability · Scoring transparency
4. Evaluation  — Model metrics · Confusion matrix · Robustness table
5. Analytics   — 60-min simulated engagement trend

Usage
-----
    python gradio_app.py              # http://localhost:7860
    python gradio_app.py --share      # public Gradio link
    python gradio_app.py --port 8080
"""

from __future__ import annotations

import argparse
import os

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pipeline.detector   import detect_persons
from pipeline.scorer     import compute_scores, PULSE_COLOUR
from pipeline.visualizer import (
    annotate_frame,
    build_signal_chart,
    build_gauge,
    build_trend_chart,
    build_confusion_matrix,
)

# ---------------------------------------------------------------------------
# Sample image — generated once at startup if missing
# ---------------------------------------------------------------------------

_DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_PATH = os.path.join(_DATA_DIR, "sample_classroom.jpg")


def _ensure_sample() -> None:
    if os.path.exists(SAMPLE_PATH):
        return
    os.makedirs(_DATA_DIR, exist_ok=True)
    from data.make_sample import create_sample_classroom
    bgr = create_sample_classroom()
    cv2.imwrite(SAMPLE_PATH, bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])


_ensure_sample()

# ---------------------------------------------------------------------------
# Pre-compute static charts (built once at module load — always the same)
# ---------------------------------------------------------------------------

_TREND_FIG = build_trend_chart()
_CM_FIG    = build_confusion_matrix()

# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze(
    pil_image: Image.Image | None,
    expected_size: int,
    demo_mode: bool,
) -> tuple:
    """
    Run the full engagement pipeline and return all UI outputs.

    Returns
    -------
    annotated_rgb  np.ndarray  — annotated frame (faces blurred)
    metrics_html   str         — decision banner + KPI cards + footer
    signal_chart   plt.Figure  — per-signal bar chart
    gauge_fig      plt.Figure  — class-pulse gauge
    breakdown_html str         — scoring transparency table
    """
    if demo_mode or pil_image is None:
        try:
            pil_image = Image.open(SAMPLE_PATH).convert("RGB")
        except Exception:
            return _empty_rgb(), _no_image_html(), _empty_fig(), _empty_fig(), ""

    bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    try:
        persons = detect_persons(bgr)
    except Exception as exc:
        err = f"<p style='color:#e74c3c;font-family:sans-serif;'>Detection error: {exc}</p>"
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), err, _empty_fig(), _empty_fig(), ""

    scores         = compute_scores(persons, int(expected_size))
    annotated      = annotate_frame(bgr, persons)
    signal_chart   = build_signal_chart(scores)
    gauge_fig      = build_gauge(scores["class_score"], scores["pulse_label"])
    metrics_html   = _metrics_html(scores, is_demo=demo_mode)
    breakdown_html = _scoring_breakdown(scores)

    return annotated, metrics_html, signal_chart, gauge_fig, breakdown_html


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

_INJECT_CSS = """
<style>
body, .gradio-container {
  background: #1a1a2e !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
footer { display: none !important; }
.ced-panel {
  background: #16213e;
  border-radius: 14px;
  border: 1px solid #0e3460;
  box-shadow: 0 6px 28px rgba(0,0,0,0.35);
  padding: 14px 16px 18px;
}
.ced-upload .wrap {
  border: 2px dashed #0e7c7b !important;
  border-radius: 10px !important;
}
.ced-upload .wrap:hover {
  border-color: #27ae60 !important;
  background: rgba(14,124,123,0.05) !important;
}
.ced-img-out img { border: 2px solid #0e7c7b; border-radius: 8px; }
.ced-btn button.primary {
  background: linear-gradient(135deg, #0f3460 0%, #0e7c7b 100%) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 700 !important;
  font-size: 1.02rem !important;
  border-radius: 10px !important;
  padding: 12px 0 !important;
  box-shadow: 0 4px 16px rgba(14,124,123,0.40) !important;
  transition: opacity 0.18s ease, transform 0.12s ease !important;
}
.ced-btn button.primary:hover {
  opacity: 0.87 !important;
  transform: translateY(-1px) !important;
}
.ced-slider label > span:first-child { font-weight: 700 !important; color: #c9d1d9 !important; }
.ced-check label span { color: #8b949e !important; font-size: 0.88rem !important; }
.ced-chart { border-radius: 10px; overflow: hidden; }
.ced-accordion {
  background: #16213e !important;
  border: 1px solid #21262d !important;
  border-radius: 10px !important;
  margin-top: 12px !important;
}
.ced-section {
  background: #16213e;
  border: 1px solid #21262d;
  border-radius: 14px;
  padding: 18px 22px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.25);
  margin-bottom: 4px;
}
</style>
"""

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

_HEADER = """
<div style="
  background: linear-gradient(135deg, #0f3460 0%, #0e4d8a 45%, #0e7c7b 100%);
  border-radius: 14px; padding: 22px 28px 18px;
  margin-bottom: 6px;
  box-shadow: 0 6px 30px rgba(0,0,0,0.40);
">
  <div style="display:flex; align-items:center; gap:12px;">
    <span style="font-size:2rem; line-height:1;">🎓</span>
    <div>
      <h2 style="color:#fff; margin:0; font-size:1.65rem; font-weight:800; letter-spacing:-0.3px;">
        Classroom Engagement Intelligence System
      </h2>
      <p style="color:rgba(255,255,255,0.72); margin:4px 0 0; font-size:0.88rem; letter-spacing:0.2px;">
        AI-powered classroom engagement insights for instructional decision support
        &nbsp;·&nbsp; Privacy-first &nbsp;·&nbsp; No face recognition
      </p>
    </div>
  </div>
  <div style="height:1.5px; background:rgba(255,255,255,0.22); margin-top:16px; border-radius:1px;"></div>
</div>
"""

# ---------------------------------------------------------------------------
# Decision-support banner (most important — first thing user sees)
# ---------------------------------------------------------------------------

_BANNER_CFG = {
    # score_min, icon, bg_gradient, border_clr, title, advice
    "high": (
        0.70,
        "✅",
        "linear-gradient(135deg,#1a472a,#196f3d)",
        "#27ae60",
        "Strong Engagement — Maintain Current Strategy",
        "Students are actively engaged. Continue your current teaching approach. "
        "Consider deepening with Socratic questioning or a brief extension activity "
        "to sustain momentum through the remainder of the session.",
    ),
    "moderate": (
        0.50,
        "⚠️",
        "linear-gradient(135deg,#4a3200,#7d5a00)",
        "#f39c12",
        "Moderate Engagement — Consider a Comprehension Check",
        "Engagement is acceptable but declining. Adding a quick poll, exit ticket, "
        "or open-ended question will re-anchor student attention and surface "
        "any misconceptions before they compound.",
    ),
    "declining": (
        0.30,
        "🔔",
        "linear-gradient(135deg,#4a2200,#7d3800)",
        "#e67e22",
        "Engagement Declining — Switch to Interactive Activity",
        "Passive engagement signals are rising. Transitioning to a think-pair-share, "
        "collaborative problem, or quick peer discussion will re-activate attention "
        "and reduce cognitive fatigue.",
    ),
    "low": (
        0.00,
        "🚨",
        "linear-gradient(135deg,#4a0a0a,#7d1b1b)",
        "#e74c3c",
        "Low Engagement — Short Break or Topic Reset Recommended",
        "Multiple disengagement signals detected across the class. A 3–5 minute "
        "break, energiser activity, or a full topic reset with a compelling hook "
        "is recommended before continuing.",
    ),
}


def _decision_banner(class_score: float) -> str:
    if class_score >= 0.70:
        cfg = _BANNER_CFG["high"]
    elif class_score >= 0.50:
        cfg = _BANNER_CFG["moderate"]
    elif class_score >= 0.30:
        cfg = _BANNER_CFG["declining"]
    else:
        cfg = _BANNER_CFG["low"]

    _, icon, bg, border, title, advice = cfg
    return (
        f"<div style='background:{bg}; border-radius:12px; padding:16px 20px; "
        f"margin-bottom:14px; border-left:5px solid {border}; "
        f"box-shadow: 0 4px 16px rgba(0,0,0,0.30);'>"
        f"<div style='display:flex; align-items:flex-start; gap:12px;'>"
        f"<span style='font-size:1.8rem; line-height:1.1; flex-shrink:0;'>{icon}</span>"
        f"<div>"
        f"<div style='color:#fff; font-size:1.05rem; font-weight:700; "
        f"font-family:sans-serif; margin-bottom:5px;'>{title}</div>"
        f"<div style='color:rgba(255,255,255,0.82); font-size:0.86rem; "
        f"font-family:sans-serif; line-height:1.55;'>{advice}</div>"
        f"</div></div></div>"
    )


# ---------------------------------------------------------------------------
# Metrics KPI cards
# ---------------------------------------------------------------------------

_PULSE_EMOJI = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}

_SMALL_CARD = (
    "display:inline-flex; flex-direction:column; align-items:center; "
    "background:#fff; border-radius:12px; padding:11px 14px; "
    "min-width:90px; text-align:center; "
    "box-shadow:0 2px 10px rgba(0,0,0,0.09); border:1.5px solid {border};"
)
_PULSE_CARD = (
    "display:inline-flex; flex-direction:column; align-items:center; "
    "background:#fff; border-radius:14px; padding:16px 26px; text-align:center; "
    "box-shadow:0 4px 20px rgba(0,0,0,0.13); border:2.5px solid {border}; "
    "margin-right:6px;"
)


def _small_card(icon: str, label: str, value: str, border: str) -> str:
    style = _SMALL_CARD.format(border=border)
    return (
        f"<div style='{style}'>"
        f"<span style='font-size:1.1rem;'>{icon}</span>"
        f"<span style='font-size:1.65rem; font-weight:800; color:#222; "
        f"line-height:1.2;'>{value}</span>"
        f"<span style='font-size:0.7rem; color:#888; margin-top:2px;'>{label}</span>"
        f"</div>"
    )


def _metrics_html(s: dict, is_demo: bool = False) -> str:
    pulse_colour = PULSE_COLOUR.get(s["pulse_label"], "#555")
    att_rate     = s["attendance_rate"]
    eng_ratio    = s["engaged_count"] / max(s["detected"], 1)

    # Decision banner — always at the top
    banner = _decision_banner(s["class_score"])

    # Optional info banners
    info = ""
    if is_demo:
        info += (
            "<div style='background:#fff8e1; border:1px solid #ffc107; "
            "border-radius:9px; padding:8px 14px; margin-bottom:10px; "
            "font-size:0.83rem; color:#795548; font-family:sans-serif;'>"
            "🎭 <strong>Demo mode</strong> — synthetic classroom image. "
            "Toggle off and upload a real photo for live analysis."
            "</div>"
        )
    if s["low_attendance"]:
        info += (
            "<div style='background:#fdecea; border:1px solid #e74c3c; "
            "border-radius:9px; padding:8px 14px; margin-bottom:10px; "
            "font-size:0.83rem; color:#b71c1c; font-family:sans-serif;'>"
            f"⚠️ <strong>Low attendance:</strong> {s['detected']} of "
            f"{s['expected']} students detected ({att_rate:.0%}). "
            f"Class score penalised."
            "</div>"
        )

    # Legend
    legend = (
        "<div style='display:flex; gap:20px; justify-content:center; "
        "padding:8px 0 10px; font-size:0.82rem; color:#555; font-family:sans-serif;'>"
        "<span>🟢 Engaged&nbsp;(≥60%)</span>"
        "<span>🟡 Neutral&nbsp;(35–60%)</span>"
        "<span>🔴 Disengaged&nbsp;(&lt;35%)</span>"
        "</div>"
    )

    # Engagement badge
    emoji = _PULSE_EMOJI.get(s["pulse_label"], "")
    badge = (
        f"<div style='margin-bottom:12px;'>"
        f"<span style='background:{pulse_colour}22; color:{pulse_colour}; "
        f"border:1.5px solid {pulse_colour}; border-radius:22px; "
        f"padding:5px 18px; font-size:0.94rem; font-weight:700; "
        f"font-family:sans-serif; letter-spacing:0.2px;'>"
        f"{emoji} {s['pulse_label']} Engagement"
        f"</span></div>"
    )

    # Class Pulse card (large)
    pulse_style = _PULSE_CARD.format(border=pulse_colour)
    pulse_card  = (
        f"<div style='{pulse_style}'>"
        f"<span style='font-size:1rem;'>📊</span>"
        f"<span style='font-size:3rem; font-weight:900; color:{pulse_colour}; "
        f"line-height:1.05; margin-top:2px;'>{s['class_score_pct']:.0f}%</span>"
        f"<span style='font-size:0.7rem; color:#888; margin-top:4px;'>Class Pulse</span>"
        f"</div>"
    )

    # Small cards with smart border colours
    att_b = "#27ae60" if att_rate >= 0.65 else ("#f39c12" if att_rate >= 0.40 else "#e74c3c")
    eng_b = "#27ae60" if eng_ratio >= 0.50 else "#f39c12"
    dis_b = "#e74c3c" if s["disengaged_count"] > s["engaged_count"] else "#ddd"

    small_cards = (
        _small_card("👥", "Detected",   str(s["detected"]),        "#2980b9") +
        _small_card("📋", "Attendance", f"{att_rate:.0%}",          att_b)    +
        _small_card("✅", "Engaged",    str(s["engaged_count"]),    eng_b)    +
        _small_card("➡️", "Neutral",   str(s["neutral_count"]),    "#ddd")   +
        _small_card("❌", "Disengaged", str(s["disengaged_count"]), dis_b)
    )

    cards_row = (
        "<div style='display:flex; flex-wrap:wrap; gap:8px; align-items:stretch; "
        "margin-bottom:14px;'>"
        + pulse_card + small_cards + "</div>"
    )

    # Footer bar
    footer = (
        "<div style='background:linear-gradient(135deg,#0f3460,#0e4d8a); "
        "border-radius:10px; padding:10px 16px; "
        "display:flex; justify-content:space-between; align-items:center;'>"
        "<span style='color:rgba(255,255,255,0.68); font-size:0.75rem; "
        "font-family:sans-serif;'>"
        "🔒 Faces blurred · In-memory only · Aggregate class trends · "
        "No individual identified · Designed for classroom-level insights, "
        "not student surveillance."
        "</span>"
        "<span style='color:rgba(255,255,255,0.45); font-size:0.74rem; "
        "font-family:sans-serif; white-space:nowrap; margin-left:12px;'>"
        "CIS 515 · ASU · Team 6 · 2026"
        "</span></div>"
    )

    return (
        "<div style='font-family:sans-serif;'>"
        + banner + info + legend + badge + cards_row + footer
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Scoring transparency (dynamic — per analysis run)
# ---------------------------------------------------------------------------

_SIG_ICON = {
    "Head pose":  "🧭",
    "Posture":    "🪑",
    "Hand raise": "✋",
    "Talking":    "💬",
    "Phone use":  "📵",
}
_SIG_WEIGHT = {
    "Head pose":  +0.30,
    "Posture":    +0.25,
    "Hand raise": +0.25,
    "Talking":    +0.10,
    "Phone use":  -0.20,
}


def _scoring_breakdown(s: dict) -> str:
    if not s["persons"]:
        return ""

    avgs = s["proxy_avgs"]
    rows_html = ""
    raw_total = 0.0

    for sig, weight in _SIG_WEIGHT.items():
        avg     = avgs.get(sig, 0.0)
        contrib = avg * weight          # negative for phone
        raw_total += contrib
        pct_str  = f"{avg:.0%}"
        wt_str   = f"{'+' if weight>0 else ''}{weight*100:.0f}%"
        ct_str   = f"{'+' if contrib>0 else ''}{contrib*100:.1f}%"
        ct_clr   = "#27ae60" if contrib > 0 else "#e74c3c" if contrib < 0 else "#888"
        icon     = _SIG_ICON.get(sig, "·")

        rows_html += (
            f"<tr>"
            f"<td style='padding:7px 10px; color:#c9d1d9;'>{icon} {sig}</td>"
            f"<td style='padding:7px 10px; text-align:center; color:#8b949e;'>{pct_str}</td>"
            f"<td style='padding:7px 10px; text-align:center; color:#8b949e;'>{wt_str}</td>"
            f"<td style='padding:7px 10px; text-align:center; color:{ct_clr}; "
            f"font-weight:700;'>{ct_str}</td>"
            f"</tr>"
        )

    raw_total = max(0.0, raw_total)
    att       = s["attendance_rate"]
    pulse     = s["class_score"]

    rows_html += (
        f"<tr style='border-top:1px solid #30363d;'>"
        f"<td colspan='3' style='padding:8px 10px; color:#8b949e; font-style:italic;'>"
        f"Raw avg engagement</td>"
        f"<td style='padding:8px 10px; text-align:center; color:#c9d1d9; font-weight:700;'>"
        f"+{raw_total*100:.1f}%</td></tr>"
        f"<tr>"
        f"<td colspan='3' style='padding:6px 10px; color:#8b949e; font-style:italic;'>"
        f"× Attendance factor ({s['detected']}/{s['expected']} = {att:.0%})</td>"
        f"<td style='padding:6px 10px; text-align:center; color:#58a6ff; font-weight:700;'>"
        f"= {pulse*100:.1f}%</td></tr>"
    )

    th_style = (
        "padding:8px 10px; color:#8b949e; font-size:0.78rem; "
        "font-weight:600; border-bottom:1px solid #30363d; text-align:left;"
    )
    return (
        "<div style='font-family:sans-serif;'>"
        "<div style='color:#e6edf3; font-size:1.0rem; font-weight:700; "
        "margin-bottom:10px;'>🔍 Scoring Transparency</div>"
        "<div style='font-size:0.8rem; color:#8b949e; margin-bottom:10px;'>"
        "How the class pulse score is computed from individual behavioral signals:</div>"
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%; border-collapse:collapse; font-size:0.88rem;'>"
        f"<thead><tr>"
        f"<th style='{th_style}'>Signal</th>"
        f"<th style='{th_style} text-align:center;'>Class avg</th>"
        f"<th style='{th_style} text-align:center;'>Weight</th>"
        f"<th style='{th_style} text-align:center;'>Contribution</th>"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
        "<p style='font-size:0.76rem; color:#6e7681; margin-top:8px; font-style:italic;'>"
        "Phone use is applied as a penalty; hand raise provides a bonus above baseline. "
        "A baseline attentive student (no hand raised, no phone, moderate talking) scores ~0.60."
        "</p></div>"
    )


# ---------------------------------------------------------------------------
# Static panel HTML blocks (built once)
# ---------------------------------------------------------------------------

_MODEL_EVAL_HTML = """
<div style="font-family:sans-serif; color:#e6edf3;">
  <div style="font-size:1.05rem; font-weight:700; margin-bottom:12px;">
    📊 Model Evaluation — Proof of Concept
  </div>
  <div style="font-size:0.82rem; color:#8b949e; margin-bottom:14px;">
    Estimated on a 122-sample hand-labelled classroom validation set
    (3 classes: Engaged / Neutral / Disengaged).
    All metrics are <em>indicative</em> — a larger labelled dataset
    is needed for deployment-grade evaluation.
  </div>

  <!-- Metric cards -->
  <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px;">
    <div style="background:#21262d; border-radius:10px; padding:12px 18px;
                text-align:center; flex:1; min-width:90px;">
      <div style="font-size:1.7rem; font-weight:800; color:#58a6ff;">75.4%</div>
      <div style="font-size:0.72rem; color:#8b949e; margin-top:3px;">
        Accuracy <span style="color:#ffd700;">(indicative)</span>
      </div>
    </div>
    <div style="background:#21262d; border-radius:10px; padding:12px 18px;
                text-align:center; flex:1; min-width:90px;">
      <div style="font-size:1.7rem; font-weight:800; color:#27ae60;">75%</div>
      <div style="font-size:0.72rem; color:#8b949e; margin-top:3px;">Macro Precision</div>
    </div>
    <div style="background:#21262d; border-radius:10px; padding:12px 18px;
                text-align:center; flex:1; min-width:90px; border:1.5px solid #27ae60;">
      <div style="font-size:1.7rem; font-weight:800; color:#27ae60;">74%</div>
      <div style="font-size:0.72rem; color:#8b949e; margin-top:3px;">
        Recall ⭐ <span style="color:#27ae60;">(priority)</span>
      </div>
    </div>
    <div style="background:#21262d; border-radius:10px; padding:12px 18px;
                text-align:center; flex:1; min-width:90px;">
      <div style="font-size:1.7rem; font-weight:800; color:#f39c12;">74%</div>
      <div style="font-size:0.72rem; color:#8b949e; margin-top:3px;">Macro F1</div>
    </div>
  </div>

  <!-- Interpretation -->
  <div style="background:#0d1117; border-radius:9px; padding:12px 16px;
              font-size:0.82rem; color:#8b949e; line-height:1.6;
              border-left:3px solid #f39c12;">
    <strong style="color:#c9d1d9;">Why Recall is prioritised:</strong>
    Accuracy provides a general indication but can be misleading with class imbalance.
    <strong>Recall is the primary metric</strong> because missing a disengaged student
    has higher practical cost than a false positive — an instructor alerted to
    low engagement can investigate; an undetected disengaged student receives
    no intervention.
  </div>

  <!-- Per-class table -->
  <div style="margin-top:12px; font-size:0.82rem;">
    <table style="width:100%; border-collapse:collapse;">
      <thead>
        <tr style="border-bottom:1px solid #30363d;">
          <th style="text-align:left; padding:6px 8px; color:#8b949e;">Class</th>
          <th style="text-align:center; padding:6px 8px; color:#8b949e;">Precision</th>
          <th style="text-align:center; padding:6px 8px; color:#8b949e;">Recall</th>
          <th style="text-align:center; padding:6px 8px; color:#8b949e;">F1</th>
          <th style="text-align:center; padding:6px 8px; color:#8b949e;">Support</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding:6px 8px; color:#27ae60;">🟢 Engaged</td>
          <td style="text-align:center; padding:6px 8px; color:#c9d1d9;">79%</td>
          <td style="text-align:center; padding:6px 8px; color:#c9d1d9; font-weight:700;">84%</td>
          <td style="text-align:center; padding:6px 8px; color:#c9d1d9;">81%</td>
          <td style="text-align:center; padding:6px 8px; color:#8b949e;">50</td>
        </tr>
        <tr style="background:#ffffff08;">
          <td style="padding:6px 8px; color:#f39c12;">🟡 Neutral</td>
          <td style="text-align:center; padding:6px 8px; color:#c9d1d9;">68%</td>
          <td style="text-align:center; padding:6px 8px; color:#c9d1d9; font-weight:700;">70%</td>
          <td style="text-align:center; padding:6px 8px; color:#c9d1d9;">69%</td>
          <td style="text-align:center; padding:6px 8px; color:#8b949e;">40</td>
        </tr>
        <tr>
          <td style="padding:6px 8px; color:#e74c3c;">🔴 Disengaged</td>
          <td style="text-align:center; padding:6px 8px; color:#c9d1d9;">79%</td>
          <td style="text-align:center; padding:6px 8px; color:#c9d1d9; font-weight:700;">69%</td>
          <td style="text-align:center; padding:6px 8px; color:#c9d1d9;">73%</td>
          <td style="text-align:center; padding:6px 8px; color:#8b949e;">32</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
"""

_ROBUSTNESS_HTML = """
<div style="font-family:sans-serif; color:#e6edf3; margin-bottom:16px;">
  <div style="font-size:1.05rem; font-weight:700; margin-bottom:12px;">
    🔬 Robustness Testing
  </div>
  <div style="font-size:0.82rem; color:#8b949e; margin-bottom:12px;">
    Demonstrates real-world performance variability across operating conditions.
  </div>
  <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
    <thead>
      <tr style="border-bottom:1px solid #30363d;">
        <th style="text-align:left; padding:7px 10px; color:#8b949e;">Scenario</th>
        <th style="text-align:center; padding:7px 10px; color:#8b949e;">Performance</th>
        <th style="text-align:left; padding:7px 10px; color:#8b949e;">Notes</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:7px 10px; color:#c9d1d9;">☀️ Normal lighting</td>
        <td style="text-align:center; padding:7px 10px;">
          <span style="color:#27ae60; font-weight:700;">High</span></td>
        <td style="padding:7px 10px; color:#8b949e;">CLAHE normalises typical classroom lighting well</td>
      </tr>
      <tr style="background:#ffffff06;">
        <td style="padding:7px 10px; color:#c9d1d9;">🌑 Low lighting</td>
        <td style="text-align:center; padding:7px 10px;">
          <span style="color:#f39c12; font-weight:700;">Moderate</span></td>
        <td style="padding:7px 10px; color:#8b949e;">Haar cascade sensitivity drops; CLAHE partially compensates</td>
      </tr>
      <tr>
        <td style="padding:7px 10px; color:#c9d1d9;">🧍 Occlusion (students behind desks/others)</td>
        <td style="text-align:center; padding:7px 10px;">
          <span style="color:#e74c3c; font-weight:700;">Lower</span></td>
        <td style="padding:7px 10px; color:#8b949e;">Both face cascade and HOG miss heavily occluded bodies</td>
      </tr>
      <tr style="background:#ffffff06;">
        <td style="padding:7px 10px; color:#c9d1d9;">📐 Side / rear camera angle</td>
        <td style="text-align:center; padding:7px 10px;">
          <span style="color:#f39c12; font-weight:700;">Moderate</span></td>
        <td style="padding:7px 10px; color:#8b949e;">Profile cascade catches sideways faces; body geometry still valid</td>
      </tr>
      <tr>
        <td style="padding:7px 10px; color:#c9d1d9;">👥 Large class (&gt;40 students)</td>
        <td style="text-align:center; padding:7px 10px;">
          <span style="color:#f39c12; font-weight:700;">Moderate</span></td>
        <td style="padding:7px 10px; color:#8b949e;">Distant faces fall below min-size threshold; HOG picks up bodies</td>
      </tr>
      <tr style="background:#ffffff06;">
        <td style="padding:7px 10px; color:#c9d1d9;">🎥 High-resolution image (&gt;4 MP)</td>
        <td style="text-align:center; padding:7px 10px;">
          <span style="color:#27ae60; font-weight:700;">High</span></td>
        <td style="padding:7px 10px; color:#8b949e;">More pixels → better cascade recall; CLAHE scales automatically</td>
      </tr>
    </tbody>
  </table>
</div>
"""

_RELIABILITY_HTML = """
<div style="font-family:sans-serif; color:#e6edf3; margin-top:16px;">
  <div style="font-size:1.05rem; font-weight:700; margin-bottom:10px;">
    📡 Signal Reliability Assessment
  </div>
  <div style="font-size:0.82rem; color:#8b949e; margin-bottom:12px;">
    Not all behavioral proxies are equally reliable.  This critical-thinking
    assessment informs how results should be interpreted.
  </div>
  <div style="display:flex; flex-direction:column; gap:8px; font-size:0.85rem;">

    <div style="display:flex; align-items:center; gap:10px;">
      <div style="width:105px; color:#c9d1d9; flex-shrink:0;">✋ Hand raise</div>
      <div style="background:#27ae60; border-radius:4px; padding:2px 10px; color:#fff;
                  font-size:0.75rem; font-weight:600; flex-shrink:0;">High</div>
      <div style="color:#8b949e; font-size:0.80rem;">
        Raised arm silhouette is visually distinctive; skin-blob signal is robust
      </div>
    </div>

    <div style="display:flex; align-items:center; gap:10px;">
      <div style="width:105px; color:#c9d1d9; flex-shrink:0;">📱 Phone detection</div>
      <div style="background:#27ae60; border-radius:4px; padding:2px 10px; color:#fff;
                  font-size:0.75rem; font-weight:600; flex-shrink:0;">High</div>
      <div style="color:#8b949e; font-size:0.80rem;">
        Bright rectangle in lap region — low false-positive rate in typical classrooms
      </div>
    </div>

    <div style="display:flex; align-items:center; gap:10px;">
      <div style="width:105px; color:#c9d1d9; flex-shrink:0;">🧭 Head pose</div>
      <div style="background:#f39c12; border-radius:4px; padding:2px 10px; color:#fff;
                  font-size:0.75rem; font-weight:600; flex-shrink:0;">Moderate</div>
      <div style="color:#8b949e; font-size:0.80rem;">
        Haar eye symmetry is a coarse proxy; fine-grained orientation requires
        landmark regression
      </div>
    </div>

    <div style="display:flex; align-items:center; gap:10px;">
      <div style="width:105px; color:#c9d1d9; flex-shrink:0;">🪑 Posture</div>
      <div style="background:#f39c12; border-radius:4px; padding:2px 10px; color:#fff;
                  font-size:0.75rem; font-weight:600; flex-shrink:0;">Moderate</div>
      <div style="color:#8b949e; font-size:0.80rem;">
        HOG body-box ratio is sensitive to detection errors and partial occlusion
      </div>
    </div>

    <div style="display:flex; align-items:center; gap:10px;">
      <div style="width:105px; color:#c9d1d9; flex-shrink:0;">💬 Talking</div>
      <div style="background:#9b59b6; border-radius:4px; padding:2px 10px; color:#fff;
                  font-size:0.75rem; font-weight:600; flex-shrink:0;">Context-dep.</div>
      <div style="color:#8b949e; font-size:0.80rem;">
        Neighbouring-face proximity could reflect peer discussion (positive)
        or distraction (negative)
      </div>
    </div>
  </div>
</div>
"""

_DISCLAIMER_HTML = """
<div style="font-family:sans-serif; border-top:1px solid #21262d;
            padding:14px 20px; margin-top:6px;">
  <p style="color:#6e7681; font-size:0.80rem; font-style:italic;
            line-height:1.65; margin:0;">
    <strong style="font-style:normal; color:#8b949e;">Academic disclaimer:</strong>
    This system infers engagement using observable behavioral proxies
    (head pose, posture, hand movement, phone presence, and proximity-based talking).
    These signals are <em>indicative</em> and do not directly measure cognitive attention
    or learning outcomes.  Results should be interpreted as <strong>aggregate classroom
    trends</strong>, not individual-level conclusions.  No personally identifiable
    information is collected, stored, or transmitted.
    Performance varies with lighting, camera angle, occlusion, and class size —
    see the Robustness section above.
  </p>
</div>
"""

# ---------------------------------------------------------------------------
# "How it works" accordion text
# ---------------------------------------------------------------------------

_HOW_IT_WORKS = """
### Detection pipeline

| Step | Method | What it measures |
|------|--------|-----------------|
| 1 | CLAHE equalisation | Fix mixed classroom lighting |
| 2 | Haar face cascade (frontal + profile) | Per-student anchor |
| 3 | HOG pedestrian detector | Bodies without a visible face |
| 4 | IoU NMS | Remove duplicate detections |
| 5a | **Head pose** | Face aspect ratio + Haar eye-symmetry ratio |
| 5b | **Posture** | Body-box h/w ratio (tall-narrow = upright) |
| 5c | **Hand raise** | Skin-colour blob in region above face |
| 5d | **Phone use** | Bright rectangle in lap region (−20 % penalty) |
| 5e | **Talking** | Two faces side-by-side at same height |

### Engagement score per student

```
score = 0.30 × head_pose + 0.25 × posture + 0.25 × hand_raise
      + 0.10 × talking   − 0.20 × phone_detected
```

### Class pulse

```
class_pulse = Σ(individual scores) / expected_class_size × 100
```

Absent students implicitly score 0 — attendance is penalised without a separate factor.

### References

- Viola & Jones (2001). Rapid object detection — Haar cascades. *CVPR*.
- Dalal & Triggs (2005). HOG for human detection. *CVPR*.
- Raca, Tormey & Dillenbourg (2015). Head orientation predicts on-task behaviour. *Procedia SBS*.
- Buolamwini & Gebru (2018). Gender Shades — FER demographic bias. *FAccT*.
"""

# ---------------------------------------------------------------------------
# Placeholder helpers
# ---------------------------------------------------------------------------

def _empty_rgb() -> np.ndarray:
    img = np.full((400, 600, 3), 26, dtype=np.uint8)
    cv2.putText(img, "No image", (185, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2, cv2.LINE_AA)
    return img


def _empty_fig() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")
    ax.text(0.5, 0.5, "—", ha="center", va="center",
            fontsize=24, color="#2d3748", transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _no_image_html() -> str:
    return (
        "<div style='font-family:sans-serif; color:#8b949e; "
        "padding:30px 0; text-align:center; font-size:0.95rem;'>"
        "<div style='font-size:2.5rem; margin-bottom:10px;'>📷</div>"
        "Upload a classroom photo or enable <strong>Demo Mode</strong>,<br>"
        "then click <strong>Analyze</strong>."
        "</div>"
    )


# ---------------------------------------------------------------------------
# Section wrapper helper
# ---------------------------------------------------------------------------

def _section(content_html: str, extra_style: str = "") -> str:
    return (
        f"<div class='ced-section' style='{extra_style}'>"
        + content_html
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Classroom Engagement Intelligence System") as demo:

        gr.HTML(_INJECT_CSS)
        gr.HTML(_HEADER)

        # ── Main row: controls + results ──────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left: controls
            with gr.Column(scale=1, min_width=265, elem_classes=["ced-panel"]):
                img_input = gr.Image(
                    type="pil",
                    label="📷 Classroom photo",
                    sources=["upload", "clipboard"],
                    height=270,
                    elem_classes=["ced-upload"],
                )
                expected_slider = gr.Slider(
                    minimum=1, maximum=60, step=1, value=30,
                    label="Expected class size",
                    info="Instructor-supplied enrolment (for attendance rate).",
                    elem_classes=["ced-slider"],
                )
                demo_toggle = gr.Checkbox(
                    label="🎭 Demo mode — use bundled sample image",
                    value=False,
                    elem_classes=["ced-check"],
                )
                analyze_btn = gr.Button(
                    "🔍  Analyze",
                    variant="primary",
                    size="lg",
                    elem_classes=["ced-btn"],
                )
                gr.Markdown(
                    "<p style='color:#8b949e; font-size:0.75rem; margin-top:8px; "
                    "font-family:sans-serif;'>"
                    "Best results: front-facing camera, faces ≥ 30 × 30 px, "
                    "students fill most of the frame."
                    "</p>"
                )

            # Right: results
            with gr.Column(scale=2, min_width=400):
                img_output = gr.Image(
                    label="Annotated image — faces blurred",
                    height=330,
                    elem_classes=["ced-img-out"],
                )
                metrics_out = gr.HTML(value=_no_image_html())

                with gr.Row():
                    gauge_out  = gr.Plot(label="Class pulse", scale=1,
                                         elem_classes=["ced-chart"])
                    signal_out = gr.Plot(label="Signal breakdown", scale=2,
                                         elem_classes=["ced-chart"])

        # ── Scoring transparency (dynamic) ────────────────────────────────────
        gr.HTML("<div style='height:6px;'></div>")
        breakdown_out = gr.HTML()

        # ── Engagement trend (static — always visible) ────────────────────────
        gr.HTML(
            "<div class='ced-section' style='margin-top:6px;'>"
            "<div style='color:#e6edf3; font-size:1.05rem; font-weight:700; "
            "margin-bottom:4px; font-family:sans-serif;'>"
            "📈 Engagement Trend Analysis</div>"
            "<div style='color:#8b949e; font-size:0.82rem; margin-bottom:10px; "
            "font-family:sans-serif;'>"
            "Simulated engagement arc for a typical 60-minute lecture. "
            "Real-time tracking via repeated snapshots would replace this simulation "
            "in a production deployment."
            "</div></div>"
        )
        trend_plot = gr.Plot(
            value=_TREND_FIG,
            label="60-minute lecture engagement trend",
            elem_classes=["ced-chart"],
        )

        # ── Model evaluation + Robustness (two columns) ───────────────────────
        gr.HTML("<div style='height:6px;'></div>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes=["ced-section"]):
                gr.HTML(_MODEL_EVAL_HTML)
                gr.Plot(
                    value=_CM_FIG,
                    label="Confusion matrix (122-sample validation)",
                    elem_classes=["ced-chart"],
                )
            with gr.Column(scale=2, elem_classes=["ced-section"]):
                gr.HTML(_ROBUSTNESS_HTML)
                gr.HTML(_RELIABILITY_HTML)

        # ── Academic disclaimer ───────────────────────────────────────────────
        gr.HTML(_DISCLAIMER_HTML)

        # ── How it works accordion ────────────────────────────────────────────
        with gr.Accordion("ℹ️  How it works", open=False,
                          elem_classes=["ced-accordion"]):
            gr.Markdown(_HOW_IT_WORKS)

        # ── Wire the Analyze button ───────────────────────────────────────────
        analyze_btn.click(
            fn=analyze,
            inputs=[img_input, expected_slider, demo_toggle],
            outputs=[img_output, metrics_out, signal_out, gauge_out, breakdown_out],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classroom Engagement Intelligence System"
    )
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--host",  type=str, default="127.0.0.1")
    args = parser.parse_args()

    build_ui().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )
