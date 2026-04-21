"""
gradio_app.py — Classroom Engagement Detector
=============================================
Single-page Gradio proof-of-concept.  No TensorFlow, no MediaPipe.

Detection pipeline (OpenCV only):
  1. CLAHE lighting normalisation
  2. Haar frontal + profile face cascade
  3. HOG pedestrian detector (bodies without visible faces)
  4. IoU NMS — merge / deduplicate
  5. Per-person behavioral signals:
       head pose   (face aspect ratio + eye symmetry)          30 %
       posture     (body-box h/w ratio)                        25 %
       hand raise  (skin-blob above face)                      25 %
       phone use   (bright rectangle in lap region)           −20 %
       talking     (face-pair proximity)                       10 %
  6. Attendance-adjusted class-pulse score

Usage:
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
from pipeline.visualizer import annotate_frame, build_signal_chart, build_gauge

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
# Core analysis function
# ---------------------------------------------------------------------------

def analyze(
    pil_image: Image.Image | None,
    expected_size: int,
    demo_mode: bool,
) -> tuple[np.ndarray, str, plt.Figure, plt.Figure]:
    """
    Run the full engagement pipeline on a PIL classroom image.

    Parameters
    ----------
    pil_image     : RGB PIL Image from the upload widget, or None.
    expected_size : Instructor-supplied expected class size (from slider).
    demo_mode     : If True, analyse the bundled sample image instead.

    Returns
    -------
    annotated_rgb  np.ndarray   — BGR→RGB annotated frame (faces blurred)
    metrics_html   str          — HTML KPI summary + legend + footer
    signal_chart   plt.Figure   — per-signal bar chart
    gauge_fig      plt.Figure   — class-pulse semicircular gauge
    """
    if demo_mode or pil_image is None:
        try:
            pil_image = Image.open(SAMPLE_PATH).convert("RGB")
        except Exception:
            return _empty_rgb(), _no_image_html(), _empty_fig(), _empty_fig()

    bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    try:
        persons = detect_persons(bgr)
    except Exception as exc:
        return (
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            f"<p style='color:#e74c3c; font-family:sans-serif;'>Detection error: {exc}</p>",
            _empty_fig(),
            _empty_fig(),
        )

    scores       = compute_scores(persons, int(expected_size))
    annotated    = annotate_frame(bgr, persons)
    signal_chart = build_signal_chart(scores)
    gauge_fig    = build_gauge(scores["class_score"], scores["pulse_label"])
    html         = _metrics_html(scores, is_demo=demo_mode)

    return annotated, html, signal_chart, gauge_fig


# ---------------------------------------------------------------------------
# CSS injected via gr.HTML — targets Gradio 6 rendered elements
# ---------------------------------------------------------------------------

_INJECT_CSS = """
<style>
/* ── Page & container ──────────────────────────────────────────────────── */
body, .gradio-container {
  background: #1a1a2e !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
/* Hide Gradio footer branding */
footer { display: none !important; }

/* ── Input panel wrapper ───────────────────────────────────────────────── */
.ced-panel {
  background: #16213e;
  border-radius: 14px;
  border: 1px solid #0e3460;
  box-shadow: 0 6px 28px rgba(0,0,0,0.35);
  padding: 14px 16px 18px;
}

/* ── Image upload — dashed teal border ─────────────────────────────────── */
.ced-upload .wrap { border: 2px dashed #0e7c7b !important; border-radius: 10px !important; }
.ced-upload .wrap:hover { border-color: #27ae60 !important; background: rgba(14,124,123,0.05) !important; }

/* ── Annotated image output — teal border ──────────────────────────────── */
.ced-img-out img { border: 2px solid #0e7c7b; border-radius: 8px; }

/* ── Gradient Analyze button ───────────────────────────────────────────── */
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

/* ── Slider label bold ─────────────────────────────────────────────────── */
.ced-slider label > span:first-child { font-weight: 700 !important; color: #c9d1d9 !important; }

/* ── Demo checkbox ─────────────────────────────────────────────────────── */
.ced-check label span { color: #8b949e !important; font-size: 0.88rem !important; }

/* ── Chart panel bg ────────────────────────────────────────────────────── */
.ced-chart { border-radius: 10px; overflow: hidden; }

/* ── Accordion ─────────────────────────────────────────────────────────── */
.ced-accordion {
  background: #16213e !important;
  border: 1px solid #21262d !important;
  border-radius: 10px !important;
  margin-top: 12px !important;
}
.ced-accordion .label-wrap span { color: #8b949e !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Header HTML
# ---------------------------------------------------------------------------

_HEADER = """
<div style="
  background: linear-gradient(135deg, #0f3460 0%, #0e4d8a 45%, #0e7c7b 100%);
  border-radius: 14px;
  padding: 22px 28px 18px;
  margin-bottom: 6px;
  box-shadow: 0 6px 30px rgba(0,0,0,0.40);
">
  <div style="display:flex; align-items:center; gap:12px;">
    <span style="font-size:2rem; line-height:1;">🎓</span>
    <div>
      <h2 style="color:#fff; margin:0; font-size:1.65rem; font-weight:800; letter-spacing:-0.3px;">
        Classroom Engagement Detector
      </h2>
      <p style="color:rgba(255,255,255,0.72); margin:4px 0 0; font-size:0.88rem; letter-spacing:0.3px;">
        Behavioral proxy analysis &nbsp;·&nbsp; Privacy-first &nbsp;·&nbsp; No face recognition
      </p>
    </div>
  </div>
  <div style="height:1.5px; background:rgba(255,255,255,0.22); margin-top:16px; border-radius:1px;"></div>
</div>
"""

# ---------------------------------------------------------------------------
# Metrics HTML builder
# ---------------------------------------------------------------------------

_PULSE_EMOJI = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}

# (icon, label_text, value_key, colour_fn)
_CARD_DEFS = [
    ("👥", "Detected",   "detected",          None),
    ("📋", "Attendance", "attendance_rate_pct", None),
    ("✅", "Engaged",    "engaged_count",       None),
    ("➡️", "Neutral",   "neutral_count",       None),
    ("❌", "Disengaged", "disengaged_count",    None),
]

_SMALL_CARD = (
    "display:inline-flex; flex-direction:column; align-items:center; "
    "background:#fff; border-radius:12px; padding:11px 14px; "
    "min-width:90px; text-align:center; "
    "box-shadow: 0 2px 10px rgba(0,0,0,0.09); "
    "border: 1.5px solid {border};"
)
_PULSE_CARD = (
    "display:inline-flex; flex-direction:column; align-items:center; "
    "background:#fff; border-radius:14px; padding:16px 26px; "
    "text-align:center; "
    "box-shadow: 0 4px 20px rgba(0,0,0,0.13); "
    "border: 2.5px solid {border}; "
    "margin-right: 6px;"
)


def _val_colour(label: str, pct: float | None = None) -> str:
    """Return a traffic-light hex colour for a KPI value."""
    if pct is None:
        return "#333"
    if pct >= 0.65:
        return "#27ae60"
    elif pct >= 0.40:
        return "#f39c12"
    return "#e74c3c"


def _small_card(icon: str, label: str, value: str, border: str) -> str:
    style = _SMALL_CARD.format(border=border)
    return (
        f"<div style='{style}'>"
        f"<span style='font-size:1.1rem;'>{icon}</span>"
        f"<span style='font-size:1.65rem; font-weight:800; color:#222; line-height:1.2;'>{value}</span>"
        f"<span style='font-size:0.7rem; color:#888; margin-top:2px;'>{label}</span>"
        f"</div>"
    )


def _metrics_html(s: dict, is_demo: bool = False) -> str:
    pulse_colour = PULSE_COLOUR.get(s["pulse_label"], "#555")
    att_rate     = s["attendance_rate"]
    eng_ratio    = s["engaged_count"] / max(s["detected"], 1)

    # ── Demo / warning banners ────────────────────────────────────────────────
    banners = ""
    if is_demo:
        banners += (
            "<div style='background:#fff8e1; border:1px solid #ffc107; "
            "border-radius:9px; padding:8px 14px; margin-bottom:10px; "
            "font-size:0.83rem; color:#795548; font-family:sans-serif;'>"
            "🎭 <strong>Demo mode</strong> — synthetic classroom image. "
            "Toggle off and upload a real photo for live analysis."
            "</div>"
        )
    if s["low_attendance"]:
        banners += (
            "<div style='background:#fdecea; border:1px solid #e74c3c; "
            "border-radius:9px; padding:8px 14px; margin-bottom:10px; "
            "font-size:0.83rem; color:#b71c1c; font-family:sans-serif;'>"
            f"⚠️ <strong>Low attendance:</strong> "
            f"{s['detected']} of {s['expected']} students detected "
            f"({att_rate:.0%}). Class score penalised."
            "</div>"
        )

    # ── Image legend ──────────────────────────────────────────────────────────
    legend = (
        "<div style='display:flex; gap:20px; justify-content:center; "
        "padding:8px 0 10px; font-size:0.82rem; color:#555; font-family:sans-serif;'>"
        "<span>🟢 Engaged&nbsp;(≥60%)</span>"
        "<span>🟡 Neutral&nbsp;(35–60%)</span>"
        "<span>🔴 Disengaged&nbsp;(&lt;35%)</span>"
        "</div>"
    )

    # ── Engagement badge ──────────────────────────────────────────────────────
    emoji = _PULSE_EMOJI.get(s["pulse_label"], "")
    badge = (
        f"<div style='margin-bottom:12px;'>"
        f"<span style='"
        f"background:{pulse_colour}22; color:{pulse_colour}; "
        f"border:1.5px solid {pulse_colour}; border-radius:22px; "
        f"padding:5px 18px; font-size:0.94rem; font-weight:700; "
        f"font-family:sans-serif; letter-spacing:0.2px;'>"
        f"{emoji} {s['pulse_label']} Engagement"
        f"</span>"
        f"</div>"
    )

    # ── Class Pulse card (large) ──────────────────────────────────────────────
    pulse_style = _PULSE_CARD.format(border=pulse_colour)
    pulse_card  = (
        f"<div style='{pulse_style}'>"
        f"<span style='font-size:1rem;'>📊</span>"
        f"<span style='font-size:3rem; font-weight:900; color:{pulse_colour}; "
        f"line-height:1.05; margin-top:2px;'>{s['class_score_pct']:.0f}%</span>"
        f"<span style='font-size:0.7rem; color:#888; margin-top:4px;'>Class Pulse</span>"
        f"</div>"
    )

    # ── Small cards ───────────────────────────────────────────────────────────
    att_border = "#27ae60" if att_rate >= 0.65 else ("#f39c12" if att_rate >= 0.40 else "#e74c3c")
    eng_border = "#27ae60" if eng_ratio >= 0.5 else "#f39c12"
    neu_border = "#ddd"
    dis_border = "#e74c3c" if s["disengaged_count"] > s["engaged_count"] else "#ddd"
    det_border = "#2980b9"

    small_cards = (
        _small_card("👥", "Detected",   str(s["detected"]),        det_border) +
        _small_card("📋", "Attendance", f"{att_rate:.0%}",          att_border) +
        _small_card("✅", "Engaged",    str(s["engaged_count"]),    eng_border) +
        _small_card("➡️", "Neutral",   str(s["neutral_count"]),    neu_border) +
        _small_card("❌", "Disengaged", str(s["disengaged_count"]), dis_border)
    )

    cards_row = (
        "<div style='display:flex; flex-wrap:wrap; gap:8px; align-items:stretch; "
        "margin-bottom:14px;'>"
        + pulse_card + small_cards +
        "</div>"
    )

    # ── Footer bar ────────────────────────────────────────────────────────────
    footer = (
        "<div style='background:linear-gradient(135deg,#0f3460,#0e4d8a); "
        "border-radius:10px; padding:10px 16px; "
        "display:flex; justify-content:space-between; align-items:center; "
        "margin-top:4px;'>"
        "<span style='color:rgba(255,255,255,0.68); font-size:0.75rem; font-family:sans-serif;'>"
        "🔒 Faces blurred · In-memory only · Aggregate class trends · No individual identified"
        "</span>"
        "<span style='color:rgba(255,255,255,0.45); font-size:0.74rem; font-family:sans-serif; "
        "white-space:nowrap; margin-left:12px;'>"
        "CIS 515 · ASU · Team 6 · 2026"
        "</span>"
        "</div>"
    )

    return (
        "<div style='font-family:sans-serif;'>"
        + banners + legend + badge + cards_row + footer
        + "</div>"
    )


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
# Placeholder outputs
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


# ---------------------------------------------------------------------------
# How-it-works accordion content
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

Baseline attentive student (head forward, upright, no phone, no hand raised)  →  **0.60 → Engaged**.

### Class pulse

```
class_pulse = Σ(individual scores) / expected_class_size × 100
```

Absent students implicitly score 0, so low attendance penalises the pulse naturally.

### Privacy safeguards

* Face regions **Gaussian-blurred** before any display.
* All processing **in-memory** — no files, no database, no logging.
* Output is **aggregate class-level only** — no individual is identified or tracked.
* Body-geometry proxies avoid the demographic biases documented in facial emotion recognition systems (Buolamwini & Gebru, 2018).
"""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Classroom Engagement Detector") as demo:

        # ── Inject CSS ────────────────────────────────────────────────────────
        gr.HTML(_INJECT_CSS)

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML(_HEADER)

        # ── Main layout ───────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left column — controls
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
                    "students in most of the frame."
                    "</p>"
                )

            # Right column — results
            with gr.Column(scale=2, min_width=400):
                img_output = gr.Image(
                    label="Annotated image — faces blurred",
                    height=330,
                    elem_classes=["ced-img-out"],
                )
                metrics_html = gr.HTML(value=_no_image_html())

                with gr.Row():
                    gauge_out  = gr.Plot(
                        label="Class pulse gauge",
                        scale=1,
                        elem_classes=["ced-chart"],
                    )
                    signal_out = gr.Plot(
                        label="Signal breakdown",
                        scale=2,
                        elem_classes=["ced-chart"],
                    )

        # ── Wire button ───────────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze,
            inputs=[img_input, expected_slider, demo_toggle],
            outputs=[img_output, metrics_html, signal_out, gauge_out],
        )

        # ── How it works accordion ────────────────────────────────────────────
        with gr.Accordion("ℹ️  How it works", open=False,
                          elem_classes=["ced-accordion"]):
            gr.Markdown(_HOW_IT_WORKS)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classroom Engagement Detector")
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
