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
       head pose   (face aspect ratio + eye symmetry)
       posture     (body-box h/w ratio)
       hand raise  (skin-blob above face)
       phone use   (bright rectangle in lap region — penalty)
       talking     (face-pair proximity)
  6. Attendance-adjusted class-pulse score

Usage:
    python gradio_app.py              # http://localhost:7860
    python gradio_app.py --share      # public Gradio link
    python gradio_app.py --port 8080
"""

from __future__ import annotations

import argparse
import os
import sys

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
    """Create data/sample_classroom.jpg if it doesn't exist."""
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
    metrics_html   str          — HTML KPI summary
    signal_chart   plt.Figure   — per-signal bar chart
    gauge_fig      plt.Figure   — class-pulse semicircular gauge
    """
    if demo_mode or pil_image is None:
        try:
            pil_image = Image.open(SAMPLE_PATH).convert("RGB")
        except Exception:
            return _empty_rgb(), _no_image_html(), _empty_fig(), _empty_fig()

    # PIL (RGB) → OpenCV (BGR)
    bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    try:
        persons = detect_persons(bgr)
    except Exception as exc:
        return (
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            f"<p style='color:red'>Detection error: {exc}</p>",
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
# HTML helpers
# ---------------------------------------------------------------------------

_CARD = (
    "display:inline-block; background:#fff; border:1px solid #e0e0e0; "
    "border-radius:10px; padding:12px 18px; text-align:center; "
    "min-width:100px; margin:4px;"
)
_VAL  = "font-size:1.9rem; font-weight:700; margin:0; line-height:1.1;"
_LBL  = "font-size:0.76rem; color:#777; margin:4px 0 0 0;"


def _kpi(value: str, label: str, colour: str = "#333") -> str:
    return (
        f"<div style='{_CARD}'>"
        f"<p style='{_VAL} color:{colour};'>{value}</p>"
        f"<p style='{_LBL}'>{label}</p>"
        f"</div>"
    )


def _metrics_html(s: dict, is_demo: bool = False) -> str:
    pulse_colour = PULSE_COLOUR.get(s["pulse_label"], "#555")
    att_colour   = "#e74c3c" if s["low_attendance"] else "#27ae60"

    demo_banner = (
        "<div style='background:#fff8e1; border:1px solid #ffc107; "
        "border-radius:8px; padding:8px 14px; margin-bottom:10px; "
        "font-size:0.85rem; color:#795548;'>"
        "🎭 <strong>Demo mode</strong> — synthetic classroom image.  "
        "Toggle Demo Mode off and upload a real photo for live analysis."
        "</div>"
        if is_demo else ""
    )

    warning = (
        "<div style='background:#fdecea; border:1px solid #e74c3c; "
        "border-radius:8px; padding:8px 14px; margin-bottom:10px; "
        "font-size:0.85rem; color:#b71c1c;'>"
        f"⚠️ <strong>Low attendance:</strong> "
        f"{s['detected']} of {s['expected']} students detected "
        f"({s['attendance_rate']:.0%}).  Class score penalised."
        "</div>"
        if s["low_attendance"] else ""
    )

    kpis = (
        _kpi(f"{s['class_score_pct']:.0f}%", "Class Pulse",       pulse_colour) +
        _kpi(str(s["detected"]),              "Detected",          "#2980b9") +
        _kpi(f"{s['attendance_rate']:.0%}",   "Attendance",        att_colour) +
        _kpi(str(s["engaged_count"]),         "Engaged",           "#27ae60") +
        _kpi(str(s["neutral_count"]),         "Neutral",           "#f39c12") +
        _kpi(str(s["disengaged_count"]),      "Disengaged",        "#e74c3c")
    )

    badge = (
        f"<span style='background:{pulse_colour}18; color:{pulse_colour}; "
        f"border:1px solid {pulse_colour}; border-radius:20px; "
        f"padding:2px 12px; font-size:0.85rem; font-weight:600;'>"
        f"{s['pulse_label']} Engagement"
        f"</span>"
    )

    privacy = (
        "<p style='color:#999; font-size:0.74rem; margin-top:10px;'>"
        "🔒 Faces blurred before display.  "
        "No images or personal data stored.  "
        "Scores represent <em>aggregate classroom trends only</em> — "
        "no individual is identified."
        "</p>"
    )

    return (
        f"<div style='font-family:sans-serif;'>"
        + demo_banner + warning
        + f"<div style='margin-bottom:8px;'>{badge}</div>"
        + f"<div>{kpis}</div>"
        + privacy
        + "</div>"
    )


def _no_image_html() -> str:
    return (
        "<div style='font-family:sans-serif; color:#888; "
        "padding:24px; text-align:center; font-size:1rem;'>"
        "📷 Upload a classroom image or enable <strong>Demo Mode</strong>, "
        "then click <strong>Analyze</strong>."
        "</div>"
    )


# ---------------------------------------------------------------------------
# Placeholder outputs
# ---------------------------------------------------------------------------

def _empty_rgb() -> np.ndarray:
    img = np.full((400, 600, 3), 240, dtype=np.uint8)
    cv2.putText(img, "No image", (200, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160, 160, 160), 2, cv2.LINE_AA)
    return img


def _empty_fig() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.text(0.5, 0.5, "—", ha="center", va="center",
            fontsize=18, color="#ccc", transform=ax.transAxes)
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_HEADER = """
<div style="font-family:sans-serif; padding:8px 0 4px 0;">
  <h2 style="margin:0; font-size:1.6rem;">🎓 Classroom Engagement Detector</h2>
  <p style="color:#666; margin:4px 0 0 0; font-size:0.9rem;">
    Upload a front-facing classroom photo — the system estimates engagement
    from <strong>body-geometry behavioral proxies</strong> using
    OpenCV only (no cloud API, no face recognition, no data stored).
  </p>
</div>
"""

_HOW_IT_WORKS = """
### Detection pipeline

| Step | Method | What it measures |
|------|--------|-----------------|
| 1 | CLAHE equalisation | Fix mixed classroom lighting |
| 2 | Haar face cascade (frontal + profile) | Per-student anchor |
| 3 | HOG pedestrian detector | Bodies with no visible face |
| 4 | IoU NMS | Remove duplicate detections |
| 5a | **Head pose** | Face aspect ratio + eye-symmetry ratio |
| 5b | **Posture** | Body-box h/w ratio (tall-narrow = upright) |
| 5c | **Hand raise** | Skin-blob in region above face |
| 5d | **Phone use** | Bright rectangle in lap region (−20 % penalty) |
| 5e | **Talking** | Two faces side-by-side at same height |

### Engagement score per student

```
score = 0.30×head_pose + 0.25×posture + 0.25×hand_raise
      + 0.10×talking   − 0.20×phone_detected
```

A baseline attentive student (head forward, upright, not talking,
no hand raised, no phone) scores **0.60 → Engaged**.

### Class pulse

```
class_pulse = Σ(individual scores) / expected_class_size × 100
```

Absent students implicitly score 0, so low attendance penalises
the pulse naturally.

### Privacy

* Face regions are **Gaussian-blurred** before any display.
* All processing is **in-memory** — no images, no personal data retained.
* Results are **aggregate only** — no individual is identified or tracked.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Classroom Engagement Detector") as demo:

        gr.HTML(_HEADER)

        with gr.Row(equal_height=False):
            # ── Left column: controls ─────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                img_input = gr.Image(
                    type="pil",
                    label="📷 Classroom photo",
                    sources=["upload", "clipboard"],
                    height=280,
                )
                expected_slider = gr.Slider(
                    minimum=1, maximum=60, step=1, value=30,
                    label="Expected class size",
                    info="Instructor-supplied enrolment (for attendance rate).",
                )
                demo_toggle = gr.Checkbox(
                    label="🎭 Demo mode (use bundled sample image)",
                    value=False,
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze", variant="primary", size="lg",
                )
                gr.Markdown(
                    "<p style='color:#999; font-size:0.75rem; margin-top:6px;'>"
                    "Best results: front-facing camera, faces ≥ 30 × 30 px, "
                    "students occupy most of the frame."
                    "</p>"
                )

            # ── Right column: results ─────────────────────────────────────
            with gr.Column(scale=2, min_width=420):
                img_output = gr.Image(
                    label="Annotated image (faces blurred)",
                    height=340,
                )
                metrics_html = gr.HTML(value=_no_image_html())

                with gr.Row():
                    gauge_out  = gr.Plot(label="Class pulse", scale=1)
                    signal_out = gr.Plot(label="Signal breakdown", scale=2)

        # ── Wire button ───────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze,
            inputs=[img_input, expected_slider, demo_toggle],
            outputs=[img_output, metrics_html, signal_out, gauge_out],
        )

        # ── How it works ──────────────────────────────────────────────────
        with gr.Accordion("ℹ️ How it works", open=False):
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
