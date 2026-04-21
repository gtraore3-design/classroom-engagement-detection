"""
gradio_app.py — Classroom Engagement Analyzer (Gradio POC)
============================================================

A self-contained Gradio proof-of-concept for analyzing classroom engagement
from a single static photograph.  No TensorFlow, no MediaPipe required —
only OpenCV, NumPy, Pillow, and Matplotlib.

Detection pipeline
------------------
1. CLAHE histogram equalisation  (normalise mixed classroom lighting)
2. Multi-scale Haar frontal + profile face detection
3. HOG pedestrian detector       (bodies without visible faces)
4. IoU NMS                       (remove duplicate detections)
5. Eye detection within face ROI → head-pose proxy → engagement label
6. Attendance-adjusted classroom score

Engagement labels
-----------------
  engaged      2 symmetric eyes detected (head facing forward)
  neutral      1 eye or asymmetric eyes (head tilted / profile)
  disengaged   0 eyes detected (head bowed)
  no_face      Person found by HOG but no face detected

Usage
-----
    python gradio_app.py          # starts on http://localhost:7860
    python gradio_app.py --share  # creates a public Gradio link

Dependencies
------------
    gradio>=4.0.0
    opencv-python-headless>=4.9.0
    numpy>=1.24.0
    pillow>=10.0.0
    matplotlib>=3.7.0
"""

from __future__ import annotations

import argparse
import sys

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Pipeline modules (same directory tree)
from pipeline.detector import detect_students
from pipeline.scorer   import annotate_image, compute_scores, make_bar_chart


# ---------------------------------------------------------------------------
# Core analysis function — called by Gradio on every button click
# ---------------------------------------------------------------------------

def analyze(
    pil_image: Image.Image | None,
    expected_size: int,
) -> tuple[np.ndarray, str, plt.Figure]:
    """
    Run the full engagement analysis pipeline on a PIL classroom image.

    Parameters
    ----------
    pil_image     : Image uploaded by the user (RGB PIL Image), or None.
    expected_size : Instructor-supplied expected number of students.

    Returns
    -------
    annotated_rgb : np.ndarray  — annotated image with blurred faces + boxes
    metrics_html  : str         — HTML KPI summary for gr.HTML
    bar_chart_fig : plt.Figure  — engagement breakdown bar chart
    """
    if pil_image is None:
        placeholder = _placeholder_image()
        return placeholder, _no_image_html(), _empty_chart()

    # Convert PIL (RGB) → OpenCV (BGR)
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # --- Run pipeline ---
    detections = detect_students(bgr)
    scores     = compute_scores(detections, expected_size)
    annotated  = annotate_image(bgr, detections)

    # Convert back to RGB for Gradio display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    metrics_html = _build_metrics_html(scores)
    bar_fig      = make_bar_chart(scores)

    return annotated_rgb, metrics_html, bar_fig


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_SCORE_COLOUR = {
    "High":     "#39c549",
    "Moderate": "#ffc31d",
    "Low":      "#e93c3c",
}

_CARD_STYLE = (
    "background:#161b22; border:1px solid #30363d; border-radius:10px; "
    "padding:14px 18px; text-align:center; flex:1; min-width:120px;"
)
_VAL_STYLE  = "font-size:2rem; font-weight:700; margin:0; line-height:1.1;"
_LBL_STYLE  = "font-size:0.78rem; color:#8b949e; margin:4px 0 0 0;"


def _kpi(value: str, label: str, colour: str = "#c9d1d9") -> str:
    return (
        f"<div style='{_CARD_STYLE}'>"
        f"<p style='{_VAL_STYLE} color:{colour};'>{value}</p>"
        f"<p style='{_LBL_STYLE}'>{label}</p>"
        f"</div>"
    )


def _build_metrics_html(s: dict) -> str:
    """Build the HTML KPI panel from a scores dict."""
    score_colour = _SCORE_COLOUR.get(s["label"], "#c9d1d9")
    att_colour   = "#e93c3c" if s["low_attendance_warning"] else "#58a6ff"

    warning_block = ""
    if s["low_attendance_warning"]:
        warning_block = (
            "<div style='background:rgba(233,69,96,0.10); border:1px solid #e94560; "
            "border-radius:8px; padding:10px 14px; margin-bottom:12px; font-size:0.85rem;'>"
            f"⚠️ <strong style='color:#e94560;'>Low attendance:</strong> "
            f"{s['detected_count']} of {s['expected_size']} students detected "
            f"({s['attendance_rate']:.0%}).  Classroom score is penalised accordingly."
            "</div>"
        )

    kpis = "".join([
        _kpi(f"{s['classroom_score_pct']:.0f}%",  "Classroom Score",      score_colour),
        _kpi(str(s["detected_count"]),              "Detected",             "#58a6ff"),
        _kpi(f"{s['attendance_rate']:.0%}",         "Attendance",           att_colour),
        _kpi(str(s["engaged_count"]),               "Engaged",              "#39c549"),
        _kpi(str(s["neutral_count"]),               "Neutral",              "#ffc31d"),
        _kpi(str(s["disengaged_count"]),            "Disengaged",           "#e93c3c"),
    ])

    label_badge = (
        f"<span style='background:{score_colour}22; color:{score_colour}; "
        f"border:1px solid {score_colour}; border-radius:20px; "
        f"padding:2px 12px; font-size:0.85rem; font-weight:600;'>"
        f"{s['label']} engagement"
        f"</span>"
    )

    return (
        f"<div style='font-family:sans-serif; color:#c9d1d9;'>"
        f"<div style='margin-bottom:10px;'>{label_badge}</div>"
        + warning_block +
        f"<div style='display:flex; gap:10px; flex-wrap:wrap;'>{kpis}</div>"
        f"<p style='color:#8b949e; font-size:0.75rem; margin-top:10px;'>"
        f"🔒 Detected faces are blurred before display.  "
        f"No images or personal data are stored. "
        f"Scores represent <em>aggregate classroom trends only</em>."
        f"</p>"
        f"</div>"
    )


def _no_image_html() -> str:
    return (
        "<div style='font-family:sans-serif; color:#8b949e; padding:20px; text-align:center;'>"
        "📷 Upload a classroom image and click <strong>Analyze</strong>."
        "</div>"
    )


# ---------------------------------------------------------------------------
# Placeholder / empty helpers
# ---------------------------------------------------------------------------

def _placeholder_image() -> np.ndarray:
    """512 × 512 dark placeholder with centred text."""
    img = np.full((512, 512, 3), 22, dtype=np.uint8)
    cv2.putText(img, "Upload an image", (90, 256),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2, cv2.LINE_AA)
    return img


def _empty_chart() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
            color="#8b949e", transform=ax.transAxes, fontsize=12)
    ax.axis("off")
    return fig


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CSS = """
body, .gradio-container { background:#0d1117 !important; }
.gr-button-primary { background:#238636 !important; border-color:#2ea043 !important; }
footer { display:none !important; }
"""

_DESCRIPTION = """
<div style="font-family:sans-serif; color:#c9d1d9; max-width:720px; margin:0 auto;">
  <h2 style="margin-bottom:4px;">🎓 Classroom Engagement Analyzer</h2>
  <p style="color:#8b949e; margin-top:0;">
    Upload a front-facing classroom photograph.  The system detects students using
    <strong>OpenCV HOG + Haar cascades</strong> and estimates engagement via
    <strong>head-pose proxies</strong> (eye-symmetry ratio) — no facial emotion
    recognition, no cloud API, no data retained.
  </p>
  <p style="color:#8b949e; font-size:0.85rem;">
    🟢 <strong>Engaged</strong> — two symmetric eyes detected (head forward) &nbsp;|&nbsp;
    🟡 <strong>Neutral</strong> — head tilted / profile view &nbsp;|&nbsp;
    🔴 <strong>Disengaged</strong> — no eyes visible (head bowed) &nbsp;|&nbsp;
    ⚫ <strong>No face</strong> — body found, face not detected
  </p>
</div>
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(css=_CSS, title="Classroom Engagement Analyzer") as demo:

        gr.HTML(_DESCRIPTION)

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                image_input = gr.Image(
                    type="pil",
                    label="📷 Classroom Image",
                    sources=["upload", "clipboard"],
                    height=320,
                )
                expected_slider = gr.Slider(
                    minimum=1, maximum=60, step=1, value=25,
                    label="Expected class size",
                    info="Instructor-supplied enrolment (used for attendance rate).",
                )
                analyze_btn = gr.Button("🔍 Analyze", variant="primary")

                gr.Markdown(
                    "<p style='color:#8b949e; font-size:0.75rem; margin-top:8px;'>"
                    "Best results: front-facing camera, students occupy ≥⅔ of frame, "
                    "faces ≥ 30 × 30 px."
                    "</p>"
                )

            # ── Right column: outputs ────────────────────────────────────
            with gr.Column(scale=2, min_width=400):
                annotated_output = gr.Image(
                    label="Annotated image (faces blurred)",
                    height=380,
                )
                metrics_output = gr.HTML(value=_no_image_html())
                chart_output   = gr.Plot(label="Engagement breakdown")

        # ── Wire up the button ───────────────────────────────────────────
        analyze_btn.click(
            fn=analyze,
            inputs=[image_input, expected_slider],
            outputs=[annotated_output, metrics_output, chart_output],
        )

        # ── How it works accordion ───────────────────────────────────────
        with gr.Accordion("ℹ️ How it works", open=False):
            gr.Markdown(
                """
### Detection pipeline

| Step | Method | Purpose |
|------|--------|---------|
| 1 | CLAHE equalisation | Normalise mixed classroom lighting (windows + fluorescent) |
| 2 | Haar frontal + profile cascade | Detect faces at multiple scales |
| 3 | HOG pedestrian detector | Find bodies when faces are not visible |
| 4 | IoU NMS | Remove duplicate detections |
| 5 | Haar eye detector (in face ROI) | Count eyes in upper-65 % of each face |
| 6 | Head-pose proxy | 2 symmetric eyes → engaged; 1 eye → neutral; 0 → disengaged |

### Engagement → classroom score

```
classroom_score = avg_engagement_value × (detected / expected)
```

This ensures a high engagement rate in a half-empty room does not hide poor attendance.

### Privacy

* Detected face regions are **Gaussian-blurred** before any display.
* All processing is **in-memory only** — no images or personal data are stored.
* Results represent **aggregate classroom trends**; no individual is identified.

### References

* Viola & Jones (2001) — Rapid object detection using a boosted cascade of simple features.
* Dalal & Triggs (2005) — Histograms of oriented gradients for human detection.
* Raca, Tormey & Dillenbourg (2015) — Sleepers' lag: motion and attention in the classroom.
"""
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classroom Engagement Analyzer (Gradio)")
    parser.add_argument("--share",  action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port",   type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--host",   type=str, default="127.0.0.1", help="Bind host")
    args = parser.parse_args()

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
