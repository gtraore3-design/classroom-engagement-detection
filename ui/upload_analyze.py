"""
upload_analyze.py — Page 1: Upload & Analyze.

This page handles two modes:

Live mode (demo_mode=False)
    The instructor uploads a JPEG/PNG classroom image.
    The full behavioral-proxy pipeline runs:
      1. Attendance detection (HOG + MediaPipe Face Detection)
      2. Behavioral analysis per person (MediaPipe Face Mesh + Pose)
      3. Weighted engagement scoring (engagement_scorer.py)
      4. Face blurring for privacy (face_blur.py)
      5. Heatmap and annotation overlay (utils/visualization.py)
    Results are stored in st.session_state["results"] for the Dashboard page.

Demo mode (demo_mode=True)
    Pre-computed synthetic results for 20 students in a 4×5 classroom grid
    are loaded from demo/demo_data.py.  The pipeline is bypassed entirely.
    Results are clearly labelled as DEMO DATA throughout the UI.

Privacy guarantees (live mode only)
------------------------------------
- Face blurring runs *before* any image is displayed.
- No face crops, embeddings, or personal data are retained after the page renders.
- A privacy notice is displayed on every upload.
- The disclaimer "no individual is identified or assessed" appears on results.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from pipeline.attendance import estimate_attendance
from pipeline.behavioral_detection import BehavioralDetector
from pipeline.engagement_scorer import classroom_engagement
from pipeline.face_blur import blur_faces, draw_face_boxes
from utils.visualization import draw_engagement_annotations, engagement_heatmap_overlay

# Module-level detector singleton — MediaPipe initialises its neural nets once
_detector = BehavioralDetector()


def _bgr_from_upload(upload) -> np.ndarray:
    """
    Convert a Streamlit UploadedFile to a BGR NumPy array.

    We route through PIL to handle all common image formats (JPEG, PNG,
    WebP, BMP) and then convert to OpenCV's native BGR colour order.
    """
    pil = Image.open(upload).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def render_upload_analyze(demo_mode: bool = False, expected_size: int = 25) -> None:
    """
    Render the Upload & Analyze page.

    Parameters
    ----------
    demo_mode     : If True, show the pre-loaded demo results instead of the
                    upload widget.
    expected_size : Instructor-supplied expected class size (from sidebar).
    """
    st.title("Upload & Analyze")

    # ------------------------------------------------------------------ #
    # Demo mode — show pre-loaded results                                  #
    # ------------------------------------------------------------------ #
    if demo_mode:
        st.info(
            "**Demo Mode is ON.**  "
            "Showing pre-computed synthetic results for a 20-student classroom.  "
            "Turn off Demo Mode in the sidebar to upload your own image.",
            icon="🎭",
        )
        _render_results_preview()
        return

    # ------------------------------------------------------------------ #
    # Privacy notice (live mode only)                                      #
    # ------------------------------------------------------------------ #
    st.info(
        "**Privacy Notice** — All processing is performed in-memory only.  "
        "No faces, images, or personal data are stored, logged, or transmitted.  "
        "Detected faces are blurred in all displayed output images.",
        icon="🔒",
    )
    st.markdown(
        "> **Disclaimer:** This tool reports **aggregate classroom trends only**.  "
        "No individual student is identified, tracked, or assessed."
    )

    # ------------------------------------------------------------------ #
    # Upload widget                                                         #
    # ------------------------------------------------------------------ #
    uploaded = st.file_uploader(
        "Choose a classroom image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help=(
            "Static images only.  For best results use a front-facing classroom "
            "camera at a distance where student faces are at least 30 × 30 pixels."
        ),
    )

    if not _detector.mediapipe_available():
        st.warning(
            "**MediaPipe solutions are unavailable** on this deployment "
            "(the `solutions` API requires a compatible platform/Python version).  "
            "Live analysis is disabled.  Enable **Demo Mode** in the sidebar to "
            "explore all app features with synthetic data.",
            icon="⚠️",
        )
        _render_instructions()
        return

    if uploaded is None:
        _render_instructions()
        return

    # ------------------------------------------------------------------ #
    # Run the live pipeline                                                #
    # ------------------------------------------------------------------ #
    bgr = _bgr_from_upload(uploaded)

    with st.spinner("Running behavioral analysis …"):
        t0 = time.perf_counter()

        # Step 1 — count people in the frame
        attendance_info = estimate_attendance(bgr)

        # Use face bounding boxes as person regions for the Pose analysis.
        # Fall back to the full image if no boxes are detected.
        person_boxes = attendance_info["face_boxes"] or [(0, 0, bgr.shape[1], bgr.shape[0])]

        # Step 2 — extract behavioral signals per person
        person_results = _detector.analyse(bgr, person_boxes=person_boxes)

        # Step 3 — weighted engagement scoring (attendance-adjusted)
        engagement_info = classroom_engagement(person_results, int(expected_size))

        # Step 4 — PRIVACY: blur all detected faces before any display
        blurred = blur_faces(bgr)

        # Step 5 — draw engagement annotations and heatmap overlay on blurred image
        annotated   = draw_engagement_annotations(blurred, person_results, show_score=True)
        heatmap_bgr = engagement_heatmap_overlay(blurred, person_results)

        elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # Store results in session state for the Dashboard page               #
    # ------------------------------------------------------------------ #
    st.session_state["results"] = {
        "engagement_info": engagement_info,
        "person_results":  person_results,
        "attendance_info": attendance_info,
        "expected_size":   int(expected_size),
        "annotated_bgr":   annotated,
        "heatmap_bgr":     heatmap_bgr,
        "blurred_bgr":     blurred,
        "elapsed_s":       elapsed,
        "is_demo":         False,
    }

    # ------------------------------------------------------------------ #
    # Results preview                                                      #
    # ------------------------------------------------------------------ #
    st.success(f"Analysis complete in {elapsed:.2f} s")
    _render_results_preview()


# ---------------------------------------------------------------------------
# Helper: render results preview (shared by live and demo modes)
# ---------------------------------------------------------------------------

def _render_results_preview() -> None:
    """Display the quick-results KPI row and image tabs."""
    if "results" not in st.session_state:
        return

    data = st.session_state["results"]
    ei   = data["engagement_info"]
    ai   = data["attendance_info"]
    is_demo = data.get("is_demo", False)

    if is_demo:
        st.warning("⚠️ The values below are **synthetic demo data** and do not "
                   "represent a real classroom.", icon="🎭")

    # KPI row
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Classroom Score",   f"{ei['classroom_score']:.0%}")
    col_b.metric("Detected students", ai["best_estimate"])
    col_c.metric("Expected",          ei["expected_size"])
    col_d.metric("Engaged",           f"{ei['engaged_count']} / {ei['detected_count']}")

    # Attendance warning
    if ei["low_attendance_warning"]:
        st.warning(
            f"⚠️ Low attendance: {ai['best_estimate']} of {ei['expected_size']} "
            f"students detected ({ei['attendance_rate']:.0%}).  "
            "The classroom engagement score is penalised accordingly.  "
            "**Do not interpret engagement figures without noting this confound.**"
        )

    # Image tabs
    tab1, tab2, tab3 = st.tabs(["Annotated", "Engagement heatmap", "Blurred original"])
    with tab1:
        st.image(
            cv2.cvtColor(data["annotated_bgr"], cv2.COLOR_BGR2RGB),
            caption="Per-person engagement score (coloured circles).  Faces blurred.",
            use_container_width=True,
        )
    with tab2:
        st.image(
            cv2.cvtColor(data["heatmap_bgr"], cv2.COLOR_BGR2RGB),
            caption="Spatial heatmap: green = engaged, red = disengaged.  Faces blurred.",
            use_container_width=True,
        )
    with tab3:
        st.image(
            cv2.cvtColor(data["blurred_bgr"], cv2.COLOR_BGR2RGB),
            caption="Original image with all detected faces blurred.",
            use_container_width=True,
        )

    st.info("Navigate to **Engagement Dashboard** in the sidebar for detailed metrics and charts.")


# ---------------------------------------------------------------------------
# Helper: instructions panel shown before upload
# ---------------------------------------------------------------------------

def _render_instructions() -> None:
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        """
The system uses **behavioral proxies** — not facial emotion recognition — to
detect classroom engagement.  Each proxy is detected with MediaPipe and
weighted in the engagement score:

| Signal | Detection method | Weight |
|--------|-----------------|--------|
| **Hand raised** | MediaPipe Pose — wrist above shoulder landmark | **30 %** |
| **Head forward** | MediaPipe Face Mesh — pitch/yaw via solvePnP | **25 %** |
| **Gaze forward** | Face Mesh refined iris landmarks — offset ratio | **20 %** |
| **Good posture** | Pose — shoulder-to-hip spine angle | **15 %** |
| **Phone absent** | Pose — wrist-near-hip proximity heuristic | **10 %** |

The final **classroom score** is attendance-adjusted:
```
classroom_score = avg_behavioral_score × (detected_students / expected_class_size)
```

Want to test without uploading? Enable **Demo Mode** in the sidebar.
"""
    )
