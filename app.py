"""
app.py — Classroom Engagement Detection System (CED)
CIS 515 Project · 2026

Entry point.  Run with:
    streamlit run app.py

Architecture
------------
The app is organised as a single-file entry point that delegates to four
page-rendering modules under ``ui/``.  A sidebar radio widget provides
navigation between pages.

Demo Mode
---------
A "Demo Mode" toggle in the sidebar bypasses the live MediaPipe pipeline and
instead loads pre-computed synthetic results (20 students, 4-row × 5-column
classroom grid).  This allows graders to test all UI components without
providing a real classroom photograph.  Demo results are labelled clearly.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so sibling packages resolve correctly
# when the app is launched from any working directory.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# Page config must be the *first* Streamlit call
st.set_page_config(
    page_title="Classroom Engagement Detector",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lazy-import page renderers (imports MediaPipe / TF, so we defer them)
from ui.upload_analyze   import render_upload_analyze
from ui.dashboard        import render_dashboard
from ui.model_evaluation import render_model_evaluation
from ui.about            import render_about


# ---------------------------------------------------------------------------
# Sidebar: navigation + demo mode toggle
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[str, bool, int]:
    """
    Render the sidebar and return (selected_page, demo_mode_enabled, expected_size).
    """
    st.sidebar.title("🎓 CED System")
    st.sidebar.markdown("**Classroom Engagement Detector**")
    st.sidebar.markdown("---")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "📤  Upload & Analyze",
            "📊  Engagement Dashboard",
            "🧪  Model Evaluation",
            "📖  About / Methodology",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")

    demo_mode = st.sidebar.toggle(
        "Demo Mode",
        value=False,
        help=(
            "Enable to load a synthetic 20-student classroom result "
            "without uploading a real image.  "
            "Useful for graders and presentations."
        ),
    )

    # Expected class size is used both in live mode (upload page) and demo mode
    expected_size = int(st.sidebar.number_input(
        "Expected class size",
        min_value=1, max_value=200, value=25,
        help="Used to compute the attendance-adjusted classroom engagement score.",
    ))

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Privacy:** All image processing is performed in memory only.  "
        "No data is stored, logged, or transmitted."
    )
    st.sidebar.caption("CIS 515 · Classroom Engagement Detection · 2026")

    return page, demo_mode, expected_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    page, demo_mode, expected_size = _sidebar()

    # ---- Demo mode: pre-load synthetic results into session state ----
    if demo_mode:
        # Only regenerate if not already cached for this expected_size
        cached = st.session_state.get("demo_results_cached_size")
        if cached != expected_size or "results" not in st.session_state:
            with st.spinner("Loading demo data …"):
                from demo import get_demo_results
                st.session_state["results"] = get_demo_results(expected_size)
                st.session_state["demo_results_cached_size"] = expected_size

    # ---- Route to selected page ----
    if "Upload" in page:
        render_upload_analyze(demo_mode=demo_mode, expected_size=expected_size)
    elif "Dashboard" in page:
        render_dashboard()
    elif "Model" in page:
        render_model_evaluation()
    elif "About" in page:
        render_about()


if __name__ == "__main__":
    main()
