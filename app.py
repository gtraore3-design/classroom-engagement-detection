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
# Global CSS injection
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
<style>
/* ── Base & background ───────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid #0f3460;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 6px 10px;
    border-radius: 6px;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(15, 52, 96, 0.6);
}

/* ── Hero banner ─────────────────────────────────────────────────────── */
.ced-hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 60%, #00b4d8 100%);
    border-radius: 14px;
    padding: 36px 40px 32px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
    border: 1px solid #0f3460;
}
.ced-hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.ced-hero p {
    font-size: 1.05rem;
    color: #a0c4e8;
    margin: 0;
}
.ced-hero .badge {
    display: inline-block;
    background: rgba(233, 69, 96, 0.25);
    color: #e94560;
    border: 1px solid #e94560;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-top: 12px;
}

/* ── KPI / stat cards ────────────────────────────────────────────────── */
.kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.35);
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #0f3460; }
.kpi-card .kpi-value {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-card .kpi-label {
    font-size: 0.82rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}
.kpi-engaged  { color: #00c49a; }
.kpi-neutral  { color: #ffd700; }
.kpi-warning  { color: #e94560; }
.kpi-blue     { color: #58a6ff; }

/* ── Section headers ─────────────────────────────────────────────────── */
h2, h3 {
    color: #e6edf3;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
}

/* ── Tables / dataframes ─────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #21262d;
}

/* ── Upload zone ─────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed #0f3460 !important;
    border-radius: 12px !important;
    background: #161b22 !important;
    padding: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00b4d8 !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"] {
    color: #8b949e;
    font-weight: 600;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #58a6ff;
    border-bottom-color: #58a6ff;
}

/* ── Alerts / info boxes ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ── Expanders ───────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
}

/* ── Metric widget ───────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] { color: #58a6ff; font-weight: 800; }
</style>
""",
        unsafe_allow_html=True,
    )


def _hero_banner() -> None:
    st.markdown(
        """
<div class="ced-hero">
  <h1>🎓 Classroom Engagement Detector</h1>
  <p>Behavioral-proxy analysis · privacy-first · attendance-adjusted scoring</p>
  <span class="badge">CIS 515 · 2026</span>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: navigation + demo mode toggle
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[str, bool, int]:
    """
    Render the sidebar and return (selected_page, demo_mode_enabled, expected_size).
    """
    # Sidebar logo / brand block
    st.sidebar.markdown(
        """
<div style="text-align:center; padding: 16px 0 8px 0;">
  <div style="font-size:2.8rem; margin-bottom:4px;">🎓</div>
  <div style="font-size:1.1rem; font-weight:800; color:#e6edf3; letter-spacing:-0.3px;">
    CED System
  </div>
  <div style="font-size:0.75rem; color:#8b949e; margin-top:2px;">
    Classroom Engagement Detection
  </div>
</div>
<hr style="border-color:#0f3460; margin: 10px 0 14px 0;">
""",
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "📤  Upload & Analyze",
            "📊  Engagement Dashboard",
            "🧪  Model Evaluation",
            "📖  About / Methodology",
        ],
        index=0,
    )

    st.sidebar.markdown(
        "<hr style='border-color:#0f3460; margin: 14px 0;'>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**⚙️ Settings**")

    demo_mode = st.sidebar.toggle(
        "Demo Mode",
        value=True,
        help=(
            "Enable to load a synthetic 20-student classroom result "
            "without uploading a real image.  "
            "Useful for graders and presentations."
        ),
    )

    if demo_mode:
        st.sidebar.markdown(
            "<div style='background:rgba(0,196,154,0.12); border:1px solid #00c49a; "
            "border-radius:8px; padding:8px 10px; font-size:0.78rem; color:#00c49a; margin-top:4px;'>"
            "🟢 Demo data active — no upload required"
            "</div>",
            unsafe_allow_html=True,
        )

    # Expected class size is used both in live mode (upload page) and demo mode
    expected_size = int(st.sidebar.number_input(
        "Expected class size",
        min_value=1, max_value=200, value=25,
        help="Used to compute the attendance-adjusted classroom engagement score.",
    ))

    st.sidebar.markdown(
        "<hr style='border-color:#0f3460; margin: 14px 0;'>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div style='font-size:0.75rem; color:#8b949e; line-height:1.5;'>"
        "🔒 <b>Privacy:</b> All processing is in-memory only. "
        "No data is stored or transmitted."
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div style='font-size:0.72rem; color:#6e7681; margin-top:8px;'>"
        "CIS 515 · 2026"
        "</div>",
        unsafe_allow_html=True,
    )

    return page, demo_mode, expected_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _inject_css()
    page, demo_mode, expected_size = _sidebar()
    _hero_banner()

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
