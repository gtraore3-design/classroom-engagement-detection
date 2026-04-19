"""
Page 2 — Engagement Dashboard
Detailed charts and metrics from the most recent analysis run.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from config import (
    ENGAGED_THRESHOLD,
    ENGAGEMENT_WEIGHTS,
    LOW_ATTENDANCE_THRESHOLD,
    SCB_BEHAVIOR_CLASSES,
    SCB_ENGAGEMENT_SCORES,
)
from pipeline.engagement_scorer import score_person


# ---------------------------------------------------------------------------
# Gauge chart (Plotly)
# ---------------------------------------------------------------------------

def _gauge_chart(value: float, title: str = "Classroom Engagement") -> go.Figure:
    color = (
        "red"    if value < 0.35 else
        "orange" if value < 0.55 else
        "gold"   if value < 0.70 else
        "green"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  35], "color": "#ffcccc"},
                {"range": [35, 55], "color": "#ffe0b2"},
                {"range": [55, 70], "color": "#fff9c4"},
                {"range": [70, 100], "color": "#c8e6c9"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "value": value * 100},
        },
        delta={"reference": 70, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=0, l=20, r=20))
    return fig


# ---------------------------------------------------------------------------
# Proxy breakdown bar chart
# ---------------------------------------------------------------------------

def _proxy_bar(proxy_breakdown: dict) -> plt.Figure:
    labels = {
        "hand_raised":  "Hand Raised",
        "head_forward": "Head Forward",
        "gaze_forward": "Gaze Forward",
        "good_posture": "Good Posture",
        "phone_absent": "Phone Absent",
    }
    keys   = list(labels.keys())
    values = [proxy_breakdown.get(k, 0.0) for k in keys]
    colors = ["#4caf50" if v >= 0.6 else "#ff9800" if v >= 0.4 else "#f44336" for v in values]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh([labels[k] for k in keys], [v * 100 for v in values],
                   color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Average score (%)")
    ax.set_title("Behavioral Proxy Breakdown")
    ax.axvline(x=60, color="grey", linestyle="--", linewidth=0.8, label="60 % threshold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-person score strip
# ---------------------------------------------------------------------------

def _scb_histogram(scb_dist: dict[str, int]) -> plt.Figure:
    """Bar chart of SCB class distribution across detected students."""
    all_classes = SCB_BEHAVIOR_CLASSES
    counts = [scb_dist.get(c, 0) for c in all_classes]

    colors = []
    for c in all_classes:
        score = SCB_ENGAGEMENT_SCORES[c]
        if score >= 0.70:
            colors.append("#4caf50")
        elif score >= 0.40:
            colors.append("#ff9800")
        else:
            colors.append("#f44336")

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(all_classes, counts, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("# students")
    ax.set_title("SCB Behavior Class Distribution (CNN predictions)")
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha="center", va="bottom", fontsize=9)
    ax.set_xticklabels(all_classes, rotation=20, ha="right")
    fig.tight_layout()
    return fig


def _person_score_df(person_results, per_person_scores) -> pd.DataFrame:
    rows = []
    for res, score in zip(person_results, per_person_scores):
        rows.append({
            "Person #":        res.person_id + 1,
            "Score":           f"{score:.0%}",
            "Engaged":         "✓" if score >= ENGAGED_THRESHOLD else "✗",
            "SCB class":       res.scb_class if res.scb_class else "—",
            "Hand raised":     f"{res.hand_raised_score:.0%}",
            "Head forward":    f"{res.head_forward_score:.0%}",
            "Gaze forward":    f"{res.gaze_forward_score:.0%}",
            "Good posture":    f"{res.good_posture_score:.0%}",
            "Phone absent":    f"{res.phone_absent_score:.0%}",
            "Head pitch (°)":  f"{res.head_pitch_deg:.1f}",
            "Head yaw (°)":    f"{res.head_yaw_deg:.1f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_dashboard():
    st.title("Engagement Dashboard")

    if "results" not in st.session_state:
        st.info("No analysis results yet. Please upload an image on the **Upload & Analyze** page first.")
        return

    data = st.session_state["results"]
    ei   = data["engagement_info"]
    ai   = data["attendance_info"]
    prs  = data["person_results"]

    # ------------------------------------------------------------------ #
    # Warning banners                                                      #
    # ------------------------------------------------------------------ #
    if ei["low_attendance_warning"]:
        st.error(
            f"**Low attendance:** {ai['best_estimate']} detected vs. "
            f"{ei['expected_size']} expected "
            f"({ei['attendance_rate']:.0%}).  "
            "Engagement metrics are computed relative to the expected class size — "
            "the classroom score is penalised accordingly.  "
            "**Do not interpret engagement scores without considering this attendance shortfall.**"
        )

    if ei["detected_count"] == 0:
        st.warning("No people detected in the image. Please try a clearer classroom image.")
        return

    # ------------------------------------------------------------------ #
    # Top KPIs                                                             #
    # ------------------------------------------------------------------ #
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classroom Score",  f"{ei['classroom_score']:.0%}")
    c2.metric("Students Detected", ei["detected_count"])
    c3.metric("Engaged",          f"{ei['engaged_count']} / {ei['detected_count']}")
    c4.metric("Attendance Rate",  f"{ei['attendance_rate']:.0%}")

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Gauge                                                                #
    # ------------------------------------------------------------------ #
    col_g, col_b = st.columns([1, 1.5])
    with col_g:
        st.plotly_chart(_gauge_chart(ei["classroom_score"]), use_container_width=True)
    with col_b:
        st.markdown("#### Attendance vs. Expected")
        att_df = pd.DataFrame({
            "Category": ["Detected", "Expected"],
            "Count":    [ei["detected_count"], ei["expected_size"]],
        })
        st.bar_chart(att_df.set_index("Category"))

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Proxy breakdown                                                      #
    # ------------------------------------------------------------------ #
    st.subheader("Behavioral Proxy Scores")
    st.caption(
        "Each bar shows the class-average score for one behavioral proxy.  "
        "Scores ≥ 60 % are green, 40–60 % orange, < 40 % red."
    )
    fig_bar = _proxy_bar(ei["proxy_breakdown"])
    st.pyplot(fig_bar, use_container_width=True)

    # Weighted contribution table
    with st.expander("Show weighted contribution breakdown"):
        rows = []
        for key, label in {
            "hand_raised":  "Hand Raised",
            "head_forward": "Head Forward",
            "gaze_forward": "Gaze Forward",
            "good_posture": "Good Posture",
            "phone_absent": "Phone Absent",
        }.items():
            avg_score = ei["proxy_breakdown"][key]
            weight    = ENGAGEMENT_WEIGHTS[key]
            rows.append({
                "Proxy":              label,
                "Weight":             f"{weight:.0%}",
                "Class avg score":    f"{avg_score:.0%}",
                "Weighted contribution": f"{avg_score * weight:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------ #
    # SCB class distribution (shown only when SCB CNN was active)         #
    # ------------------------------------------------------------------ #
    scb_dist   = ei.get("scb_class_distribution", {})
    scb_active = ei.get("scb_model_active", False)
    if scb_active and scb_dist:
        st.subheader("SCB Behavior Class Distribution")
        st.caption(
            "Distribution of CNN-predicted behavior classes across all detected students.  "
            "Green = high engagement class, Orange = moderate, Red = low engagement class."
        )
        fig_scb = _scb_histogram(scb_dist)
        st.pyplot(fig_scb, use_container_width=True)
    elif not scb_active:
        st.info(
            "SCB behavior model not loaded — running in MediaPipe-only mode.  "
            "Train the model (`python3 -m model.train`) to enable CNN class predictions.",
            icon="ℹ️",
        )

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Per-person table                                                     #
    # ------------------------------------------------------------------ #
    st.subheader("Per-Person Results")
    st.caption(
        "Engagement scores per detected person.  "
        "Person IDs are arbitrary and do not identify individuals."
    )
    df = _person_score_df(prs, ei["per_person_scores"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------ #
    # Heatmap image                                                        #
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.subheader("Spatial Engagement Heatmap")
    import cv2
    st.image(
        cv2.cvtColor(data["heatmap_bgr"], cv2.COLOR_BGR2RGB),
        caption="Green zones = high engagement, Red zones = low engagement.  Faces blurred.",
        use_container_width=True,
    )

    st.markdown("---")
    st.caption(
        "Analysis completed in "
        f"{data.get('elapsed_s', 0):.1f}s.  "
        "All processing performed in-memory; no data retained after session ends."
    )
