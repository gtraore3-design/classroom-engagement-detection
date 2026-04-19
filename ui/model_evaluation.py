"""
model_evaluation.py — Page 3: Model Evaluation.

Primary section: SCB-Dataset behavior classification (6 classes)
    MobileNetV2 fine-tuned to predict:
    hand_raising, paying_attention, writing, distracted, bored, phone_use

Secondary / baseline section: FER2013 emotion classification (7 classes)
    Kept for academic comparison only; emotion predictions are NEVER used
    as the primary engagement signal.

The SCB model is the primary signal because:
  1. It is trained on real classroom behavior images (domain match).
  2. Its output classes map directly to engagement proxies.
  3. No ambiguous emotion-to-engagement conversion is required.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from config import (
    SCB_BEHAVIOR_CLASSES,
    EMOTION_CLASSES,
    AMBIGUOUS_EMOTION_CLASSES,
    SCB_ENGAGEMENT_SCORES,
)
from model.evaluate import SCB_DEMO_METRICS, FER_DEMO_METRICS

# load_scb_model / load_fer_model are imported lazily inside render_model_evaluation()
# so that TensorFlow is never required at app startup.

# Optional DeepFace (graceful degradation)
try:
    from deepface import DeepFace as _DeepFace
    _DEEPFACE_AVAILABLE = True
except ImportError:
    _DEEPFACE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared chart helpers
# ---------------------------------------------------------------------------

def _cm_fig(cm: np.ndarray, classes: list[str], title: str) -> plt.Figure:
    """Row-normalised confusion matrix heatmap (raw counts annotated)."""
    fig, ax = plt.subplots(figsize=(max(7, len(classes)), max(5, len(classes)-1)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(cm_norm, annot=cm, fmt="d",
                xticklabels=classes, yticklabels=classes,
                cmap="Blues", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Recall (row-normalised)"})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title,        fontsize=12)
    fig.tight_layout()
    return fig


def _per_class_bar(per_class: dict, classes: list[str],
                   ambiguous: set | None = None) -> plt.Figure:
    """Grouped precision / recall / F1 bar chart."""
    prec = [per_class[c]["precision"] for c in classes]
    rec  = [per_class[c]["recall"]    for c in classes]
    f1   = [per_class[c]["f1"]        for c in classes]
    x    = np.arange(len(classes))
    w    = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(classes)*1.5), 4))
    ax.bar(x-w, prec, w, label="Precision", color="#5c85d6")
    ax.bar(x,   rec,  w, label="Recall",    color="#4caf50")
    ax.bar(x+w, f1,   w, label="F1",        color="#ff9800")

    if ambiguous:
        for i, c in enumerate(classes):
            if c in ambiguous:
                ax.axvline(x=i, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.axhline(0.70, color="grey", linestyle=":", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_model_evaluation() -> None:
    st.title("Model Evaluation")

    # TensorFlow is optional — not available on Python 3.14 / Streamlit Cloud.
    # Everything except live model loading works without it.
    try:
        from model.behavior_model import load_scb_model
        from model.fer_model import load_fer_model
        _tf_available = True
    except ImportError:
        _tf_available = False
        load_scb_model = lambda: None  # noqa: E731
        load_fer_model = lambda: None  # noqa: E731

    if not _tf_available:
        st.info(
            "**TensorFlow is not installed** on this deployment.  "
            "Reference metrics and charts are shown below.  "
            "Live model inference requires TensorFlow — install it locally to "
            "train and evaluate a custom checkpoint.",
            icon="ℹ️",
        )

    tab_scb, tab_fer = st.tabs([
        "📊  SCB-Dataset (Primary)",
        "📉  FER2013 (Baseline — Secondary Signal Only)",
    ])

    # ======================================================================
    # Tab 1: SCB-Dataset — PRIMARY model
    # ======================================================================
    with tab_scb:
        st.subheader("SCB-Dataset Behavior Classification")
        st.markdown(
            "**Primary dataset:** Mohanta, A. et al. (2023). "
            "*SCB-Dataset: A Dataset for Detecting Student Classroom Behavior.* "
            "[arXiv:2304.02488](https://arxiv.org/abs/2304.02488) | "
            "[Kaggle](https://www.kaggle.com/datasets/asthalochanmohanta/"
            "class-room-student-behaviour/data)"
        )

        scb_model  = load_scb_model()
        scb_metrics = SCB_DEMO_METRICS    # default to reference

        if scb_model is not None:
            st.success(
                "SCB model checkpoint found (`model/scb_mobilenetv2.keras`).  "
                "Showing reference metrics — supply test data to recompute."
            )
        else:
            st.info(
                "SCB model not yet trained.  "
                "Showing **reference metrics** approximating expected MobileNetV2 "
                "performance on the SCB-Dataset.  "
                "Train with: `python3 -m model.train --data_dir /path/to/scb`  "
                "Download data: `python3 setup.py --download-scb`"
            )

        if scb_metrics.get("note"):
            st.caption(f"ℹ️ {scb_metrics['note']}")

        # ---- KPI ----
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{scb_metrics['accuracy']:.1%}")
        c2.metric("Macro F1", f"{scb_metrics['macro_f1']:.1%}")

        # ---- Engagement score mapping table ----
        with st.expander("SCB class → engagement score mapping"):
            rows = []
            for cls in SCB_BEHAVIOR_CLASSES:
                m = scb_metrics["per_class"].get(cls, {})
                rows.append({
                    "Behavior class":     cls,
                    "Engagement score":   f"{SCB_ENGAGEMENT_SCORES[cls]:.0%}",
                    "F1":                 f"{m.get('f1', 0):.2f}",
                    "Precision":          f"{m.get('precision', 0):.2f}",
                    "Recall":             f"{m.get('recall', 0):.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "Engagement scores are derived from educational research literature "
                "(Raca et al. 2015; Dewan et al. 2019) and validated against the "
                "SCB class semantics.  Writing receives a moderate score (0.70) "
                "because it indicates on-task behaviour despite the head-bowed posture."
            )

        # ---- Per-class bar chart ----
        st.subheader("Per-Class Metrics (SCB)")
        fig_bar = _per_class_bar(scb_metrics["per_class"], SCB_BEHAVIOR_CLASSES)
        st.pyplot(fig_bar, use_container_width=True)

        # ---- Confusion matrix ----
        st.subheader("Confusion Matrix (SCB)")
        fig_cm = _cm_fig(
            scb_metrics["confusion_matrix"],
            SCB_BEHAVIOR_CLASSES,
            "SCB-Dataset — MobileNetV2 (Reference)",
        )
        st.pyplot(fig_cm, use_container_width=True)

        with st.expander("Interpreting the SCB confusion matrix"):
            st.markdown(
                """
- **paying_attention ↔ writing**: the most frequent confusion pair.
  Both involve the student seated upright with their head slightly bowed —
  the difference is whether a pen is visible, which a 96×96 crop at typical
  camera-to-student distances may not resolve.
- **distracted ↔ bored**: both are passive low-engagement states; the main
  distinguishing cue is gaze direction (away vs. downward), which can be
  subtle at classroom distances.
- **hand_raising** has the highest F1 because the raised-arm silhouette is
  visually distinctive even in low-resolution crops.
- **phone_use** is well-detected due to the characteristic two-hand-together
  posture, consistent with MediaPipe heuristics reinforcing the CNN.
"""
            )

        # ---- Why SCB beats FER2013 for this task ----
        st.markdown("---")
        st.subheader("Why SCB-Dataset is Superior to FER2013 for Engagement Detection")
        st.markdown(
            """
| Criterion | SCB-Dataset | FER2013 |
|-----------|-------------|---------|
| **Domain** | Real classroom images | Internet-sourced face crops |
| **Output classes** | Direct behavior labels | Abstract emotion labels |
| **Engagement mapping** | Direct (1:1 class → score) | Indirect (requires emotion→engagement conversion) |
| **Body context** | Included (arm, posture visible) | Face only |
| **Accuracy** | ~70 % (behavior, hard task) | ~67 % (emotion) |
| **Ambiguity** | Low — classes are observable behaviors | High — Neutral/Sad/Surprise are ambiguous proxies |
| **Bias** | Real classroom diversity | Internet image bias (skin tone, age) |
"""
        )

        # ---- Architecture summary ----
        with st.expander("SCB model architecture"):
            st.code(
                """
Input:  (batch, 96, 96, 3)  — RGB student crop
  → MobileNetV2 preprocess (scale to [-1, 1])
  → MobileNetV2 backbone (ImageNet pretrained)
      Phase 1: all layers FROZEN
      Phase 2: top 30 layers UNFROZEN          → (batch, 3, 3, 1280)
  → GlobalAveragePooling2D                     → (batch, 1280)
  → BatchNormalization
  → Dense(256, relu) + Dropout(0.4)
  → Dense(6, softmax)                          → (batch, 6)

Classes: hand_raising | paying_attention | writing |
         distracted   | bored            | phone_use

Training:
  Phase 1  Adam lr=1e-3  ~20 epochs + augmentation + class weights
  Phase 2  Adam lr=1e-4  ~20 epochs (top 30 unfrozen)
""",
                language="text",
            )

    # ======================================================================
    # Tab 2: FER2013 — SECONDARY / BASELINE
    # ======================================================================
    with tab_fer:
        st.warning(
            "**FER2013 is a SECONDARY / BASELINE signal only.**  "
            "Its predictions are **never used** for primary engagement scoring.  "
            "It is kept here solely for academic comparison.",
            icon="⚠️",
        )

        fer_model   = load_fer_model()
        fer_metrics = FER_DEMO_METRICS

        if fer_model is not None:
            st.success("FER2013 model checkpoint found (`model/fer_mobilenetv2.keras`).")
        else:
            st.info(
                "FER2013 model not trained.  "
                "Train with: `python3 -m model.train_fer --data_dir /path/to/fer2013`"
            )

        if fer_metrics.get("note"):
            st.caption(f"ℹ️ {fer_metrics['note']}")

        c1, c2 = st.columns(2)
        c1.metric("FER2013 Accuracy", f"{fer_metrics['accuracy']:.1%}")
        c2.metric("FER2013 Macro F1", f"{fer_metrics['macro_f1']:.1%}")

        st.subheader("Per-Class Metrics (FER2013)")
        fig_fer_bar = _per_class_bar(
            fer_metrics["per_class"], EMOTION_CLASSES, ambiguous=AMBIGUOUS_EMOTION_CLASSES
        )
        st.pyplot(fig_fer_bar, use_container_width=True)

        rows = []
        for cls in EMOTION_CLASSES:
            m      = fer_metrics["per_class"][cls]
            flag   = "⚠️" if cls in AMBIGUOUS_EMOTION_CLASSES else ""
            rows.append({
                "Emotion":            f"{flag} {cls}".strip(),
                "Precision":          f"{m['precision']:.2f}",
                "Recall":             f"{m['recall']:.2f}",
                "F1":                 f"{m['f1']:.2f}",
                "Engagement proxy?":  "Ambiguous" if cls in AMBIGUOUS_EMOTION_CLASSES else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("⚠️ = emotion whose engagement interpretation is ambiguous or unreliable.")

        st.subheader("Confusion Matrix (FER2013)")
        fig_fer_cm = _cm_fig(
            fer_metrics["confusion_matrix"],
            EMOTION_CLASSES,
            "FER2013 — MobileNetV2 Baseline (Secondary Signal Only)",
        )
        st.pyplot(fig_fer_cm, use_container_width=True)

        st.error(
            "**FER2013 known limitations:**\n"
            "- Skewed toward lighter skin tones (Buolamwini & Gebru, 2018).\n"
            "- Label ambiguity: inter-rater agreement ≈ 65 %.\n"
            "- No body context — face-only crops miss posture / phone signals.\n"
            "- Emotion ≠ engagement: Neutral / Sad / Surprise cannot be reliably "
            "mapped to engaged or disengaged states.",
            icon="🚨",
        )

        # Optional DeepFace section
        st.markdown("---")
        st.subheader("DeepFace Fallback (Optional, Secondary Signal)")
        if _DEEPFACE_AVAILABLE:
            st.success("DeepFace is installed (secondary signal demo only).")
            uploaded_face = st.file_uploader(
                "Upload a face crop — DeepFace will predict its emotion (not engagement)",
                type=["jpg", "jpeg", "png"], key="deepface_upload",
            )
            if uploaded_face is not None:
                import cv2 as _cv2
                from PIL import Image as _Image
                pil  = _Image.open(uploaded_face).convert("RGB")
                arr  = np.array(pil)
                blur = _cv2.GaussianBlur(arr, (31, 31), 0)
                st.image(blur, caption="Uploaded face (blurred for display)", width=200)
                with st.spinner("Running DeepFace …"):
                    try:
                        result = _DeepFace.analyze(img_path=arr, actions=["emotion"],
                                                    enforce_detection=False, silent=True)
                        probs  = result[0]["emotion"]
                        st.bar_chart(pd.DataFrame.from_dict(
                            {"Probability (%)": probs}, orient="columns"
                        ))
                        st.caption(
                            "DeepFace (VGG-Face) emotion predictions — secondary reference only.  "
                            "NOT used by the SCB behavioral-proxy pipeline."
                        )
                    except Exception as e:
                        st.error(f"DeepFace failed: {e}")
        else:
            st.info(
                "Install DeepFace for FER-based live emotion inference:\n\n"
                "```bash\npip install deepface==0.0.89\n```\n\n"
                "Downloads VGG-Face weights (~500 MB) on first use.  "
                "Fully offline thereafter.  Secondary signal only."
            )
