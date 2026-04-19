"""
Page 4 — About / Methodology
Explains the behavioral-proxy approach, limitations, and ethical framework.
"""

import streamlit as st


def render_about():
    st.title("About & Methodology")

    # ------------------------------------------------------------------ #
    st.header("Why Behavioral Proxies Instead of Facial Emotions?")
    st.markdown(
        """
Facial emotion recognition (FER) has significant scientific and ethical problems as an
engagement detector:

- **Emotion ≠ engagement.** A student with a neutral expression may be deeply focused;
  a smiling student may be off-task.
- **Demographic bias.** Commercial FER systems perform substantially worse on darker skin
  tones, women, and non-Western faces (Buolamwini & Gebru 2018; Rhue 2018).
- **Label ambiguity.** Human annotators agree on FER2013 labels only ~65 % of the time.

**Behavioral proxies** bypass these problems by observing *what students do*
rather than inferring what they feel:

| Proxy | Justification |
|-------|---------------|
| Hand raises | Direct active-participation measure; gold standard in educational research |
| Head orientation | Attentional direction strongly predicts on-task behavior (Raca et al. 2015) |
| Gaze direction | Visual attention is the most robust physiological engagement correlate |
| Posture | Leaning-forward correlates with interest; slouching with fatigue/disengagement |
| Phone use | Off-task device use is the strongest behavioral disengagement indicator |
"""
    )

    # ------------------------------------------------------------------ #
    st.header("System Architecture")
    st.markdown(
        """
```
Classroom Image
       │
       ▼
┌─────────────────────┐
│  Privacy Layer      │  ← blur_faces() — immediate, before any display
│  (face_blur.py)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Attendance         │  ← MediaPipe FaceDetection + HOG people detector
│  (attendance.py)    │    → detected count vs. expected → attendance rate
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│  Behavioral Detection  (behavioral_detection.py) │
│                                                   │
│  ┌───────────────────────┐  ┌───────────────────┐│
│  │  MediaPipe (primary)  │  │  SCB CNN (primary) ││
│  │  FaceMesh: head pose, │  │  MobileNetV2 fine- ││
│  │  gaze (iris offset)   │  │  tuned on SCB-     ││
│  │  Pose: hand raise,    │  │  Dataset → one of  ││
│  │  posture, phone       │  │  6 behavior classes││
│  └──────────┬────────────┘  └────────┬──────────┘│
│             └──────── fused ─────────┘            │
│          SCB_CNN_WEIGHT=0.55 / MP_WEIGHT=0.45     │
└──────────────────────┬──────────────────────────--┘
                       │
                       ▼
┌─────────────────────┐
│  Engagement Scorer  │  ← Weighted rubric → per-person score [0, 1]
│  (engagement_       │    Attendance-adjusted classroom score
│   scorer.py)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Visualization      │  ← Heatmap overlay on blurred image
│  + Streamlit UI     │    Charts, gauges, tables (all in-memory)
└─────────────────────┘
           │ (optional secondary signal — never primary)
           ▼
┌─────────────────────┐
│  FER2013 CNN        │  ← MobileNetV2 emotion classifier
│  (fer_model.py)     │    Accuracy ~67 %; shown on Model Evaluation
│  SECONDARY ONLY     │    page with explicit bias/limitation caveats
└─────────────────────┘
```
"""
    )

    # ------------------------------------------------------------------ #
    st.header("Engagement Scoring Rubric")
    st.markdown(
        """
Each person's engagement score is a weighted sum of five behavioral signals, each in [0, 1]:

```
score = 0.30 × hand_raised
      + 0.25 × head_forward
      + 0.20 × gaze_forward
      + 0.15 × good_posture
      + 0.10 × phone_absent
```

A person is classified as **engaged** if their score ≥ 0.50.

The **classroom engagement score** is attendance-adjusted:

```
classroom_score = avg_behavioral_score × (detected_students / expected_students)
```

This ensures that an apparently high engagement rate in a half-empty room does not
mask poor attendance — a well-documented confound in classroom analytics.
"""
    )

    # ------------------------------------------------------------------ #
    st.header("Known Limitations")
    st.markdown(
        """
1. **Single-frame analysis.** Engagement is a temporal state; a single image captures only
   a moment.  Short video clips would give more reliable aggregates.

2. **Occlusion.** Students seated behind others will be missed by both the face detector
   and the pose estimator, leading to undercount.

3. **Camera angle.** Side or rear views degrade head-pose and gaze estimation accuracy
   significantly.  The system is calibrated for a front-facing classroom camera.

4. **MediaPipe accuracy.** MediaPipe Pose is trained on single-person crops; accuracy
   on multi-person classroom crops depends on bounding-box quality from HOG.

5. **Attendance confound.** `detected / expected` can only penalise, not reward, engagement.
   It cannot detect present-but-obscured students.

6. **The emotion-confusion mapping problem.** Even with accurate FER, the mapping
   emotion → engagement is not monotonic.  We document this but include the model
   as a secondary signal for research completeness only.

7. **Demographic generalisation.** MediaPipe is more accurate on faces it was trained on.
   Performance across all student populations has not been independently audited for
   this system.
"""
    )

    # ------------------------------------------------------------------ #
    st.header("Ethical Safeguards")
    st.markdown(
        """
The following safeguards are **built into the system**, not just documented:

| Safeguard | Implementation |
|-----------|----------------|
| No face storage | `blur_faces()` runs before any image display; no crop is retained |
| In-memory only | All processing in `np.ndarray`; no file writes, no database |
| On-screen notice | Privacy banner shown on every image upload |
| Aggregate output | Scores are class-level; no individual is named or tracked |
| Disclaimer | "No individual is identified or assessed" — shown on every results page |
| Ethical caveats | Bias notes built into Model Evaluation page |
| Secondary signal | CNN emotion predictions clearly labelled as secondary; never used alone |
"""
    )

    # ------------------------------------------------------------------ #
    st.header("References")
    st.markdown(
        """
- Mohanta, A. et al. (2023). *SCB-Dataset: A Dataset for Detecting Student Classroom Behavior Using Computer Vision*. arXiv:2304.02488. [Kaggle](https://www.kaggle.com/datasets/asthalochanmohanta/class-room-student-behaviour)
- Buolamwini, J. & Gebru, T. (2018). *Gender Shades*. FAT* 2018.
- Raca, M., Tormey, R., & Dillenbourg, P. (2015). Sleepers' lag — study on
  motion and attention. *Procedia — Social and Behavioral Sciences*.
- Khare, S.K. et al. (2021). Affect recognition using facial video: review.
  *Biomedical Signal Processing and Control*.
- Goodfellow, I. et al. (2013). Challenges in representation learning: A report
  on three machine learning contests (FER2013). *ICML Workshop*.
- MediaPipe: https://mediapipe.dev
- FER2013 dataset: https://www.kaggle.com/datasets/msambare/fer2013
"""
    )

    st.caption(
        "CIS 515 Project — Classroom Engagement Detection System.  "
        "Built with Streamlit, MediaPipe, OpenCV, TensorFlow, and scikit-learn."
    )
