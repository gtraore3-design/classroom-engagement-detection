---
title: Classroom Engagement Detector
emoji: 🎓
colorFrom: blue
colorTo: teal
sdk: gradio
sdk_version: 4.0.0
app_file: gradio_app.py
pinned: false
---

# 🎓 Classroom Engagement Detector

An OpenCV-only classroom engagement analyzer built as a Gradio web app.
**No TensorFlow, no MediaPipe, no cloud API** — everything runs locally
using classical computer-vision techniques.

## What it does

Upload a front-facing classroom photograph and the system will:

1. **Detect students** using Haar face cascades and a HOG pedestrian detector
2. **Estimate behavioral signals** from body geometry alone:
   - Head pose (face aspect ratio + Haar eye symmetry)
   - Posture (body-box height/width ratio)
   - Hand raise (skin-colour blob above face)
   - Phone use (bright rectangle in lap — applied as penalty)
   - Talking (two faces side-by-side at same height)
3. **Score each student** with a weighted rubric
4. **Compute the class pulse** — attendance-adjusted aggregate score
5. **Display results** with colour-coded boxes, a gauge, and a signal chart

All detected faces are **Gaussian-blurred** before display.
No images or personal data are stored or transmitted.

---

## Engagement score formula

```
score_i = 0.30 × head_pose
        + 0.25 × posture
        + 0.25 × hand_raise
        + 0.10 × talking
        − 0.20 × phone_detected

class_pulse = Σ(score_i) / expected_class_size × 100
```

A baseline attentive student (head forward, upright, no phone, not talking,
no hand raised) scores **0.60 → Engaged**.

Absent students implicitly contribute 0, so low attendance naturally
penalises the class pulse without a separate factor.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/gtraore3-design/classroom-engagement-detection.git
cd classroom-engagement-detection

# 2. Install (Python 3.9–3.13)
pip install -r requirements.txt

# 3. Run
python gradio_app.py
# → http://localhost:7860
```

For a public link (Gradio tunnel):

```bash
python gradio_app.py --share
```

---

## File structure

```
gradio_app.py          ← Gradio Blocks UI + analyze() entry point
pipeline/
  detector.py          ← HOG + Haar detection + behavioral signals
  scorer.py            ← Weighted scoring + class-pulse computation
  visualizer.py        ← Privacy blurring + annotation + charts
data/
  make_sample.py       ← Synthetic demo-classroom image generator
  sample_classroom.jpg ← Auto-generated on first run
requirements.txt
README.md
```

---

## Detection tips

| Situation | Recommendation |
|-----------|----------------|
| Few faces detected | Try a higher-resolution image; ensure faces are ≥ 30 × 30 px |
| HOG misses seated students | HOG is trained on standing figures; face cascade is more reliable for seated students |
| Lighting issues | CLAHE pre-processing handles most indoor mixed-lighting scenarios |
| Back-row students | Use a wide-angle or zoom lens to bring distant students closer |

---

## Privacy & ethics

- **No facial emotion recognition** — engagement is inferred from body
  geometry, not from face appearance or emotion labels
- **Faces blurred** before any image is displayed
- **In-memory only** — no database writes, no file storage, no logging
- **Aggregate output** — class-level scores only; no individual is named
  or tracked
- **Demographic neutrality** — body-geometry proxies are less susceptible
  to the skin-tone and demographic biases documented in FER systems
  (Buolamwini & Gebru, 2018)

---

## References

- Viola & Jones (2001). Rapid object detection using a boosted cascade of
  simple features. *CVPR*.
- Dalal & Triggs (2005). Histograms of oriented gradients for human
  detection. *CVPR*.
- Raca, Tormey & Dillenbourg (2015). Sleepers' lag — motion and attention
  in the classroom. *Procedia SBS*.
- Buolamwini & Gebru (2018). Gender Shades. *FAccT*.

---

*CIS 515 Project — Classroom Engagement Detection System.*
