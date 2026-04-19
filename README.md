# Classroom Engagement Detection System

**CIS 515 Project — 2026**

An AI-powered classroom engagement detector built with Streamlit, MediaPipe, OpenCV, and TensorFlow.
Engagement is measured via **behavioral proxies** (hand raises, head pose, gaze direction, posture, phone detection) — *not* facial emotion recognition — making the system more reliable and ethically defensible.

---

## Quick Start

### 1. Clone / unzip the project

```bash
cd "classroom_engagement"
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** TensorFlow 2.15 requires the `tensorflow-macos` and
> `tensorflow-metal` packages instead of `tensorflow`. Run:
> ```bash
> pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
> pip install -r requirements.txt --ignore-requires-python
> ```

### 4. Launch the app

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## Demo Mode (no classroom image required)

Toggle **Demo Mode** in the left sidebar.  This loads pre-computed synthetic
results for 20 students in a 4 × 5 classroom grid — all pipeline modules,
charts, and the engagement heatmap are populated with realistic-looking data.

Demo results are clearly labelled throughout the UI.

---

## Training the SCB Behavior Model

The primary model is trained on the **SCB-Dataset** (Student Classroom Behavior
by Mohanta et al., 2023), which contains labeled images across six behavioral
classes: `hand_raising`, `paying_attention`, `writing`, `distracted`, `bored`,
`phone_use`.

The Model Evaluation page shows reference metrics by default.  Follow the steps
below to train the actual model.

### Step 1 — Download the SCB-Dataset

**Option A — Automated (requires free Kaggle account):**

```bash
# Set up Kaggle credentials once:
# 1. Log in at kaggle.com → Account → Settings → API → "Create New API Token"
# 2. Place kaggle.json at ~/.kaggle/kaggle.json
# Then:
pip install kaggle
python3 setup.py --download-scb           # extracts to ./data/scb by default
python3 setup.py --download-scb --output-dir /custom/path
```

**Option B — Manual (no credentials needed):**

```
https://www.kaggle.com/datasets/asthalochanmohanta/class-room-student-behaviour
```

Download, unzip to `./data/scb/`.  Two layouts are supported automatically:

| Layout | Structure |
|--------|-----------|
| ImageFolder | `data/scb/train/<class>/` and `data/scb/test/<class>/` |
| YOLO annotation | `data/scb/images/`, `data/scb/annotations/`, `train.txt`, `test.txt` |

### Step 2 — Verify the download

```bash
python3 setup.py --check-env
```

### Step 3 — Train

```bash
python3 -m model.train --data_dir ./data/scb --epochs 40
```

Training runs in two phases:

| Phase | Layers unfrozen | Learning rate | Epochs |
|-------|----------------|---------------|--------|
| 1 | Classification head only | 1e-3 | 20 |
| 2 | Top 30 MobileNetV2 layers | 1e-4 | 20 |

The trained model is saved to `model/scb_mobilenetv2.keras` and automatically
loaded by the app on the next launch.  Class weights are computed automatically
to handle SCB class imbalance.

---

## Training the FER2013 Emotion Baseline (optional)

The FER2013 model is a **secondary signal only**, shown on the Model Evaluation
page for reference.  It is not required to run the app.

```
https://www.kaggle.com/datasets/msambare/fer2013
```

Download and unzip.  Then:

```bash
python3 -m model.train_fer --data_dir /path/to/fer2013 --epochs 30
```

### DeepFace fallback (no training required)

Uncomment `deepface==0.0.89` in `requirements.txt` and re-install.  DeepFace
downloads a pre-trained VGG-Face model on first use.

---

## File Structure

```
classroom_engagement/
├── app.py                         Main Streamlit entry point
├── config.py                      All weights, thresholds, constants
├── requirements.txt               Pinned Python dependencies
├── setup.py                       SCB-Dataset download helper + env check
│
├── pipeline/
│   ├── behavioral_detection.py    MediaPipe Face Mesh + Pose analysis
│   ├── attendance.py              HOG body detector + face count
│   ├── engagement_scorer.py       Weighted rubric + attendance adjustment
│   └── face_blur.py               Privacy: Gaussian-blur all detected faces
│
├── model/
│   ├── behavior_model.py          MobileNetV2 architecture for SCB-Dataset
│   ├── fer_model.py               MobileNetV2 architecture for FER2013 baseline
│   ├── train.py                   SCB two-phase transfer-learning training script
│   ├── evaluate.py                sklearn metrics + demo fallback metrics
│   └── (train_fer.py)             Optional FER2013 training script
│
├── ui/
│   ├── upload_analyze.py          Page 1: upload, pipeline, image preview
│   ├── dashboard.py               Page 2: gauge, proxy bars, heatmap table
│   ├── model_evaluation.py        Page 3: confusion matrix, per-class F1
│   └── about.py                   Page 4: methodology, ethics, references
│
├── demo/
│   ├── demo_data.py               Synthetic PersonResult objects (20 students)
│   └── sample_image.py            Generates synthetic classroom diagram (PIL)
│
└── utils/
    └── visualization.py           Heatmap overlay + annotation drawing
```

---

## Methodology

### Behavioral Proxy Pipeline

For each person detected in a classroom image:

| Signal | Method | Weight |
|--------|--------|--------|
| **Hand raised** | Wrist landmark y < shoulder y (MediaPipe Pose) | 30 % |
| **Head forward** | Euler pitch/yaw via solvePnP on 6 Face Mesh landmarks | 25 % |
| **Gaze forward** | Iris offset in eye bounding box (refined Face Mesh landmarks) | 20 % |
| **Good posture** | Shoulder–hip spine angle from vertical | 15 % |
| **Phone absent** | Wrist-near-hip + wrist-separation heuristic (Pose) | 10 % |

Each signal is a continuous score in **[0, 1]**.  The per-person score is the weighted dot product.  A student is classified as *engaged* if their score ≥ 0.50.

### Attendance-Adjusted Classroom Score

```
classroom_score = avg_behavioral_score × (detected / expected_class_size)
```

This prevents a small highly-engaged group from masking low attendance.  A warning banner appears when fewer than 50 % of expected students are detected.

### CNN Emotion Model (Secondary Signal)

A MobileNetV2 backbone (ImageNet pre-trained) is fine-tuned on FER2013 using two-phase transfer learning:

- **Phase 1** — Freeze MobileNetV2, train classification head (Adam lr=1e-3, 15 epochs)
- **Phase 2** — Unfreeze top 30 MobileNetV2 layers, fine-tune (Adam lr=1e-4, 15 epochs)

Emotion predictions are displayed with explicit caveats and are never used as the sole engagement signal.

---

## Evaluation

### Quantitative — SCB-Dataset (primary model)

| Metric | Value (MobileNetV2 reference) |
|--------|-------------------------------|
| Accuracy | ~78 % |
| Macro F1 | ~75 % |
| Best class F1 | hand_raising — high (visually distinct) |
| Worst class F1 | bored / distracted (visually similar) |

> Reference values from the SCB-Dataset paper (Mohanta et al., 2023).
> Run `python3 -m model.train --data_dir ./data/scb` to compute exact metrics
> on your local test split.

### Quantitative — FER2013 (secondary emotion baseline)

| Metric | Value (MobileNetV2 reference) |
|--------|-------------------------------|
| Accuracy | 67.2 % |
| Macro F1 | 64.3 % |
| Best class F1 | Happy — 0.89 |
| Worst class F1 | Fear — 0.52 |

### Qualitative (Demo Mode)

Enable Demo Mode in the sidebar to see the system's full output — engagement gauge, proxy breakdown bar chart, spatial heatmap, and per-student table — on a synthetic 20-student classroom.

---

## Known Limitations

1. **Single-frame analysis** — Engagement is a temporal phenomenon; one image is a snapshot.
2. **Occlusion** — Students behind others are missed by both face and body detectors.
3. **Camera angle** — System is calibrated for a front-facing camera; side views degrade accuracy.
4. **MediaPipe Pose is single-person** — We run it per-crop from HOG boxes, which degrades with heavy overlap.
5. **Emotion → engagement ambiguity** — Neutral, Sad, and Surprise are unreliable engagement proxies.
6. **Phone heuristic coarseness** — The wrist-proximity heuristic produces false positives for students who rest their hands in their laps.
7. **Demographic coverage** — MediaPipe's face detection has known performance gaps across skin tones.

---

## Ethical Safeguards (built-in, not just documented)

| Safeguard | Implementation |
|-----------|----------------|
| No face storage | `blur_faces()` runs before any display; no crop is retained |
| In-memory only | All arrays are ephemeral; no file I/O during analysis |
| Privacy notice | Displayed on every image upload |
| Aggregate output | No individual is named or tracked |
| Disclaimer banner | Visible on every results page |
| Bias disclosure | Built into the Model Evaluation page |
| Secondary signal | CNN emotion output is clearly labelled and never primary |

---

## Dependencies

See `requirements.txt` for pinned versions.  Key packages:

- `streamlit 1.32` — web UI
- `mediapipe 0.10.11` — Face Mesh, Pose, Face Detection
- `opencv-python 4.9` — HOG detector, image processing, solvePnP
- `tensorflow 2.15` — MobileNetV2 backbone
- `scikit-learn 1.4` — evaluation metrics
- `plotly 5.20` — interactive gauge chart
- `kaggle` (optional) — automated SCB-Dataset download via `setup.py`

---

## References

- Mohanta, A. et al. (2023). SCB-Dataset: A Dataset for Detecting Student Classroom Behavior Using Computer Vision. *arXiv:2304.02488*.
- Buolamwini, J. & Gebru, T. (2018). Gender Shades. *FAT\* 2018*.
- Dewan, M.A.A. et al. (2019). Engagement detection in online learning: a review. *Smart Learning Environments*, 6(1).
- Goodfellow, I. et al. (2013). Challenges in representation learning: FER2013. *ICML Workshop*.
- Mota, S. & Picard, R. (2003). Automated posture analysis for detecting learner's interest level. *CVPR Workshop*.
- Raca, M., Tormey, R., & Dillenbourg, P. (2015). Sleepers' lag — study on motion and attention. *Procedia Social and Behavioral Sciences*.
- Whitehill, J. et al. (2014). The faces of engagement. *IEEE Trans. Affective Computing*, 5(1), 86–98.
- MediaPipe: https://mediapipe.dev
- SCB-Dataset: https://www.kaggle.com/datasets/asthalochanmohanta/class-room-student-behaviour
- FER2013 dataset: https://www.kaggle.com/datasets/msambare/fer2013
