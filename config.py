"""
config.py — Central configuration: weights, thresholds, class labels, and
CNN fusion parameters.

Primary dataset: SCB-Dataset (Student Classroom Behavior)
    Mohanta, A. et al. (2023). SCB-Dataset: A Dataset for Detecting Student
    Classroom Behavior. arXiv:2304.02488.
    Kaggle: kaggle.com/datasets/asthalochanmohanta/class-room-student-behaviour

Baseline / secondary dataset: FER2013 (facial emotion recognition only,
    kept as a historical comparator on the Model Evaluation page).
"""

# ===========================================================================
# PRIMARY: SCB-Dataset behavior classes
# ===========================================================================

# The six student behavioral categories in the SCB-Dataset.
# Order matches the model output layer and confusion matrix rows/columns.
SCB_BEHAVIOR_CLASSES = [
    "hand_raising",
    "paying_attention",
    "writing",
    "distracted",
    "bored",
    "phone_use",
]

# Engagement score assigned to each SCB class (literature-informed).
# Used to convert CNN class probabilities → a scalar engagement signal.
#
# Justification:
#   hand_raising      — 1.00  strongest active-participation signal
#   paying_attention  — 0.85  attentive orientation; high engagement
#   writing           — 0.70  on-task but head may be bowed; moderate signal
#   distracted        — 0.25  off-task, low engagement
#   bored             — 0.15  passive disengagement; lowest energy state
#   phone_use         — 0.05  clear off-task device use
SCB_ENGAGEMENT_SCORES = {
    "hand_raising":     1.00,
    "paying_attention": 0.85,
    "writing":          0.70,
    "distracted":       0.25,
    "bored":            0.15,
    "phone_use":        0.05,
}

# How each SCB behavior class maps to individual MediaPipe proxy signals.
# Used in behavioral_detection.py to update proxy scores when the CNN fires.
# Each inner dict maps proxy name → expected proxy score for that behavior.
SCB_TO_PROXY = {
    "hand_raising": {
        "hand_raised_score":   1.00,
        "head_forward_score":  0.80,
        "gaze_forward_score":  0.70,
        "good_posture_score":  0.85,
        "phone_absent_score":  1.00,
    },
    "paying_attention": {
        "hand_raised_score":   0.10,
        "head_forward_score":  0.92,
        "gaze_forward_score":  0.90,
        "good_posture_score":  0.82,
        "phone_absent_score":  1.00,
    },
    "writing": {
        "hand_raised_score":   0.00,
        "head_forward_score":  0.55,   # head tilted down toward paper
        "gaze_forward_score":  0.45,
        "good_posture_score":  0.65,
        "phone_absent_score":  1.00,
    },
    "distracted": {
        "hand_raised_score":   0.00,
        "head_forward_score":  0.28,
        "gaze_forward_score":  0.22,
        "good_posture_score":  0.40,
        "phone_absent_score":  0.85,
    },
    "bored": {
        "hand_raised_score":   0.00,
        "head_forward_score":  0.32,
        "gaze_forward_score":  0.28,
        "good_posture_score":  0.22,
        "phone_absent_score":  0.92,
    },
    "phone_use": {
        "hand_raised_score":   0.00,
        "head_forward_score":  0.12,
        "gaze_forward_score":  0.08,
        "good_posture_score":  0.18,
        "phone_absent_score":  0.00,
    },
}

# CNN / MediaPipe fusion weights (must sum to 1.0).
# When the SCB model is loaded, final proxy scores are a weighted blend of
# CNN-derived proxy values and raw MediaPipe landmark measurements.
SCB_CNN_WEIGHT     = 0.55   # CNN is slightly preferred (domain-matched training)
MEDIAPIPE_WEIGHT   = 0.45   # MediaPipe heuristics retained for robustness

# SCB model checkpoint path (relative to model/)
SCB_MODEL_FILENAME = "scb_mobilenetv2.keras"

# ===========================================================================
# Behavioral proxy engagement weights (must sum to 1.0)
# ===========================================================================
ENGAGEMENT_WEIGHTS = {
    "hand_raised":   0.30,   # strongest active-participation signal
    "head_forward":  0.25,   # head pose facing instructor / board
    "gaze_forward":  0.20,   # eyes directed toward front
    "good_posture":  0.15,   # upright or forward lean
    "phone_absent":  0.10,   # no phone / device detected
}

assert abs(sum(ENGAGEMENT_WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

# Engagement threshold: per-person score ≥ this → classified as "engaged"
ENGAGED_THRESHOLD = 0.50

# ===========================================================================
# Head-pose thresholds (degrees from frontal)
# ===========================================================================
HEAD_PITCH_DOWN_DEG = 20   # chin-down angle that fully penalises head_forward
HEAD_YAW_AWAY_DEG   = 30   # lateral turn that fully penalises head_forward
HEAD_ENGAGED_PITCH  = 15   # small downward tilt tolerated (note-taking)

# ===========================================================================
# Gaze thresholds
# ===========================================================================
GAZE_AWAY_RATIO = 0.30     # iris offset > 30 % of eye width → looking away

# ===========================================================================
# Posture thresholds
# ===========================================================================
SLOUCH_ANGLE_DEG = 25      # shoulder–hip spine angle > this → slouching

# ===========================================================================
# Hand-raise detection
# ===========================================================================
HAND_RAISE_MARGIN_FRAC = 0.05   # wrist must be ≥ 5 % of image height above shoulder

# ===========================================================================
# Attendance warning
# ===========================================================================
LOW_ATTENDANCE_THRESHOLD = 0.50   # < 50 % of expected class → warning banner

# ===========================================================================
# SECONDARY: FER2013 (emotion baseline, shown on Model Evaluation page only)
# ===========================================================================
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Ambiguous emotion → engagement mappings (discussed in Model Evaluation page)
AMBIGUOUS_EMOTION_CLASSES = {"Neutral", "Sad", "Surprise"}

# FER2013 model training hyper-parameters
FER_IMG_SIZE  = (48, 48)
FER_BATCH_SIZE = 64
FER_EPOCHS     = 30

# Legacy mapping (kept for the Model Evaluation baseline section)
EMOTION_ENGAGEMENT_MAP = {
    "Happy":    0.80,
    "Surprise": 0.60,
    "Neutral":  0.50,
    "Angry":    0.25,
    "Fear":     0.20,
    "Sad":      0.15,
    "Disgust":  0.10,
}

# ===========================================================================
# SCB-Dataset training hyper-parameters
# ===========================================================================
SCB_IMG_SIZE   = (96, 96)    # larger than FER (real classroom crops, RGB)
SCB_BATCH_SIZE = 32          # smaller batch — SCB images are larger and more varied
SCB_EPOCHS     = 40          # more epochs: real classroom domain shift from ImageNet

# ===========================================================================
# MediaPipe
# ===========================================================================
MAX_NUM_FACES      = 12
POSE_MIN_DETECTION = 0.5
POSE_MIN_TRACKING  = 0.5
FACE_MIN_DETECTION = 0.5

# ===========================================================================
# Person detection (OpenCV HOG)
# ===========================================================================
HOG_WIN_STRIDE    = (8, 8)
HOG_PADDING       = (4, 4)
HOG_SCALE         = 1.05
HOG_NMS_THRESHOLD = 0.45

# ===========================================================================
# Visualisation
# ===========================================================================
HEATMAP_ALPHA = 0.45

# ===========================================================================
# Privacy
# ===========================================================================
FACE_BLUR_KERNEL = 51   # Gaussian kernel size (must be odd)
