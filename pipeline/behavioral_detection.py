"""
behavioral_detection.py — Behavioral proxy detection pipeline with SCB CNN fusion.

Signal sources
--------------
This module fuses two complementary signal sources:

1. **MediaPipe landmark heuristics** (always available, no training required)
   - Face Mesh → head pose (pitch/yaw via solvePnP), gaze (iris offset)
   - Pose       → hand raise (wrist vs. shoulder y), posture (spine angle),
                  phone-use heuristic (wrist-near-hip proximity)

2. **SCB-Dataset CNN** (available when model/scb_mobilenetv2.keras exists)
   MobileNetV2 fine-tuned on the SCB-Dataset predicts one of six behavior
   classes per student crop.  Class probabilities are converted into
   per-proxy signal scores (config.SCB_TO_PROXY) and fused with MediaPipe
   scores according to:

       final_score = SCB_CNN_WEIGHT × cnn_score + MEDIAPIPE_WEIGHT × mp_score

   When the SCB model is not loaded, all weight falls back to MediaPipe only.

Why fuse rather than replace?
------------------------------
- The SCB CNN captures global body-level patterns (arm raised, head bowed)
  that MediaPipe may miss with imprecise bounding boxes.
- MediaPipe provides sub-pixel precision on landmark geometry (iris offset,
  spine angle) that a 96×96 crop CNN cannot resolve.
- Combining the two sources is consistently more robust than either alone
  (ensemble effect, different failure modes).

SCB behavior classes → proxy alignment
---------------------------------------
  hand_raising       →  hand_raised↑  head_forward↑  good_posture↑
  paying_attention   →  head_forward↑ gaze_forward↑  good_posture↑
  writing            →  head_forward↓ (bowed)         good_posture~
  distracted         →  head_forward↓ gaze_forward↓
  bored              →  good_posture↓
  phone_use          →  phone_absent↓ head_forward↓

References
----------
Mohanta, A. et al. (2023). SCB-Dataset. arXiv:2304.02488.
Raca, M. et al. (2015). Sleepers' lag. Procedia Social and Behavioral Sciences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from pipeline._mp_probe import MEDIAPIPE_OK, TF_OK
if MEDIAPIPE_OK:
    import mediapipe as mp
else:
    mp = None

from config import (
    HEAD_PITCH_DOWN_DEG,
    HEAD_YAW_AWAY_DEG,
    HEAD_ENGAGED_PITCH,
    GAZE_AWAY_RATIO,
    SLOUCH_ANGLE_DEG,
    HAND_RAISE_MARGIN_FRAC,
    MAX_NUM_FACES,
    FACE_MIN_DETECTION,
    POSE_MIN_DETECTION,
    POSE_MIN_TRACKING,
    SCB_CNN_WEIGHT,
    MEDIAPIPE_WEIGHT,
    SCB_IMG_SIZE,
)

# ---------------------------------------------------------------------------
# MediaPipe solution handles (module-level singletons)
#
# mediapipe 0.10.30+ restructured the solutions API; on some platforms the
# attribute is absent entirely.  All three handles fall back to None so the
# rest of the module can degrade gracefully without crashing at import time.
# ---------------------------------------------------------------------------

try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands     = mp.solutions.hands
    mp_pose      = mp.solutions.pose
except AttributeError:
    mp_face_mesh = None
    mp_hands     = None
    mp_pose      = None

try:
    _face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=MAX_NUM_FACES,
        refine_landmarks=True,        # iris landmarks (469–477)
        min_detection_confidence=FACE_MIN_DETECTION,
    ) if mp_face_mesh is not None else None

    _pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=POSE_MIN_DETECTION,
        min_tracking_confidence=POSE_MIN_TRACKING,
    ) if mp_pose is not None else None
except Exception:
    _face_mesh = None
    _pose      = None

# ---------------------------------------------------------------------------
# SCB CNN model (optional singleton — None if not trained yet)
# ---------------------------------------------------------------------------
# Loaded lazily at module import time.  If the checkpoint does not exist,
# _scb_model is None and the pipeline falls back to MediaPipe-only scoring.

def _try_load_scb() -> Optional[object]:
    """Silently attempt to load the SCB model checkpoint."""
    if not TF_OK:
        return None          # TF crashes on import (SIGABRT) on this platform
    try:
        from model.behavior_model import load_scb_model
        return load_scb_model()
    except Exception:
        return None

_scb_model = _try_load_scb()

# ---------------------------------------------------------------------------
# 3-D face model for head-pose estimation (PnP)
# ---------------------------------------------------------------------------

# Six canonical 3-D facial landmark positions in mm (nose-tip = origin).
# Standard 6-point model from Kazemi & Sullivan (2014) / Guo et al. (2020).
_FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0   ],   # 1. Nose tip
    [0.0,  -330.0,  -65.0  ],  # 2. Chin
    [-225.0, 170.0, -135.0 ],  # 3. Left eye outer corner
    [ 225.0, 170.0, -135.0 ],  # 4. Right eye outer corner
    [-150.0,-150.0, -125.0 ],  # 5. Left mouth corner
    [ 150.0,-150.0, -125.0 ],  # 6. Right mouth corner
], dtype=np.float64)

_FACE_LANDMARK_IDS = [4, 152, 263, 33, 287, 57]    # corresponding Face Mesh indices

# Iris landmark indices (available with refine_landmarks=True)
_LEFT_IRIS        = [474, 475, 476, 477]
_RIGHT_IRIS       = [469, 470, 471, 472]
_LEFT_EYE_CORNERS = [362, 263]    # outer-left, outer-right of left eye
_RIGHT_EYE_CORNERS= [33,  133]    # outer-left, outer-right of right eye


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera_matrix(h: int, w: int) -> np.ndarray:
    """Pinhole camera approximation: focal = image width, centre = image centre."""
    f = float(w)
    return np.array([[f, 0., w/2.], [0., f, h/2.], [0., 0., 1.]], dtype=np.float64)


def _lm_xy(lm, w: int, h: int) -> tuple[float, float]:
    """Normalised MediaPipe landmark → pixel (x, y)."""
    return lm.x * w, lm.y * h


# ---------------------------------------------------------------------------
# Intermediate signal containers
# ---------------------------------------------------------------------------

@dataclass
class FaceSignals:
    """Signals from MediaPipe Face Mesh for one face."""
    head_pitch_deg:     float = 0.0
    head_yaw_deg:       float = 0.0
    head_forward_score: float = 0.5   # 1.0 = directly forward
    gaze_forward_score: float = 0.5   # 1.0 = iris perfectly centred
    face_center:        tuple[float, float] = (0.0, 0.0)


@dataclass
class PoseSignals:
    """Signals from MediaPipe Pose for one person crop."""
    hand_raised_score:  float = 0.0
    good_posture_score: float = 0.5
    phone_absent_score: float = 1.0   # 1.0 = no phone detected
    person_center:      tuple[float, float] = (0.0, 0.0)
    valid:              bool = False


@dataclass
class PersonResult:
    """
    Aggregated per-person engagement signals after combining all sources.

    Fields
    ------
    person_id           : Zero-based person index (arbitrary, not an identifier).
    hand_raised_score   : [0,1] — MediaPipe wrist–shoulder comparison,
                          optionally blended with SCB CNN.
    head_forward_score  : [0,1] — Face Mesh head-pose Euler angles.
    gaze_forward_score  : [0,1] — Face Mesh iris offset.
    good_posture_score  : [0,1] — Pose spine angle.
    phone_absent_score  : [0,1] — Pose phone-use heuristic (1 = absent).
    signals_available   : Which sources contributed (pose, face_mesh, scb_cnn).
    center              : (x, y) pixel centre for heatmap overlay.
    head_pitch_deg      : Raw pitch angle (display only).
    head_yaw_deg        : Raw yaw angle (display only).
    scb_class           : Most likely SCB class predicted by CNN
                          (empty string if CNN not available).
    scb_probs           : Full SCB class probability distribution.
    scb_engagement      : CNN-derived engagement score [0,1], or -1 if unavailable.
    """
    person_id:           int
    hand_raised_score:   float
    head_forward_score:  float
    gaze_forward_score:  float
    good_posture_score:  float
    phone_absent_score:  float
    signals_available:   dict  = field(default_factory=dict)
    center:              tuple[float, float] = (0.0, 0.0)
    head_pitch_deg:      float = 0.0
    head_yaw_deg:        float = 0.0
    # SCB CNN predictions (new fields)
    scb_class:           str   = ""
    scb_probs:           dict  = field(default_factory=dict)
    scb_engagement:      float = -1.0   # -1 signals "no CNN prediction available"


# ---------------------------------------------------------------------------
# Face Mesh analysis
# ---------------------------------------------------------------------------

def _analyse_face(face_landmarks, h: int, w: int) -> FaceSignals:
    """
    Extract head-pose and gaze signals from a single Face Mesh result.

    Head-pose: OpenCV solvePnP on 6 landmark points → rotation matrix →
    Euler pitch and yaw.
    Gaze: iris landmark horizontal offset within eye bounding box.
    """
    lms = face_landmarks.landmark

    try:
        pts_2d = np.array([_lm_xy(lms[i], w, h) for i in _FACE_LANDMARK_IDS],
                           dtype=np.float64)
    except IndexError:
        return FaceSignals()

    cam  = _camera_matrix(h, w)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_FACE_3D_MODEL, pts_2d, cam, dist,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return FaceSignals()

    rmat, _ = cv2.Rodrigues(rvec)
    pitch_deg = math.degrees(math.atan2(-rmat[2, 1], rmat[2, 2]))
    yaw_deg   = math.degrees(math.atan2(
        rmat[2, 0], math.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2)
    ))

    pitch_excess  = max(0.0, abs(pitch_deg) - HEAD_ENGAGED_PITCH)
    pitch_penalty = pitch_excess / max(HEAD_PITCH_DOWN_DEG - HEAD_ENGAGED_PITCH, 1.0)
    yaw_penalty   = abs(yaw_deg)  / max(HEAD_YAW_AWAY_DEG, 1.0)
    head_fwd      = float(np.clip(1.0 - 0.5*pitch_penalty - 0.5*yaw_penalty, 0.0, 1.0))

    def _gaze_score(iris_ids: list[int], corner_ids: list[int]) -> float:
        try:
            iris_pts   = np.array([[lms[i].x*w, lms[i].y*h] for i in iris_ids])
            corner_pts = np.array([[lms[i].x*w, lms[i].y*h] for i in corner_ids])
            iris_cx    = iris_pts[:, 0].mean()
            eye_cx     = corner_pts[:, 0].mean()
            eye_width  = abs(corner_pts[0, 0] - corner_pts[1, 0]) + 1e-9
            return float(np.clip(1.0 - abs(iris_cx - eye_cx)/eye_width/GAZE_AWAY_RATIO,
                                  0.0, 1.0))
        except Exception:
            return 0.5

    gaze = 0.5*_gaze_score(_LEFT_IRIS, _LEFT_EYE_CORNERS) + \
           0.5*_gaze_score(_RIGHT_IRIS, _RIGHT_EYE_CORNERS)

    return FaceSignals(
        head_pitch_deg=pitch_deg,
        head_yaw_deg=yaw_deg,
        head_forward_score=head_fwd,
        gaze_forward_score=float(np.clip(gaze, 0.0, 1.0)),
        face_center=(lms[4].x*w, lms[4].y*h),
    )


# ---------------------------------------------------------------------------
# Pose analysis
# ---------------------------------------------------------------------------

_PL = mp_pose.PoseLandmark if mp_pose is not None else None


def _analyse_pose_in_crop(
    bgr_crop: np.ndarray,
    crop_origin: tuple[int, int],
    frame_h: int,
    frame_w: int,
) -> PoseSignals:
    """
    Run MediaPipe Pose on a single-person crop and return body signals.

    Parameters
    ----------
    bgr_crop    : BGR image crop (one student).
    crop_origin : (x, y) top-left corner in full-frame pixel coords.
    frame_h/w   : Full-frame dimensions for fraction-based thresholds.
    """
    if _pose is None:
        return PoseSignals()
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    res = _pose.process(rgb)
    if not res.pose_landmarks:
        return PoseSignals()

    lms    = res.pose_landmarks.landmark
    ch, cw = bgr_crop.shape[:2]
    ox, oy = crop_origin

    def _pt(lm_id: int) -> tuple[float, float, float]:
        lm = lms[lm_id]
        return (lm.x*cw + ox, lm.y*ch + oy, lm.visibility)

    ls = _pt(_PL.LEFT_SHOULDER);  rs = _pt(_PL.RIGHT_SHOULDER)
    lh = _pt(_PL.LEFT_HIP);       rh = _pt(_PL.RIGHT_HIP)
    lw = _pt(_PL.LEFT_WRIST);     rw = _pt(_PL.RIGHT_WRIST)

    def _vis(*pts, thr=0.40): return all(p[2] > thr for p in pts)

    # ---- Hand raised ----
    hand_raised = 0.0
    margin      = HAND_RAISE_MARGIN_FRAC * frame_h
    for wrist, shoulder in [(lw, ls), (rw, rs)]:
        if not _vis(wrist, shoulder): continue
        if wrist[1] < shoulder[1] - margin:
            hand_raised = max(hand_raised, 1.0)
        else:
            dist    = shoulder[1] - wrist[1]
            partial = float(np.clip((dist + margin) / (2*margin), 0.0, 0.5))
            hand_raised = max(hand_raised, partial)

    # ---- Posture (spine angle from vertical) ----
    posture_score = 0.5
    if _vis(ls, rs, lh, rh):
        sx = (ls[0]+rs[0])/2; sy = (ls[1]+rs[1])/2
        hx = (lh[0]+rh[0])/2; hy = (lh[1]+rh[1])/2
        dx = sx - hx; dy = sy - hy
        spine_angle = abs(math.degrees(math.atan2(dx, -dy))) if abs(dy) > 1e-3 else 90.0
        if   spine_angle < 10:             posture_score = 1.0
        elif spine_angle < SLOUCH_ANGLE_DEG:
            posture_score = 1.0 - (spine_angle-10)/(SLOUCH_ANGLE_DEG-10)*0.5
        else:
            posture_score = float(np.clip(1.0 - spine_angle/90.0, 0.0, 0.5))

    # ---- Phone-use heuristic ----
    phone_absent = 1.0
    if _vis(lw, rw, lh, rh, thr=0.35):
        hip_y      = (lh[1]+rh[1])/2
        hip_x_span = abs(lh[0]-rh[0]) + 1e-9
        wy_avg     = (lw[1]+rw[1])/2
        wx_sep     = abs(lw[0]-rw[0])
        if abs(wy_avg-hip_y) < 0.15*frame_h and wx_sep < 0.20*hip_x_span:
            phone_absent = 0.1   # strong phone heuristic
        elif abs(wy_avg-hip_y) < 0.15*frame_h:
            phone_absent = 0.6   # ambiguous

    # ---- Person centre (shoulder midpoint) ----
    cx = (ls[0]+rs[0])/2 if _vis(ls, rs) else ox+cw/2
    cy = (ls[1]+rs[1])/2 if _vis(ls, rs) else oy+ch/2

    return PoseSignals(
        hand_raised_score=hand_raised,
        good_posture_score=posture_score,
        phone_absent_score=phone_absent,
        person_center=(cx, cy),
        valid=True,
    )


# ---------------------------------------------------------------------------
# SCB CNN inference + signal fusion
# ---------------------------------------------------------------------------

def _run_scb_cnn(
    bgr_crop: np.ndarray,
) -> tuple[str, dict[str, float], float]:
    """
    Run the SCB CNN on a BGR person crop.

    Returns (predicted_class, probability_dict, engagement_score).
    Returns ("", {}, -1.0) if the model is not loaded.
    """
    if _scb_model is None:
        return "", {}, -1.0

    try:
        from model.behavior_model import predict_behavior, scb_probs_to_engagement
        rgb      = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        probs    = predict_behavior(_scb_model, rgb)
        top_cls  = max(probs, key=probs.__getitem__)
        eng_score = scb_probs_to_engagement(probs)
        return top_cls, probs, eng_score
    except Exception:
        return "", {}, -1.0


def _fuse_proxy_scores(
    mp_scores: dict[str, float],
    scb_probs: dict[str, float],
) -> dict[str, float]:
    """
    Blend MediaPipe proxy scores with SCB CNN proxy scores.

    When scb_probs is empty (CNN unavailable) returns mp_scores unchanged.

    Parameters
    ----------
    mp_scores  : dict of MediaPipe-derived proxy scores
    scb_probs  : SCB class probability dict from predict_behavior()

    Returns
    -------
    dict of fused proxy scores in [0, 1]
    """
    if not scb_probs:
        return mp_scores

    from model.behavior_model import scb_probs_to_proxy_scores
    cnn_scores = scb_probs_to_proxy_scores(scb_probs)

    fused = {}
    for field in mp_scores:
        mp_val  = mp_scores[field]
        cnn_val = cnn_scores.get(field, mp_val)   # fallback to mp if key missing
        fused[field] = float(np.clip(
            SCB_CNN_WEIGHT * cnn_val + MEDIAPIPE_WEIGHT * mp_val, 0.0, 1.0
        ))
    return fused


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class BehavioralDetector:
    """
    Orchestrate the full behavioral-proxy + SCB CNN pipeline for one frame.

    Checks at import time whether the SCB model checkpoint exists.
    If not present, the detector operates in MediaPipe-only mode.
    Call BehavioralDetector.scb_model_loaded() to query current mode.

    Usage::
        detector = BehavioralDetector()
        print("SCB model loaded:", detector.scb_model_loaded())
        results = detector.analyse(bgr_frame, person_boxes=[(x,y,w,h), ...])
    """

    def mediapipe_available(self) -> bool:
        """Return True if MediaPipe solutions initialised successfully."""
        return _face_mesh is not None or _pose is not None

    def scb_model_loaded(self) -> bool:
        """Return True if the SCB CNN is available for inference."""
        return _scb_model is not None

    def analyse(
        self,
        bgr_image: np.ndarray,
        person_boxes: Optional[list[tuple[int, int, int, int]]] = None,
    ) -> list[PersonResult]:
        """
        Run the full pipeline on a BGR classroom image.

        Parameters
        ----------
        bgr_image    : Full-frame BGR image.
        person_boxes : Optional list of (x, y, w, h) person bounding boxes.
                       If None, the whole image is treated as one person.

        Returns
        -------
        List of PersonResult, one per detected / assumed person.
        """
        h, w = bgr_image.shape[:2]
        rgb  = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # ---- Phase A: Face Mesh on full image (multi-face) ----
        face_signals: list[FaceSignals] = []
        if _face_mesh is not None:
            fm_res = _face_mesh.process(rgb)
            if fm_res.multi_face_landmarks:
                for fl in fm_res.multi_face_landmarks:
                    face_signals.append(_analyse_face(fl, h, w))

        # ---- Phase B: Pose + SCB CNN per person crop ----
        if not person_boxes:
            person_boxes = [(0, 0, w, h)]

        pose_signals:   list[PoseSignals] = []
        scb_predictions: list[tuple[str, dict, float]] = []

        for (px, py, pw, ph) in person_boxes:
            py_pad = max(0, py - int(0.05*h))
            py_end = min(h, py + ph + int(0.05*h))
            crop   = bgr_image[py_pad:py_end, px:px+pw]

            if crop.size == 0:
                pose_signals.append(PoseSignals())
                scb_predictions.append(("", {}, -1.0))
                continue

            pose_signals.append(_analyse_pose_in_crop(crop, (px, py_pad), h, w))
            scb_predictions.append(_run_scb_cnn(crop))

        # ---- Phase C: Merge signals into PersonResult ----
        n_persons = max(len(person_boxes), len(face_signals))
        results: list[PersonResult] = []

        for pid in range(n_persons):
            ps  = pose_signals[pid]   if pid < len(pose_signals)   else PoseSignals()
            scb = scb_predictions[pid] if pid < len(scb_predictions) else ("", {}, -1.0)
            scb_cls, scb_probs_dict, scb_eng = scb

            # Nearest-face assignment by horizontal centre
            person_cx = (ps.person_center[0] if ps.valid else
                         (person_boxes[pid][0] + person_boxes[pid][2]/2
                          if pid < len(person_boxes) else w/2))
            fs: Optional[FaceSignals] = None
            if face_signals:
                fs = min(face_signals, key=lambda f: abs(f.face_center[0] - person_cx))

            # Collect MediaPipe-derived proxy scores
            mp_scores = {
                "hand_raised_score":   ps.hand_raised_score   if ps.valid else 0.0,
                "head_forward_score":  fs.head_forward_score  if fs else 0.5,
                "gaze_forward_score":  fs.gaze_forward_score  if fs else 0.5,
                "good_posture_score":  ps.good_posture_score  if ps.valid else 0.5,
                "phone_absent_score":  ps.phone_absent_score  if ps.valid else 1.0,
            }

            # Fuse with SCB CNN scores (no-op if CNN unavailable)
            fused = _fuse_proxy_scores(mp_scores, scb_probs_dict)

            # Person centre for heatmap
            if ps.valid:
                centre = ps.person_center
            elif pid < len(person_boxes):
                bx, by, bw_, bh_ = person_boxes[pid]
                centre = (bx + bw_/2.0, by + bh_/2.0)
            else:
                centre = (w/2.0, h/2.0)

            results.append(PersonResult(
                person_id=pid,
                hand_raised_score  =fused["hand_raised_score"],
                head_forward_score =fused["head_forward_score"],
                gaze_forward_score =fused["gaze_forward_score"],
                good_posture_score =fused["good_posture_score"],
                phone_absent_score =fused["phone_absent_score"],
                signals_available={
                    "pose":      ps.valid,
                    "face_mesh": fs is not None,
                    "scb_cnn":   scb_cls != "",
                },
                center=centre,
                head_pitch_deg=fs.head_pitch_deg if fs else 0.0,
                head_yaw_deg  =fs.head_yaw_deg   if fs else 0.0,
                scb_class     =scb_cls,
                scb_probs     =scb_probs_dict,
                scb_engagement=scb_eng,
            ))

        return results
