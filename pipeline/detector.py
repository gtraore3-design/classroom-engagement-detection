"""
pipeline/detector.py
====================
Person detection and behavioral-signal extraction for classroom images.

OpenCV only — no TensorFlow, no MediaPipe.

Detection strategy
------------------
Primary   : Haar frontal + profile face cascade → reliable per-student anchor
Secondary : HOG pedestrian detector → catch students with no visible face

For each detected face the pipeline estimates a body region (extend ~3×
face-height below, ~0.8× face-width to either side) and tries to snap
that estimate onto any nearby HOG body box.

Behavioral signals (body geometry only — no emotion recognition)
----------------------------------------------------------------
HEAD POSE    face aspect ratio + Haar eye symmetry          weight 30 %
POSTURE      body-box h/w ratio (tall-narrow = upright)     weight 25 %
HAND RAISE   skin-blob in the region above the face         weight 25 %
PHONE USE    bright rectangle in lap region (penalty)       weight −20 %
TALKING      two face centroids close & same height          weight 10 %

Head-pose → engagement mapping
-------------------------------
forward  (2 symmetric eyes, normal aspect)  → score 1.00
tilted   (1 eye, or slightly narrow aspect) → score 0.65
away     (very narrow aspect, ≤1 eye)       → score 0.20
down     (normal aspect but 0 eyes)         → score 0.30
unknown  (no face detected)                 → score 0.40

References
----------
Viola & Jones (2001); Dalal & Triggs (2005); Raca et al. (2015).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Cascade handles (bundled with every OpenCV installation)
# ---------------------------------------------------------------------------

_face_front = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_face_prof = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
_eye_casc = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# HOG pedestrian detector (Dalal & Triggs SVM, bundled with OpenCV)
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Face cascade — tuned for students 1–7 m from a front-of-class camera
_FACE_SCALE   = 1.10
_FACE_NEIGH   = 4
_FACE_MIN     = (20, 20)    # catches distant back-row students
_FACE_MAX     = (220, 220)  # avoids whiteboard false positives

# Eye cascade — run on face ROI only
_EYE_SCALE    = 1.10
_EYE_NEIGH    = 3
_EYE_MIN      = (5, 5)

# HOG — tuned for seated/partial-body detections
_HOG_STRIDE   = (8, 8)
_HOG_PAD      = (16, 16)
_HOG_SCALE    = 1.05
_HOG_MAX_DIM  = 640          # downscale before HOG for speed

# NMS thresholds
_FACE_NMS     = 0.40
_HOG_NMS      = 0.45
_BODY_NMS     = 0.35

# Head pose aspect-ratio thresholds (face_w / face_h)
_ASPECT_FRONTAL = 0.60       # ≥ this → likely frontal
_ASPECT_AWAY    = 0.40       # ≤ this → turned away / profile

# Posture: body_h / body_w ratio
_POSTURE_UPRIGHT = 2.0       # ≥ this → very upright
_POSTURE_SLUMP   = 1.2       # ≤ this → slouching

# Skin detection HSV ranges (covers diverse skin tones)
_SKIN_LO1 = np.array([ 0,  15,  50], dtype=np.uint8)
_SKIN_HI1 = np.array([25, 255, 255], dtype=np.uint8)
_SKIN_LO2 = np.array([160,  15,  50], dtype=np.uint8)
_SKIN_HI2 = np.array([180, 255, 255], dtype=np.uint8)

# Fraction of skin pixels in raise-region required for hand-raise signal
_RAISE_SKIN_RATIO = 0.10

# Phone contour: w/h aspect and area bounds
_PHONE_ASPECT_MIN = 0.35
_PHONE_ASPECT_MAX = 0.80
_PHONE_AREA_MIN   = 400
_PHONE_AREA_MAX   = 7000

# Talking: faces within N face-widths horizontally, M face-heights vertically
_TALK_X_MULT = 1.8
_TALK_Y_FRAC = 0.45

# Eye y-symmetry: |Δcy| / face_h below this → frontal
_EYE_TILT_THRESH = 0.15


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class PersonDetection:
    """
    One detected student.

    Body box (bx/by/bw/bh) is used for the annotation rectangle.
    Face box (fx/fy/fw/fh) is used for blurring; −1 when absent.
    All score fields are in [0, 1]; phone_score = 1.0 means phone detected.
    """
    # Body bounding box
    bx: int;  by: int;  bw: int;  bh: int
    # Face bounding box (-1 when no face found)
    fx: int = -1;  fy: int = -1;  fw: int = -1;  fh: int = -1
    # Behavioral signal scores
    head_pose_score:  float = 0.40
    head_pose_label:  str   = "unknown"
    posture_score:    float = 0.50
    hand_raise_score: float = 0.00
    phone_score:      float = 0.00    # 1.0 → phone detected (penalty in scorer)
    talking_score:    float = 0.50    # 0.5 = neutral / not observed
    # Icon flags for the annotation overlay
    hand_raised:      bool  = False
    phone_detected:   bool  = False
    talking:          bool  = False
    has_face:         bool  = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nms(rects: list[tuple], thresh: float) -> list[tuple]:
    """IoU-based non-maximum suppression on (x, y, w, h) tuples."""
    if not rects:
        return []
    arr  = np.array(rects, dtype=float)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    area   = (arr[:, 2] + 1) * (arr[:, 3] + 1)
    idxs   = np.argsort(y2)
    keep: list[int] = []
    while len(idxs):
        last = idxs[-1];  keep.append(last)
        xx1  = np.maximum(x1[last], x1[idxs[:-1]])
        yy1  = np.maximum(y1[last], y1[idxs[:-1]])
        xx2  = np.minimum(x2[last], x2[idxs[:-1]])
        yy2  = np.minimum(y2[last], y2[idxs[:-1]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (area[idxs[:-1]] + area[last] - inter + 1e-6)
        idxs  = np.delete(idxs, np.where(iou > thresh)[0])
        idxs  = idxs[:-1]
    return [tuple(map(int, arr[i])) for i in keep]


def _clahe(bgr: np.ndarray) -> np.ndarray:
    """CLAHE histogram equalisation on the L channel (mixed lighting fix)."""
    lab        = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b    = cv2.split(lab)
    l_eq       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def _skin_mask(bgr_roi: np.ndarray) -> np.ndarray:
    """Binary mask of skin-coloured pixels (covers diverse skin tones)."""
    hsv  = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, _SKIN_LO1, _SKIN_HI1)
    m2   = cv2.inRange(hsv, _SKIN_LO2, _SKIN_HI2)
    mask = cv2.bitwise_or(m1, m2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


# ---------------------------------------------------------------------------
# Per-face behavioral signals
# ---------------------------------------------------------------------------

def _head_pose(gray_face: np.ndarray, fw: int, fh: int) -> tuple[float, str]:
    """
    Estimate head orientation from a grayscale face ROI.

    Returns (score ∈ [0,1], label).
    """
    aspect = fw / (fh + 1e-6)

    # Very narrow face → turned away (profile or >60° yaw)
    if aspect < _ASPECT_AWAY:
        return 0.20, "away"

    # Eye detection in upper 65 % of face (eyes never in lower third)
    eye_roi = gray_face[: int(fh * 0.65), :]
    raw_eyes = _eye_casc.detectMultiScale(
        eye_roi,
        scaleFactor=_EYE_SCALE,
        minNeighbors=_EYE_NEIGH,
        minSize=_EYE_MIN,
    )
    n_eyes = min(len(raw_eyes) if isinstance(raw_eyes, np.ndarray) else 0, 2)

    # Moderately narrow → tilted / partial profile
    if aspect < _ASPECT_FRONTAL:
        if n_eyes >= 1:
            return 0.55, "tilted"
        return 0.20, "away"

    # Frontal aspect ratio
    if n_eyes == 0:
        return 0.30, "down"        # head bowed
    if n_eyes == 1:
        return 0.65, "tilted"

    # Two eyes — check vertical symmetry for precise forward/tilted distinction
    eyes_s = sorted(raw_eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    cy1 = eyes_s[0][1] + eyes_s[0][3] / 2
    cy2 = eyes_s[1][1] + eyes_s[1][3] / 2
    if abs(cy1 - cy2) / (fh + 1e-6) < _EYE_TILT_THRESH:
        return 1.00, "forward"
    return 0.70, "tilted"


def _posture_score(bh: int, bw: int) -> float:
    """
    Score posture from body bounding-box aspect ratio (h/w).

    Seated upright students produce tall-narrow HOG boxes;
    slouching students produce shorter, wider boxes.
    """
    ratio = bh / (bw + 1e-6)
    if   ratio >= _POSTURE_UPRIGHT:  return 1.00
    elif ratio >= 1.70:              return 0.85
    elif ratio >= 1.40:              return 0.65
    elif ratio >= _POSTURE_SLUMP:    return 0.40
    else:                            return 0.20


def _hand_raise_score(
    bgr: np.ndarray,
    fx: int, fy: int, fw: int, fh: int,
    img_h: int, img_w: int,
) -> float:
    """
    Detect raised hand via skin-coloured blob above the face bounding box.

    Searches a region spanning one face-height above the face and
    1.5× face-width across.  Returns a continuous score [0, 1].
    """
    pad = fw // 2
    rx1 = max(0, fx - pad);   rx2 = min(img_w, fx + fw + pad)
    ry1 = max(0, fy - fh);    ry2 = max(0, fy)   # directly above face
    if ry2 <= ry1 or rx2 <= rx1:
        return 0.0

    roi  = bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return 0.0

    mask  = _skin_mask(roi)
    ratio = float(mask.sum()) / (255.0 * roi.shape[0] * roi.shape[1] + 1e-6)
    if ratio >= _RAISE_SKIN_RATIO:
        return 1.0
    return min(ratio / (_RAISE_SKIN_RATIO + 1e-6), 0.5)


def _phone_detected(
    bgr: np.ndarray,
    bx: int, by: int, bw: int, bh: int,
    img_h: int, img_w: int,
) -> bool:
    """
    Detect a phone screen in the lower half of the body box.

    Phone screens are small, bright, and roughly portrait-oriented
    rectangles.  We threshold on brightness then scan contours for
    rectangular shapes matching phone proportions and area.
    """
    y0 = by + int(bh * 0.50);   y1 = min(img_h, by + bh)
    x0 = max(0, bx);             x1 = min(img_w, bx + bw)
    if y1 <= y0 or x1 <= x0:
        return False

    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    th    = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area   = w * h
        aspect = w / (h + 1e-6)
        if (_PHONE_AREA_MIN < area < _PHONE_AREA_MAX and
                _PHONE_ASPECT_MIN < aspect < _PHONE_ASPECT_MAX):
            return True
    return False


def _estimate_body(
    fx: int, fy: int, fw: int, fh: int,
    img_h: int, img_w: int,
) -> tuple[int, int, int, int]:
    """
    Estimate a body bounding box from a face box.

    For seated students the torso extends roughly 3× face-height below the
    face; shoulders are about 0.9× face-width on each side.
    """
    pad_x = int(fw * 0.9)
    bx = max(0, fx - pad_x)
    by = max(0, fy)
    bw = min(img_w - bx, fw + 2 * pad_x)
    bh = min(img_h - by, int(fh * 3.2))
    return bx, by, bw, bh


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_persons(bgr: np.ndarray) -> list[PersonDetection]:
    """
    Detect all visible students in a classroom BGR image and compute
    per-person behavioral signal scores.

    Parameters
    ----------
    bgr : Full classroom frame in BGR colour order (any resolution).

    Returns
    -------
    List of PersonDetection, one per detected person.  May be empty if the
    image is featureless or contains no people.

    Processing is deterministic — same image always produces the same output.
    """
    if bgr is None or bgr.size == 0:
        return []

    img_h, img_w = bgr.shape[:2]

    # ── Step 1: CLAHE lighting normalisation ─────────────────────────────────
    eq   = _clahe(bgr)
    gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)

    # ── Step 2: Face detection (frontal + profile, merged with NMS) ──────────
    raw_front = _face_front.detectMultiScale(
        gray, scaleFactor=_FACE_SCALE, minNeighbors=_FACE_NEIGH,
        minSize=_FACE_MIN, maxSize=_FACE_MAX, flags=cv2.CASCADE_SCALE_IMAGE,
    )
    raw_prof = _face_prof.detectMultiScale(
        gray, scaleFactor=_FACE_SCALE, minNeighbors=_FACE_NEIGH,
        minSize=_FACE_MIN, maxSize=_FACE_MAX,
    )

    face_list: list[tuple] = (
        ([tuple(map(int, r)) for r in raw_front]
         if isinstance(raw_front, np.ndarray) else []) +
        ([tuple(map(int, r)) for r in raw_prof]
         if isinstance(raw_prof, np.ndarray) else [])
    )
    faces = _nms(face_list, _FACE_NMS)

    # ── Step 3: HOG body detection (on downscaled image for speed) ───────────
    scale_hog = min(1.0, _HOG_MAX_DIM / max(img_h, img_w, 1))
    small     = cv2.resize(eq, (0, 0), fx=scale_hog, fy=scale_hog)
    hog_raw, _ = _hog.detectMultiScale(
        small,
        winStride=_HOG_STRIDE,
        padding=_HOG_PAD,
        scale=_HOG_SCALE,
    )
    hog_bodies: list[tuple] = []
    if isinstance(hog_raw, np.ndarray) and len(hog_raw):
        hog_bodies = _nms([
            (int(x / scale_hog), int(y / scale_hog),
             int(w / scale_hog), int(h / scale_hog))
            for x, y, w, h in hog_raw
        ], _HOG_NMS)

    # ── Step 4: Build PersonDetection for each face ───────────────────────────
    persons: list[PersonDetection] = []
    hog_used: set[int] = set()

    for (fx, fy, fw, fh) in faces:
        # Default body from face geometry
        bx, by, bw, bh = _estimate_body(fx, fy, fw, fh, img_h, img_w)

        # Snap onto a nearby HOG body if one fits
        for i, (hx, hy, hw, hh) in enumerate(hog_bodies):
            if i in hog_used:
                continue
            face_in_upper = (
                hx - fw // 2 <= fx <= hx + hw and
                hy <= fy <= hy + int(hh * 0.40)
            )
            if face_in_upper:
                bx, by, bw, bh = hx, hy, hw, hh
                hog_used.add(i)
                break

        # Face ROI for head-pose
        fx1, fy1 = max(0, fx), max(0, fy)
        fx2, fy2 = min(img_w, fx + fw), min(img_h, fy + fh)
        face_gray = gray[fy1:fy2, fx1:fx2]

        hp_score, hp_label = (
            _head_pose(face_gray, fw, fh)
            if face_gray.size > 0
            else (0.40, "unknown")
        )

        hr_score = _hand_raise_score(eq, fx, fy, fw, fh, img_h, img_w)
        phone    = _phone_detected(eq, bx, by, bw, bh, img_h, img_w)

        persons.append(PersonDetection(
            bx=bx, by=by, bw=bw, bh=bh,
            fx=fx, fy=fy, fw=fw, fh=fh,
            head_pose_score=hp_score,
            head_pose_label=hp_label,
            posture_score=_posture_score(bh, bw),
            hand_raise_score=hr_score,
            hand_raised=(hr_score >= 0.80),
            phone_score=1.0 if phone else 0.0,
            phone_detected=phone,
            has_face=True,
        ))

    # ── Step 5: HOG-only detections (bodies with no matched face) ────────────
    for i, (hx, hy, hw, hh) in enumerate(hog_bodies):
        if i in hog_used:
            continue
        # Quick secondary face search inside the top 40 % of the body box
        sub = gray[max(0, hy): min(img_h, hy + int(hh * 0.40)),
                   max(0, hx): min(img_w, hx + hw)]
        sub_faces = (
            _face_front.detectMultiScale(sub, 1.10, 3, minSize=(10, 10))
            if sub.size > 0 else []
        )
        has_face_here = isinstance(sub_faces, np.ndarray) and len(sub_faces) > 0

        phone = _phone_detected(eq, hx, hy, hw, hh, img_h, img_w)
        persons.append(PersonDetection(
            bx=hx, by=hy, bw=hw, bh=hh,
            head_pose_score=0.40,
            head_pose_label="unknown",
            posture_score=_posture_score(hh, hw),
            phone_score=1.0 if phone else 0.0,
            phone_detected=phone,
            has_face=has_face_here,
        ))

    # ── Step 6: Talking detection (face-pair proximity) ───────────────────────
    face_ps = [p for p in persons if p.has_face and p.fx >= 0]
    for i, p1 in enumerate(face_ps):
        cx1 = p1.fx + p1.fw / 2;  cy1 = p1.fy + p1.fh / 2
        for p2 in face_ps[i + 1:]:
            cx2 = p2.fx + p2.fw / 2;  cy2 = p2.fy + p2.fh / 2
            avg_fw = (p1.fw + p2.fw) / 2
            avg_fh = (p1.fh + p2.fh) / 2
            if (abs(cx1 - cx2) < _TALK_X_MULT * avg_fw and
                    abs(cy1 - cy2) < _TALK_Y_FRAC * avg_fh):
                p1.talking = True;  p1.talking_score = 1.0
                p2.talking = True;  p2.talking_score = 1.0

    return persons
