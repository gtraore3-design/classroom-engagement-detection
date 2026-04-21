"""
pipeline/detector.py — Person and face detection for real classroom images.

No MediaPipe or TensorFlow required.  Uses only OpenCV built-ins:
  • Haar cascade face detector  (frontalface + profileface)
  • Haar cascade eye detector   (head-pose proxy via eye count)
  • HOG pedestrian detector     (bodies without visible faces)

Detection pipeline
------------------
1. CLAHE histogram equalisation on the luminance channel to normalise
   mixed classroom lighting (windows + artificial lights).
2. Multi-scale frontal-face detection (Haar) tuned for the range of
   distances typical in a front-of-class camera view.
3. Profile-face detection to catch students turned sideways.
4. Eye detection within each face ROI → head-pose proxy:
      2 symmetric eyes  →  'forward'    (engaged)
      1 eye             →  'tilted'     (neutral)
      0 eyes            →  'down'       (disengaged)
5. HOG pedestrian detection to find bodies with no visible face → counted
   as present-but-not-visible (labelled 'no_face').
6. IoU-based NMS to remove duplicate detections across detectors.

Head-pose angle proxy (no landmark model required)
---------------------------------------------------
The presence and relative vertical positions of the two eyes encodes the
coarse head rotation angle:
  |Δy_eyes| / face_h < 0.15   →  nearly frontal  (engaged)
  |Δy_eyes| / face_h ≥ 0.15   →  tilt > ~15 °    (neutral)
  no eyes detected             →  head bowed       (disengaged)
  face aspect ratio < 0.50    →  turned > ~60 °    (neutral override)

Reference: Viola & Jones (2001), Dalal & Triggs (2005).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Haar cascade handles — bundled with every OpenCV installation
# ---------------------------------------------------------------------------

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# HOG pedestrian detector (Dalal & Triggs SVM, bundled with OpenCV)
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ---------------------------------------------------------------------------
# Detection tuning constants
# ---------------------------------------------------------------------------

# Haar face detector — tuned for students 1–6 m from camera
_FACE_SCALE      = 1.08   # image pyramid step (smaller = more sensitive, slower)
_FACE_NEIGHBORS  = 4      # higher = fewer false positives
_FACE_MIN_SIZE   = (18, 18)   # smallest face detected (pixels) — catches distant rows
_FACE_MAX_SIZE   = (220, 220) # largest face — avoids catching the whole whiteboard

# Haar eye detector — run on grayscale face ROI (equalised)
_EYE_SCALE       = 1.1
_EYE_NEIGHBORS   = 3
_EYE_MIN_SIZE    = (5, 5)

# HOG pedestrian detector parameters
_HOG_WIN_STRIDE  = (8, 8)
_HOG_PADDING     = (4, 4)
_HOG_SCALE       = 1.05

# NMS overlap threshold for merging redundant detections
_NMS_THRESH      = 0.35

# Face aspect-ratio threshold below which a face is considered profile/turned
_PROFILE_ASPECT  = 0.50   # w/h < 0.50 → turned ≥ ~60 °

# Eye y-symmetry threshold: |Δy| / face_h above this → tilted head
_EYE_TILT_THRESH = 0.15


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class StudentDetection:
    """
    One detected person/student in a classroom frame.

    Attributes
    ----------
    x, y, w, h      : Bounding box of the person region used for the coloured
                       annotation box.  For face-based detections this is the
                       face box; for HOG-only detections it is the body box.
    face_x, face_y,
    face_w, face_h   : Face sub-region (same as x/y/w/h for face detections).
                       All four are -1 for HOG-only detections.
    engagement       : 'engaged' | 'neutral' | 'disengaged' | 'no_face'
    eye_count        : Number of eyes detected within face ROI (0–2+).
    is_hog_only      : True when the person was found by HOG but no face was
                       detected in their body region.
    head_pose        : Coarse pose label: 'forward' | 'tilted' | 'down' | 'unknown'
    """
    x: int;  y: int;  w: int;  h: int
    face_x: int = -1;  face_y: int = -1
    face_w: int = -1;  face_h: int = -1
    engagement:  str = "no_face"
    eye_count:   int = 0
    is_hog_only: bool = False
    head_pose:   str = "unknown"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nms(rects: list[tuple], overlap_thresh: float = _NMS_THRESH) -> list[tuple]:
    """
    IoU-based non-maximum suppression.

    Parameters
    ----------
    rects         : list of (x, y, w, h) tuples
    overlap_thresh: IoU threshold above which a weaker box is suppressed

    Returns
    -------
    Filtered list of (x, y, w, h) tuples.
    """
    if not rects:
        return []
    arr  = np.array(rects, dtype=float)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]
    area   = (arr[:, 2] + 1) * (arr[:, 3] + 1)
    idxs   = np.argsort(y2)
    keep   = []
    while len(idxs):
        last = idxs[-1]
        keep.append(last)
        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (area[idxs[:-1]] + area[last] - inter + 1e-6)
        idxs  = np.delete(idxs, np.where(iou > overlap_thresh)[0])
        idxs  = idxs[:-1]
    return [tuple(map(int, arr[i])) for i in keep]


def _clahe_equalise(bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast-Limited Adaptive Histogram Equalisation) to the
    luminance channel of an LAB-colour image.

    CLAHE improves detection reliability under mixed indoor/outdoor lighting —
    a common challenge in classroom photography — without introducing the
    halos that plain histogram equalisation produces.
    """
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _rect_overlap(r1: tuple, r2: tuple) -> float:
    """Return IoU between two (x, y, w, h) rectangles."""
    ax1, ay1 = r1[0], r1[1]
    ax2, ay2 = r1[0]+r1[2], r1[1]+r1[3]
    bx1, by1 = r2[0], r2[1]
    bx2, by2 = r2[0]+r2[2], r2[1]+r2[3]
    ix = max(0, min(ax2,bx2) - max(ax1,bx1))
    iy = max(0, min(ay2,by2) - max(ay1,by1))
    inter = ix * iy
    union = (r1[2]*r1[3]) + (r2[2]*r2[3]) - inter
    return inter / (union + 1e-6)


def _face_near_body(face_rect: tuple, body_rect: tuple) -> bool:
    """
    Return True if a face rect lies in the upper-60 % of a body rect.
    Used to suppress HOG detections that already have a Haar face match.
    """
    fx, fy = face_rect[0], face_rect[1]
    bx, by, bw, bh = body_rect
    in_x = bx - face_rect[2] <= fx <= bx + bw
    in_y = by <= fy <= by + int(bh * 0.65)
    return in_x and in_y


def _classify_head_pose(face_gray: np.ndarray, face_w: int, face_h: int) -> tuple[str, int]:
    """
    Estimate head pose and count eyes from a grayscale face ROI.

    Strategy
    --------
    1. Run Haar eye detector on the upper-65 % of the face (eyes are never
       in the lower third of a face).
    2. Keep at most the 2 strongest detections (by area).
    3. Compute the vertical symmetry of detected eyes.

    Returns
    -------
    (pose_label, eye_count)
        pose_label: 'forward' | 'tilted' | 'down'
        eye_count : 0, 1, or 2
    """
    # Only search the upper 65 % of the face for eyes
    eye_region = face_gray[: int(face_h * 0.65), :]

    eyes = _eye_cascade.detectMultiScale(
        eye_region,
        scaleFactor=_EYE_SCALE,
        minNeighbors=_EYE_NEIGHBORS,
        minSize=_EYE_MIN_SIZE,
    )

    if not isinstance(eyes, np.ndarray) or len(eyes) == 0:
        return "down", 0

    # Keep at most 2 eyes, preferring larger detections
    eyes_sorted = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
    n_eyes = len(eyes_sorted)

    if n_eyes == 1:
        return "tilted", 1

    # Two eyes — check vertical symmetry
    ey1 = eyes_sorted[0][1] + eyes_sorted[0][3] / 2   # y-centre of eye 1
    ey2 = eyes_sorted[1][1] + eyes_sorted[1][3] / 2   # y-centre of eye 2
    delta_y_ratio = abs(ey1 - ey2) / (face_h + 1e-6)

    if delta_y_ratio < _EYE_TILT_THRESH:
        return "forward", 2
    else:
        return "tilted", 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_students(bgr_image: np.ndarray) -> list[StudentDetection]:
    """
    Detect all visible students in a classroom BGR image.

    Parameters
    ----------
    bgr_image : Full classroom frame in BGR colour order (any resolution).

    Returns
    -------
    List of StudentDetection, one per detected person.  The list may be
    empty if the image is too dark, too small, or contains no people.

    Notes
    -----
    Detection is deterministic — same image always produces the same output.
    Processing typically takes 200–800 ms on a modern laptop CPU for a
    1280×960 classroom image.
    """
    if bgr_image is None or bgr_image.size == 0:
        return []

    h, w = bgr_image.shape[:2]

    # --- Step 1: Normalise lighting with CLAHE ---
    bgr_eq  = _clahe_equalise(bgr_image)
    gray_eq = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2GRAY)

    # --- Step 2: Frontal face detection ---
    frontal_rects = _face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=_FACE_SCALE,
        minNeighbors=_FACE_NEIGHBORS,
        minSize=_FACE_MIN_SIZE,
        maxSize=_FACE_MAX_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    frontal_list: list[tuple] = (
        [tuple(map(int, r)) for r in frontal_rects]
        if isinstance(frontal_rects, np.ndarray)
        else []
    )

    # --- Step 3: Profile face detection (students turned sideways) ---
    profile_rects = _profile_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=_FACE_SCALE,
        minNeighbors=_FACE_NEIGHBORS,
        minSize=_FACE_MIN_SIZE,
        maxSize=_FACE_MAX_SIZE,
    )
    profile_list: list[tuple] = (
        [tuple(map(int, r)) for r in profile_rects]
        if isinstance(profile_rects, np.ndarray)
        else []
    )

    # Merge frontal + profile, removing duplicates with NMS
    all_face_rects = _nms(frontal_list + profile_list, overlap_thresh=0.40)

    # --- Step 4: Per-face eye detection and head-pose classification ---
    detections: list[StudentDetection] = []

    for (fx, fy, fw, fh) in all_face_rects:
        # Clamp ROI to image bounds
        fx1, fy1 = max(0, fx), max(0, fy)
        fx2, fy2 = min(w, fx+fw), min(h, fy+fh)
        face_gray = gray_eq[fy1:fy2, fx1:fx2]

        if face_gray.size == 0:
            continue

        # Face aspect ratio check (very narrow → profile/turned)
        aspect = fw / (fh + 1e-6)
        if aspect < _PROFILE_ASPECT:
            pose, n_eyes = "tilted", 1   # profile face; assume 1 visible eye
        else:
            pose, n_eyes = _classify_head_pose(face_gray, fw, fh)

        # Map pose → engagement label
        if pose == "forward":
            engagement = "engaged"
        elif pose in ("tilted",):
            engagement = "neutral"
        else:   # "down"
            engagement = "disengaged"

        # Expand the annotation box slightly for readability
        pad = int(fh * 0.12)
        bx = max(0, fx - pad)
        by = max(0, fy - pad)
        bw = min(w - bx, fw + 2*pad)
        bh = min(h - by, fh + 2*pad)

        detections.append(StudentDetection(
            x=bx, y=by, w=bw, h=bh,
            face_x=fx, face_y=fy, face_w=fw, face_h=fh,
            engagement=engagement,
            eye_count=n_eyes,
            is_hog_only=False,
            head_pose=pose,
        ))

    # --- Step 5: HOG pedestrian detection → persons without a visible face ---
    # Scale down large images for HOG speed (HOG is O(pixels))
    max_dim    = 800
    scale_hog  = min(1.0, max_dim / max(h, w))
    bgr_small  = cv2.resize(bgr_eq, (0, 0), fx=scale_hog, fy=scale_hog)

    hog_rects_raw, _ = _hog.detectMultiScale(
        bgr_small,
        winStride=_HOG_WIN_STRIDE,
        padding=_HOG_PADDING,
        scale=_HOG_SCALE,
    )

    hog_rects: list[tuple] = []
    if isinstance(hog_rects_raw, np.ndarray) and len(hog_rects_raw):
        # Re-scale back to original image coordinates
        hog_rects = [
            (int(x/scale_hog), int(y/scale_hog),
             int(bw_/scale_hog), int(bh_/scale_hog))
            for (x, y, bw_, bh_) in hog_rects_raw
        ]
        hog_rects = _nms(hog_rects, overlap_thresh=0.45)

    # Add HOG detections that don't already have a Haar face match
    for hbox in hog_rects:
        has_face = any(_face_near_body((d.face_x, d.face_y, d.face_w, d.face_h), hbox)
                       for d in detections if not d.is_hog_only)
        if not has_face:
            hx, hy, hbw, hbh = hbox
            detections.append(StudentDetection(
                x=hx, y=hy, w=hbw, h=hbh,
                engagement="no_face",
                is_hog_only=True,
                head_pose="unknown",
            ))

    return detections
