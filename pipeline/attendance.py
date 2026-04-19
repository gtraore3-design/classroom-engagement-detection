"""
Attendance detection: estimate how many people are present in a frame.

Strategy (cascade, fastest-first):
  1. MediaPipe Face Detection  → face count (fast, reliable indoors)
  2. OpenCV HOG person detector → body count (fallback / cross-check)

The face count is the primary signal because it is more robust in typical
classroom images (frontal seated poses, partial occlusion of bodies).
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline._mp_probe import MEDIAPIPE_OK
if MEDIAPIPE_OK:
    import mediapipe as mp
else:
    mp = None

from config import HOG_WIN_STRIDE, HOG_PADDING, HOG_SCALE, HOG_NMS_THRESHOLD, FACE_MIN_DETECTION

try:
    _face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=FACE_MIN_DETECTION,
    )
except AttributeError:
    _face_det = None

_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def _nms(boxes: list[tuple], overlap_thresh: float) -> list[tuple]:
    """Simple IoU-based non-maximum suppression."""
    if not boxes:
        return []
    boxes_arr = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes], dtype=float)
    x1, y1, x2, y2 = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    pick = []
    while len(idxs):
        last = idxs[-1]
        pick.append(last)
        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])
        ov = np.maximum(0, (xx2 - xx1 + 1) * (yy2 - yy1 + 1)) / area[idxs[:-1]]
        idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(ov > overlap_thresh)[0])))
    return [boxes[i] for i in pick]


def count_faces(bgr_image: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    """
    Count faces via MediaPipe Face Detection.
    Returns (count, list_of_bbox_xywh).
    """
    if _face_det is None:
        return 0, []
    h, w = bgr_image.shape[:2]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = _face_det.process(rgb)
    boxes = []
    if results.detections:
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = int(bb.width  * w)
            bh = int(bb.height * h)
            boxes.append((x, y, bw, bh))
    return len(boxes), boxes


def count_bodies(bgr_image: np.ndarray) -> tuple[int, list[tuple]]:
    """
    Count people via OpenCV HOG descriptor.
    Returns (count, list_of_bbox_xywh).
    """
    # Resize for speed; HOG works best on ~400-600 px wide images
    scale_factor = min(1.0, 640 / max(bgr_image.shape[:2]))
    small = cv2.resize(bgr_image, (0, 0), fx=scale_factor, fy=scale_factor)
    rects, _ = _hog.detectMultiScale(
        small,
        winStride=HOG_WIN_STRIDE,
        padding=HOG_PADDING,
        scale=HOG_SCALE,
    )
    if len(rects) == 0:
        return 0, []
    rects_list = [(int(x / scale_factor), int(y / scale_factor),
                   int(w / scale_factor), int(h / scale_factor))
                  for (x, y, w, h) in rects]
    picked = _nms(rects_list, HOG_NMS_THRESHOLD)
    return len(picked), picked


def estimate_attendance(bgr_image: np.ndarray) -> dict:
    """
    Full attendance estimate combining face and body counts.

    Returns a dict with keys:
        face_count    – MediaPipe face detections
        body_count    – HOG body detections
        best_estimate – recommended count (max of the two)
        face_boxes    – list of (x, y, w, h) for faces
        body_boxes    – list of (x, y, w, h) for bodies
    """
    face_count, face_boxes = count_faces(bgr_image)
    body_count, body_boxes = count_bodies(bgr_image)
    best = max(face_count, body_count)
    return {
        "face_count":    face_count,
        "body_count":    body_count,
        "best_estimate": best,
        "face_boxes":    face_boxes,
        "body_boxes":    body_boxes,
    }
