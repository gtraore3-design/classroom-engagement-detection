"""
Privacy layer: blur all detected faces before any image is displayed.
No face crops are stored or returned — only the blurred composite.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline._mp_probe import MEDIAPIPE_OK
if MEDIAPIPE_OK:
    import mediapipe as mp
else:
    mp = None

from config import FACE_BLUR_KERNEL, MAX_NUM_FACES, FACE_MIN_DETECTION

try:
    _face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1,                    # full-range model
        min_detection_confidence=FACE_MIN_DETECTION,
    )
except AttributeError:
    _face_detection = None


def blur_faces(bgr_image: np.ndarray) -> np.ndarray:
    """
    Return a copy of *bgr_image* with every detected face Gaussian-blurred.
    The original array is never mutated and no face crop is kept in memory.
    """
    output = bgr_image.copy()
    if _face_detection is None:
        return output
    h, w = output.shape[:2]

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = _face_detection.process(rgb)

    if not results.detections:
        return output

    kernel = FACE_BLUR_KERNEL | 1   # ensure odd
    for det in results.detections:
        bb = det.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w))
        y1 = max(0, int(bb.ymin * h))
        x2 = min(w, int((bb.xmin + bb.width)  * w))
        y2 = min(h, int((bb.ymin + bb.height) * h))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = output[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        output[y1:y2, x1:x2] = blurred_roi

    return output


def draw_face_boxes(bgr_image: np.ndarray, color=(0, 200, 0)) -> tuple[np.ndarray, int]:
    """
    Draw bounding rectangles around faces (on an already-blurred image).
    Returns (annotated_image, face_count).
    """
    output = bgr_image.copy()
    if _face_detection is None:
        return output, 0
    h, w = output.shape[:2]

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = _face_detection.process(rgb)

    count = 0
    if results.detections:
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width)  * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            count += 1

    return output, count
