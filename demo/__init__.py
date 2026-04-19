"""
demo/ — synthetic classroom data and image generation for demo / grader mode.

When no real classroom image is available (e.g., during grading), the demo
module supplies:
  - A pre-generated classroom layout image (PNG)
  - Synthetic PersonResult objects with realistic engagement variation
  - Pre-computed engagement_info and attendance_info dicts

All synthetic results are clearly labelled as DEMO DATA in the UI.
"""

from .demo_data import get_demo_results
from .sample_image import get_demo_image_bgr

__all__ = ["get_demo_results", "get_demo_image_bgr"]
