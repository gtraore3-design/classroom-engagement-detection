"""
Subprocess probes for mediapipe and TensorFlow.

Both C extensions abort the process (SIGABRT) on unsupported Python versions
(e.g. 3.13+).  SIGABRT cannot be caught with try/except, so we test each
import in an isolated subprocess; a non-zero exit means the library is unsafe
to import and every dependent module degrades gracefully to demo/fallback mode.

Both probes run once at package-import time (Python module cache).
"""

from __future__ import annotations

import subprocess
import sys


def _probe(pkg: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


MEDIAPIPE_OK: bool = _probe("mediapipe")
TF_OK: bool        = _probe("tensorflow")
