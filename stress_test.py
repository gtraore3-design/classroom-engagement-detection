"""
stress_test.py
==============
Reproducible robustness evaluation for the Classroom Engagement Detection System.

Purpose
-------
This module supports robustness validation by subjecting the existing detection
pipeline to controlled image-degradation scenarios.  It answers the question:
"Does the system degrade gracefully when image quality drops?"

It is intentionally kept as a standalone script — it imports and exercises the
*existing* pipeline (pipeline/detector.py + pipeline/scorer.py) without
modifying any production code.

Usage
-----
    python stress_test.py

Outputs
-------
data/stress_tests/<scenario>.jpg   — saved degraded images
data/stress_results.csv            — per-scenario metrics table
data/stress_tests/pulse_chart.png  — bar chart of pulse score by scenario

Scenarios
---------
normal    — baseline; no modification
low_light — heavy luminance reduction (simulates dim lecture-hall lighting)
blurred   — strong Gaussian blur  (simulates camera out of focus / motion)
noisy     — additive Gaussian noise (simulates low-ISO sensor noise)
occlusion — synthetic rectangles block parts of the image (pillars, glare)

Real-world images
-----------------
If any .jpg / .png images are placed in data/real_world_tests/, they are
evaluated automatically using the same pipeline and appended to the results.

References
----------
Dodge & Karam (2016). Understanding How Image Quality Affects Deep Neural
Networks. ICIP. — motivates the chosen degradation types.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — works whether run from repo root or from any subdirectory
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))   # ensure local pipeline package is importable

from pipeline.detector import detect_persons
from pipeline.scorer import compute_scores

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
_DATA          = _REPO / "data"
_SAMPLE_IMG    = _DATA / "sample_classroom.jpg"
_STRESS_DIR    = _DATA / "stress_tests"
_REALWORLD_DIR = _DATA / "real_world_tests"
_CSV_OUT       = _DATA / "stress_results.csv"
_CHART_OUT     = _STRESS_DIR / "pulse_chart.png"

_STRESS_DIR.mkdir(parents=True, exist_ok=True)
_REALWORLD_DIR.mkdir(parents=True, exist_ok=True)

# Default expected class size used for all stress-test evaluations.
# Matches the demo default in gradio_app.py.
_DEFAULT_EXPECTED = 20

# ---------------------------------------------------------------------------
# Image degradation generators
# ---------------------------------------------------------------------------

def _apply_normal(img: np.ndarray) -> np.ndarray:
    """Return the image unchanged (baseline scenario)."""
    return img.copy()


def _apply_low_light(img: np.ndarray, gamma: float = 3.0) -> np.ndarray:
    """
    Simulate dim lecture-hall lighting via inverse gamma correction.

    gamma > 1 darkens the image.  A value of 3.0 reduces mean luminance to
    roughly 30 % of the original — representative of a poorly lit classroom
    recorded without flash or supplemental lighting.
    """
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img, table)


def _apply_blur(img: np.ndarray, ksize: int = 21) -> np.ndarray:
    """
    Apply strong Gaussian blur to simulate a defocused or motion-blurred camera.

    ksize=21 is typical of a camera that is moderately out of focus at a
    2–3 m working distance.  Must be odd.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _apply_noise(img: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """
    Add zero-mean Gaussian noise (sigma in [0, 255] units).

    sigma=30 is representative of an ISO-3200 consumer camera sensor in
    moderate indoor light — a common classroom recording scenario.
    """
    noise = np.random.default_rng(seed=42).normal(0.0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def _apply_occlusion(img: np.ndarray, n_blocks: int = 6) -> np.ndarray:
    """
    Overlay dark rectangles to simulate partial occlusion.

    Occlusions mimic real-world obstacles: pillars, students standing in
    front of the camera, or specular glare patches.  Blocks are placed
    pseudo-randomly but with a fixed seed for reproducibility.
    """
    out = img.copy()
    h, w = out.shape[:2]
    rng  = np.random.default_rng(seed=7)

    for _ in range(n_blocks):
        bw = int(rng.integers(w // 10, w // 5))
        bh = int(rng.integers(h // 8,  h // 4))
        bx = int(rng.integers(0, max(1, w - bw)))
        by = int(rng.integers(0, max(1, h - bh)))
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)

    return out


# Registry: (scenario_name, transform_function)
# Order matches the robustness table in the Gradio UI for consistency.
SCENARIOS: list[tuple[str, callable]] = [
    ("normal",    _apply_normal),
    ("low_light", _apply_low_light),
    ("blurred",   _apply_blur),
    ("noisy",     _apply_noise),
    ("occlusion", _apply_occlusion),
]

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    bgr: np.ndarray,
    expected: int = _DEFAULT_EXPECTED,
) -> dict:
    """
    Run detect_persons → compute_scores on a single BGR image.

    Returns a flat dict with the metrics we care about for robustness reporting:
        detected      — number of students detected
        class_pulse   — attendance-adjusted engagement score (0–100)
        pulse_label   — 'High' | 'Moderate' | 'Low'
        elapsed_ms    — wall-clock time for the full pipeline (ms)
    """
    t0      = time.perf_counter()
    persons = detect_persons(bgr)
    result  = compute_scores(persons, expected)
    elapsed = (time.perf_counter() - t0) * 1000.0

    return {
        "detected":    result["detected"],
        "class_pulse": result["class_score_pct"],
        "pulse_label": result["pulse_label"],
        "elapsed_ms":  round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def _save_bar_chart(rows: list[dict], out_path: Path) -> None:
    """
    Save a horizontal bar chart of class-pulse score by scenario.

    Colour coding mirrors the gauge zones used in the main Gradio UI:
        green  ≥ 70 %  →  High
        amber  ≥ 40 %  →  Moderate
        red    <  40 %  →  Low
    """
    names   = [r["scenario"] for r in rows]
    pulses  = [r["class_pulse"] for r in rows]
    colours = []
    for p in pulses:
        if p >= 70:
            colours.append("#2ecc71")
        elif p >= 40:
            colours.append("#f39c12")
        else:
            colours.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.55 + 1.5)))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    bars = ax.barh(names, pulses, color=colours, height=0.55,
                   edgecolor="#ffffff22", linewidth=0.8)

    # Value labels
    for bar, val in zip(bars, pulses):
        ax.text(
            min(val + 1.5, 97), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", ha="left", fontsize=9, color="#e6edf3", fontweight="600",
        )

    # Zone lines
    for xv, lbl, clr in [(40, "40 % (Moderate)", "#f39c12"),
                          (70, "70 % (High)",     "#2ecc71")]:
        ax.axvline(xv, color=clr, lw=1.0, ls="--", alpha=0.55)
        ax.text(xv + 0.5, -0.55, lbl, color=clr, fontsize=7.5, va="top")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Class Pulse Score (%)", fontsize=9, color="#8b949e", labelpad=6)
    ax.set_title(
        "Robustness Evaluation — Class Pulse by Scenario",
        fontsize=11, color="#e6edf3", pad=10, fontweight="700",
    )
    ax.tick_params(colors="#8b949e", labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.grid(True, color="#ffffff15", lw=0.5)

    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pretty table printer
# ---------------------------------------------------------------------------

_COL_W = [18, 10, 13, 9, 12]   # column widths for the terminal table

def _print_table(rows: list[dict]) -> None:
    """Print a formatted results table to stdout."""
    header = ("Scenario", "Detected", "Pulse Score", "Label", "Time (ms)")
    sep    = "  ".join("-" * w for w in _COL_W)

    print()
    print("=" * 70)
    print("  Classroom Engagement — Stress Test Results")
    print("=" * 70)
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, _COL_W)))
    print("  " + sep)
    for r in rows:
        row = (
            r["scenario"],
            str(r["detected"]),
            f"{r['class_pulse']:.1f}%",
            r["pulse_label"],
            f"{r['elapsed_ms']:.1f}",
        )
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, _COL_W)))
    print("  " + sep)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stress_tests(expected: int = _DEFAULT_EXPECTED) -> list[dict]:
    """
    Run all controlled stress scenarios on the bundled sample image,
    then evaluate any images found in data/real_world_tests/.

    Parameters
    ----------
    expected : Expected class enrolment fed to compute_scores().

    Returns
    -------
    List of result dicts, one per scenario/image evaluated.
    """
    # ── Load base image ───────────────────────────────────────────────────────
    if not _SAMPLE_IMG.exists():
        # Auto-generate the sample if it hasn't been created yet
        print(f"[stress_test] Sample image not found — generating via make_sample.py …")
        from data.make_sample import create_sample_classroom
        base = create_sample_classroom()
        cv2.imwrite(str(_SAMPLE_IMG), base, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        base = cv2.imread(str(_SAMPLE_IMG))

    if base is None:
        sys.exit(f"[stress_test] ERROR: Could not load {_SAMPLE_IMG}")

    print(f"[stress_test] Base image loaded: {_SAMPLE_IMG.name}"
          f"  ({base.shape[1]}×{base.shape[0]} px)")
    print(f"[stress_test] Expected class size: {expected}")
    print(f"[stress_test] Running {len(SCENARIOS)} controlled scenarios …\n")

    rows: list[dict] = []

    # ── Controlled scenarios ──────────────────────────────────────────────────
    for name, transform in SCENARIOS:
        degraded = transform(base)

        # Save degraded image for visual inspection / reproducibility
        out_img = _STRESS_DIR / f"{name}.jpg"
        cv2.imwrite(str(out_img), degraded, [cv2.IMWRITE_JPEG_QUALITY, 90])

        metrics = _run_pipeline(degraded, expected)
        row = {"scenario": name, "source": "controlled", **metrics}
        rows.append(row)
        print(f"  [{name:<10}] detected={metrics['detected']:>3}  "
              f"pulse={metrics['class_pulse']:5.1f}%  "
              f"label={metrics['pulse_label']:<8}  "
              f"time={metrics['elapsed_ms']:.1f} ms")

    # ── Real-world images (optional) ──────────────────────────────────────────
    real_imgs = sorted(_REALWORLD_DIR.glob("*.jpg")) + \
                sorted(_REALWORLD_DIR.glob("*.png")) + \
                sorted(_REALWORLD_DIR.glob("*.jpeg"))

    if real_imgs:
        print(f"\n[stress_test] Found {len(real_imgs)} real-world image(s) in"
              f" {_REALWORLD_DIR.name}/")
        for img_path in real_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [SKIP] {img_path.name} — could not load")
                continue
            metrics = _run_pipeline(img, expected)
            row = {"scenario": img_path.stem, "source": "real_world", **metrics}
            rows.append(row)
            print(f"  [{img_path.name:<18}] detected={metrics['detected']:>3}  "
                  f"pulse={metrics['class_pulse']:5.1f}%  "
                  f"label={metrics['pulse_label']:<8}  "
                  f"time={metrics['elapsed_ms']:.1f} ms")
    else:
        print(f"\n[stress_test] No images in {_REALWORLD_DIR.name}/ — skipping "
              f"real-world evaluation.")
        print(f"  Tip: place .jpg or .png classroom photos there and re-run.")

    return rows


def save_results(rows: list[dict]) -> None:
    """Write rows to data/stress_results.csv."""
    fieldnames = ["scenario", "source", "detected",
                  "class_pulse", "pulse_label", "elapsed_ms"]
    with open(_CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[stress_test] Results saved → {_CSV_OUT.relative_to(_REPO)}")


def save_chart(rows: list[dict]) -> None:
    """Save bar chart to data/stress_tests/pulse_chart.png."""
    _save_bar_chart(rows, _CHART_OUT)
    print(f"[stress_test] Bar chart saved → {_CHART_OUT.relative_to(_REPO)}")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stress-test the classroom engagement detection pipeline."
    )
    parser.add_argument(
        "--expected", type=int, default=_DEFAULT_EXPECTED,
        help=f"Expected class enrolment (default: {_DEFAULT_EXPECTED})",
    )
    args = parser.parse_args()

    results = run_stress_tests(expected=args.expected)

    _print_table(results)
    save_results(results)
    save_chart(results)

    print("[stress_test] Done.\n")
