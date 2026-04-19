"""
setup.py — Dataset download helper and environment setup script.

Downloads the SCB-Dataset from Kaggle and verifies the directory structure.
If Kaggle credentials are unavailable, prints clear instructions and exits
gracefully — the app will run in Demo Mode without the dataset.

Usage
-----
    python3 setup.py --download-scb [--output-dir ./data/scb]
    python3 setup.py --check-env
    python3 setup.py --help

Kaggle credentials
------------------
Option A (recommended): kaggle.json
    1. Log in at kaggle.com → Account → API → "Create New API Token"
    2. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json
    3. Run: python3 setup.py --download-scb

Option B: environment variables
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
    python3 setup.py --download-scb

Option C: manual download (no credentials needed)
    1. Visit kaggle.com/datasets/asthalochanmohanta/class-room-student-behaviour
    2. Click Download → unzip to ./data/scb/
    3. Run: python3 -m model.train --data_dir ./data/scb

Fallback behaviour
------------------
If credentials are not found, the script prints manual download instructions
and exits with code 0 (not an error).  The Streamlit app automatically
activates Demo Mode when no trained model is available.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).parent

# Default dataset output directory
DEFAULT_SCB_DIR = PROJECT_ROOT / "data" / "scb"

# Kaggle dataset slug
SCB_DATASET_SLUG = "asthalochanmohanta/class-room-student-behaviour"

# Expected SCB class directories (used for verification)
SCB_CLASSES = [
    "hand_raising", "paying_attention", "writing",
    "distracted", "bored", "phone_use",
]


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def check_env() -> None:
    """Print a summary of installed packages and model checkpoint status."""
    print("=" * 60)
    print("Classroom Engagement Detection — Environment Check")
    print("=" * 60)

    packages = {
        "streamlit":       "streamlit",
        "opencv-python":   "cv2",
        "mediapipe":       "mediapipe",
        "tensorflow":      "tensorflow",
        "scikit-learn":    "sklearn",
        "numpy":           "numpy",
        "pandas":          "pandas",
        "matplotlib":      "matplotlib",
        "seaborn":         "seaborn",
        "plotly":          "plotly",
        "Pillow":          "PIL",
        "kaggle (opt.)":   "kaggle",
        "deepface (opt.)": "deepface",
    }

    print("\nPackage status:")
    all_required_ok = True
    for display_name, module_name in packages.items():
        optional = "(opt.)" in display_name
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "?")
            print(f"  ✓  {display_name:<22} {version}")
        except ImportError:
            if optional:
                print(f"  -  {display_name:<22} not installed (optional)")
            else:
                print(f"  ✗  {display_name:<22} MISSING — run: pip install -r requirements.txt")
                all_required_ok = False

    print("\nModel checkpoints:")
    scb_ckpt = PROJECT_ROOT / "model" / "scb_mobilenetv2.keras"
    fer_ckpt = PROJECT_ROOT / "model" / "fer_mobilenetv2.keras"
    print(f"  {'✓' if scb_ckpt.exists() else '✗'}  SCB model:   {scb_ckpt}")
    print(f"  {'✓' if fer_ckpt.exists() else '-'}  FER model:   {fer_ckpt}")

    print("\nDataset:")
    if DEFAULT_SCB_DIR.exists():
        n_imgs = len(list(DEFAULT_SCB_DIR.rglob("*.jpg"))) + \
                 len(list(DEFAULT_SCB_DIR.rglob("*.png")))
        print(f"  ✓  SCB data at {DEFAULT_SCB_DIR} ({n_imgs} images found)")
    else:
        print(f"  -  SCB data not found at {DEFAULT_SCB_DIR}")
        print("     Run: python3 setup.py --download-scb")

    print("\nConclusion:")
    if all_required_ok:
        print("  All required packages installed.")
    else:
        print("  Missing required packages — install with: pip install -r requirements.txt")
    print("  Run 'streamlit run app.py' to launch the app (Demo Mode works without data).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Kaggle credential detection
# ---------------------------------------------------------------------------

def _kaggle_credentials_available() -> bool:
    """
    Return True if Kaggle API credentials can be found.

    Checks in order:
      1. Environment variables KAGGLE_USERNAME and KAGGLE_KEY
      2. ~/.kaggle/kaggle.json
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def _print_manual_instructions(output_dir: Path) -> None:
    """Print step-by-step manual download instructions."""
    print("\n" + "=" * 60)
    print("Kaggle credentials not found — manual download instructions")
    print("=" * 60)
    print(
        "\nOption A: Set up Kaggle credentials (automated future downloads)"
        "\n  1. Log in at https://www.kaggle.com"
        "\n  2. Go to: Account → Settings → API → 'Create New API Token'"
        "\n  3. Place kaggle.json at: ~/.kaggle/kaggle.json"
        "\n  4. Re-run: python3 setup.py --download-scb"
        "\n"
        "\nOption B: Manual download"
        "\n  1. Visit: https://www.kaggle.com/datasets/"
        "asthalochanmohanta/class-room-student-behaviour/data"
        "\n  2. Click 'Download' (requires free Kaggle account)"
        "\n  3. Unzip to: " + str(output_dir) +
        "\n  4. Run: python3 -m model.train --data_dir " + str(output_dir) +
        "\n"
        "\nFallback: Demo Mode"
        "\n  The app runs fully without the dataset."
        "\n  Enable 'Demo Mode' in the sidebar to test all features."
        "\n" + "=" * 60 + "\n"
    )


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_scb(output_dir: Path = DEFAULT_SCB_DIR) -> bool:
    """
    Download the SCB-Dataset from Kaggle using the Kaggle Python API.

    Parameters
    ----------
    output_dir : Directory where the dataset will be extracted.

    Returns
    -------
    True if download succeeded; False otherwise.
    """
    if not _kaggle_credentials_available():
        _print_manual_instructions(output_dir)
        return False

    # Import kaggle API (installed via requirements.txt or pip install kaggle)
    try:
        import kaggle
    except ImportError:
        print(
            "The 'kaggle' package is not installed.\n"
            "Install it with: pip install kaggle\n"
            "Or download manually — see instructions above."
        )
        _print_manual_instructions(output_dir)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SCB-Dataset to: {output_dir}")
    print(f"Dataset: {SCB_DATASET_SLUG}")

    try:
        # Download as a zip archive
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            SCB_DATASET_SLUG,
            path=str(output_dir),
            unzip=False,      # we handle extraction manually for progress reporting
            quiet=False,
        )
    except Exception as e:
        print(f"Download failed: {e}")
        _print_manual_instructions(output_dir)
        return False

    # Extract the zip
    zip_files = list(output_dir.glob("*.zip"))
    if not zip_files:
        print("Download complete but no .zip file found — dataset may already be extracted.")
        return _verify_scb(output_dir)

    zip_path = zip_files[0]
    print(f"Extracting {zip_path.name} …")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        zip_path.unlink()   # remove zip after extraction
        print("Extraction complete.")
    except zipfile.BadZipFile as e:
        print(f"Extraction failed: {e}")
        return False

    return _verify_scb(output_dir)


def _verify_scb(data_dir: Path) -> bool:
    """
    Verify that the SCB-Dataset directory contains the expected class folders
    or annotation files.  Print a summary and return True if usable.
    """
    print(f"\nVerifying SCB-Dataset at: {data_dir}")

    # Check for ImageFolder layout
    for split in ["train", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            found_classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
            print(f"  {split}/: {found_classes}")

    # Count images
    n_jpg = len(list(data_dir.rglob("*.jpg")))
    n_png = len(list(data_dir.rglob("*.png")))
    print(f"  Total images: {n_jpg + n_png} (.jpg: {n_jpg}, .png: {n_png})")

    # Check for YOLO annotation layout
    has_annotations = (data_dir / "annotations").exists() or (data_dir / "train.txt").exists()
    if has_annotations:
        print("  YOLO annotation files detected.")

    if n_jpg + n_png == 0 and not has_annotations:
        print(
            "\n  WARNING: No images or annotations found.  "
            "The dataset may be in a subdirectory.  "
            f"Check the contents of {data_dir} and adjust --output-dir accordingly."
        )
        return False

    print("\n  Dataset verified.  Ready to train:")
    print(f"  python3 -m model.train --data_dir {data_dir}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CED System — Dataset download and environment setup helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--download-scb", action="store_true",
        help="Download SCB-Dataset from Kaggle (requires credentials or manual fallback)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_SCB_DIR,
        help=f"Directory to extract the dataset (default: {DEFAULT_SCB_DIR})",
    )
    parser.add_argument(
        "--check-env", action="store_true",
        help="Print installed package versions and model checkpoint status",
    )
    args = parser.parse_args()

    if args.check_env:
        check_env()
    elif args.download_scb:
        success = download_scb(args.output_dir)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
