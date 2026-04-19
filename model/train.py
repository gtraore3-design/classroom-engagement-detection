"""
train.py — Training script for the SCB-Dataset behavior classification model.

Primary dataset: SCB-Dataset (Student Classroom Behavior)
    Mohanta, A. et al. (2023). arXiv:2304.02488
    Kaggle: kaggle.com/datasets/asthalochanmohanta/class-room-student-behaviour

Download the dataset first:
    python setup.py --download-scb          (requires Kaggle credentials)
    OR see setup.py for manual instructions.

Usage (from project root):
    python -m model.train --data_dir /path/to/scb_dataset --epochs 40

Expected dataset layout
-----------------------
The SCB-Dataset may arrive in one of two layouts.  Both are supported:

  Layout A — ImageFolder (one subdirectory per class):
      scb_dataset/
          train/
              hand_raising/    paying_attention/    writing/
              distracted/      bored/               phone_use/
          test/
              hand_raising/    paying_attention/    ...

  Layout B — Annotation file (YOLO or CSV with bounding boxes):
      scb_dataset/
          images/
              img001.jpg  img002.jpg  ...
          annotations/
              img001.txt  img002.txt  ...   (YOLO format: class cx cy w h)
          classes.txt   (one class name per line)
          train.txt     (list of training image paths)
          test.txt      (list of test image paths)

FER2013 baseline training
-------------------------
The original FER2013 emotion model can still be trained by running:
    python -m model.train_fer --data_dir /path/to/fer2013 --epochs 30
(kept in train_fer.py for backward compatibility)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SCB_BEHAVIOR_CLASSES,
    SCB_IMG_SIZE,
    SCB_BATCH_SIZE,
    SCB_EPOCHS,
)
from model.behavior_model import (
    MODEL_PATH,
    build_scb_model,
    unfreeze_top_layers,
)

NUM_CLASSES = len(SCB_BEHAVIOR_CLASSES)

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_imagefolder(data_dir: Path) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
    """
    Load SCB data from an ImageFolder-style directory structure.

    Images are read, resized to SCB_IMG_SIZE (96×96), normalised to [0,255]
    float32 (preprocessing to [-1,1] is done inside the model graph).

    Parameters
    ----------
    data_dir : Root directory containing ``train/`` and ``test/`` subdirs.

    Returns
    -------
    (X_train, y_train, X_test, y_test) — uint8 arrays and integer labels.
    """
    def _scan(split: str):
        images, labels = [], []
        split_dir = data_dir / split
        if not split_dir.exists():
            return np.array([]), np.array([])

        for idx, cls in enumerate(SCB_BEHAVIOR_CLASSES):
            # Try exact class name, then lowercase, then replace underscores
            candidates = [cls, cls.lower(), cls.replace("_", ""),
                          cls.replace("_", "-")]
            cls_dir = None
            for name in candidates:
                d = split_dir / name
                if d.exists():
                    cls_dir = d
                    break
            if cls_dir is None:
                print(f"  [warn] No directory found for class '{cls}' in {split_dir}")
                continue

            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                try:
                    raw = tf.io.read_file(str(img_path))
                    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
                    img = tf.image.resize(img, SCB_IMG_SIZE)
                    images.append(img.numpy().astype(np.float32))
                    labels.append(idx)
                except Exception as e:
                    print(f"  [warn] Could not load {img_path}: {e}")

        if not images:
            return np.array([]), np.array([])
        return np.array(images), np.array(labels)

    X_tr, y_tr = _scan("train")
    X_te, y_te = _scan("test")
    return X_tr, y_tr, X_te, y_te


def _load_yolo_annotations(data_dir: Path) -> tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
    """
    Load SCB data from YOLO-format annotations.

    Expects:
      - data_dir/images/       — full classroom images
      - data_dir/annotations/  — .txt files (one line per student bbox)
      - data_dir/train.txt     — paths of training images (one per line)
      - data_dir/test.txt      — paths of test images
      - data_dir/classes.txt   — class names, one per line (must match SCB_BEHAVIOR_CLASSES)

    Each annotation line: ``class_id cx cy width height``
    (YOLO normalised format, relative to image dimensions)

    Students are cropped from the full image using the bounding box,
    then resized to 96×96.

    Parameters
    ----------
    data_dir : Root directory of the YOLO-annotated dataset.

    Returns
    -------
    (X_train, y_train, X_test, y_test)
    """
    classes_file = data_dir / "classes.txt"
    if classes_file.exists():
        # Remap from dataset class order to SCB_BEHAVIOR_CLASSES order
        dataset_classes = [l.strip() for l in classes_file.read_text().splitlines() if l.strip()]
    else:
        dataset_classes = SCB_BEHAVIOR_CLASSES

    def _label_remap(dataset_idx: int) -> int | None:
        """Map dataset class index → SCB_BEHAVIOR_CLASSES index."""
        name = dataset_classes[dataset_idx] if dataset_idx < len(dataset_classes) else None
        if name is None:
            return None
        # Normalise: lowercase, strip spaces/hyphens
        norm = name.lower().replace("-", "_").replace(" ", "_")
        for i, scb_cls in enumerate(SCB_BEHAVIOR_CLASSES):
            if scb_cls.lower() == norm:
                return i
        return None

    def _scan_split(split_file: Path):
        images, labels = [], []
        if not split_file.exists():
            return np.array([]), np.array([])

        for rel_path in split_file.read_text().splitlines():
            rel_path = rel_path.strip()
            if not rel_path:
                continue
            img_path  = data_dir / rel_path
            ann_path  = (data_dir / "annotations" /
                         img_path.with_suffix(".txt").name)
            if not img_path.exists() or not ann_path.exists():
                continue

            try:
                raw  = tf.io.read_file(str(img_path))
                full = tf.image.decode_image(raw, channels=3,
                                              expand_animations=False).numpy()
                ih, iw = full.shape[:2]
            except Exception as e:
                print(f"  [warn] Could not load {img_path}: {e}")
                continue

            for line in ann_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # Convert normalised YOLO → pixel coords
                x1 = int((cx - bw / 2) * iw)
                y1 = int((cy - bh / 2) * ih)
                x2 = int((cx + bw / 2) * iw)
                y2 = int((cy + bh / 2) * ih)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop      = full[y1:y2, x1:x2]
                crop_resized = tf.image.resize(crop, SCB_IMG_SIZE).numpy().astype(np.float32)
                scb_idx   = _label_remap(cls_id)
                if scb_idx is None:
                    continue

                images.append(crop_resized)
                labels.append(scb_idx)

        if not images:
            return np.array([]), np.array([])
        return np.array(images), np.array(labels)

    X_tr, y_tr = _scan_split(data_dir / "train.txt")
    X_te, y_te = _scan_split(data_dir / "test.txt")
    return X_tr, y_tr, X_te, y_te


def load_scb_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
    """
    Auto-detect dataset layout and load SCB train/test arrays.

    Tries ImageFolder layout first, then YOLO annotation layout.

    Parameters
    ----------
    data_dir : Root path of the SCB dataset.

    Returns
    -------
    (X_train, y_train, X_test, y_test)
        X arrays: float32, shape (N, 96, 96, 3), values in [0, 255]
        y arrays: int,     shape (N,), values in {0, …, 5}
    """
    data_dir = Path(data_dir)
    print(f"Loading SCB-Dataset from: {data_dir}")

    # Try ImageFolder
    if (data_dir / "train").exists():
        print("  Detected ImageFolder layout.")
        X_tr, y_tr, X_te, y_te = _load_imagefolder(data_dir)
        if len(X_tr) > 0:
            return X_tr, y_tr, X_te, y_te

    # Try YOLO annotation layout
    if (data_dir / "train.txt").exists() or (data_dir / "annotations").exists():
        print("  Detected YOLO annotation layout.")
        X_tr, y_tr, X_te, y_te = _load_yolo_annotations(data_dir)
        if len(X_tr) > 0:
            return X_tr, y_tr, X_te, y_te

    raise FileNotFoundError(
        f"Could not find SCB dataset under {data_dir}.\n"
        "Expected layout:\n"
        "  Option A: data_dir/train/<class_name>/*.jpg\n"
        "  Option B: data_dir/images/*.jpg + data_dir/annotations/*.txt + train.txt\n"
        "Run setup.py to download the dataset from Kaggle."
    )


# ---------------------------------------------------------------------------
# tf.data pipeline with augmentation
# ---------------------------------------------------------------------------

def _build_dataset(X: np.ndarray, y: np.ndarray, augment: bool) -> tf.data.Dataset:
    """
    Construct an augmented tf.data.Dataset for one split.

    Augmentations (training only):
        Horizontal flip — behaviour classes are mirror-symmetric
        Brightness jitter ±20 % — accommodates variable classroom lighting
        Contrast jitter ±20 % — handles camera exposure differences
        Small random crop + resize — adds robustness to bounding-box imprecision

    Parameters
    ----------
    X       : float32 array, shape (N, 96, 96, 3), pixel values [0, 255]
    y       : int array, shape (N,)
    augment : If True apply data augmentation; False for validation / test.

    Returns
    -------
    tf.data.Dataset yielding (image, one_hot_label) batches.
    """
    ds = tf.data.Dataset.from_tensor_slices(
        (X, tf.one_hot(y, NUM_CLASSES))
    )

    if augment:
        ds = ds.shuffle(buffer_size=min(len(X), 5000), reshuffle_each_iteration=True)
        ds = ds.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(SCB_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def _augment_fn(img: tf.Tensor, lbl: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply random augmentations to a single training image."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.20 * 255)
    img = tf.image.random_contrast(img, lower=0.80, upper=1.20)
    # Random crop: pad by 8 px, then crop back to 96×96
    img = tf.image.resize_with_crop_or_pad(img, 104, 104)
    img = tf.image.random_crop(img, [*SCB_IMG_SIZE, 3])
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, lbl


# ---------------------------------------------------------------------------
# Class-weight computation (handles SCB class imbalance)
# ---------------------------------------------------------------------------

def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Compute inverse-frequency class weights to mitigate SCB class imbalance.

    The SCB-Dataset is known to have more ``paying_attention`` samples than
    ``hand_raising`` or ``phone_use`` samples.  Without weighting, the model
    would optimise toward the majority class.

    Formula: weight_i = N_total / (N_classes × count_i)
    """
    counts  = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    n_total = len(y)
    weights = {}
    for i, count in enumerate(counts):
        weights[i] = (n_total / (NUM_CLASSES * count)) if count > 0 else 1.0
    return weights


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    epochs: int = SCB_EPOCHS,
    output: Path = MODEL_PATH,
) -> keras.Model:
    """
    Full two-phase transfer-learning training on the SCB-Dataset.

    Phase 1 — head only (backbone frozen, Adam lr=1e-3, epochs//2)
    Phase 2 — top 30 backbone layers unfrozen (Adam lr=1e-4, epochs//2)

    Parameters
    ----------
    data_dir : Path to the SCB dataset root.
    epochs   : Total number of epochs (split equally across phases).
    output   : Path to save the trained .keras checkpoint.

    Returns
    -------
    Trained keras.Model
    """
    X_tr, y_tr, X_te, y_te = load_scb_dataset(data_dir)
    print(f"  Train samples: {len(X_tr)}  |  Test samples: {len(X_te)}")
    print(f"  Class distribution (train): "
          f"{dict(zip(SCB_BEHAVIOR_CLASSES, np.bincount(y_tr, minlength=NUM_CLASSES)))}")

    train_ds = _build_dataset(X_tr, y_tr, augment=True)
    test_ds  = _build_dataset(X_te, y_te, augment=False)

    class_weights = _compute_class_weights(y_tr)
    print(f"  Class weights: {class_weights}")

    # ---- Phase 1: head only ----
    model = build_scb_model(dropout=0.4)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    phase1_epochs = max(epochs // 2, 10)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True,
                                       monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3,
                                           monitor="val_loss", min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(str(output) + ".phase1.keras",
                                        save_best_only=True,
                                        monitor="val_accuracy"),
    ]
    print(f"\n=== Phase 1: training head ({phase1_epochs} epochs) ===")
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=phase1_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ---- Phase 2: fine-tune top backbone layers ----
    unfreeze_top_layers(model, n_layers=30)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    phase2_epochs = max(epochs - phase1_epochs, 10)
    callbacks[2] = keras.callbacks.ModelCheckpoint(str(output),
                                                    save_best_only=True,
                                                    monitor="val_accuracy")
    print(f"\n=== Phase 2: fine-tuning top layers ({phase2_epochs} epochs) ===")
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=phase2_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(str(output))
    print(f"\nModel saved → {output}")

    # Clean up phase-1 checkpoint
    phase1_ckpt = Path(str(output) + ".phase1.keras")
    if phase1_ckpt.exists():
        phase1_ckpt.unlink()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SCB-Dataset behavior classification model"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to SCB dataset root (see setup.py to download)",
    )
    parser.add_argument("--epochs", type=int, default=SCB_EPOCHS)
    parser.add_argument("--output", default=str(MODEL_PATH))
    args = parser.parse_args()
    train(args.data_dir, args.epochs, Path(args.output))
