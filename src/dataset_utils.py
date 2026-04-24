"""
dataset_utils.py
================
Dataset preparation utilities for the cricket cover drive ML pipeline.

Functions
---------
  generate_synthetic_dataset  — create a labeled CSV from biomechanical rules
  load_dataset                — load a CSV and return X, y arrays
  export_features_from_video  — extract raw feature rows from an analyzed video
  describe_dataset            — print summary stats for a dataset

The synthetic generator uses realistic biomechanical ranges from config.py
but adds gaussian noise so each generated sample is unique — giving the ML
models enough variance to generalise without trivially memorising rules.

If you have real labeled videos, export features with export_features_from_video()
and label each row manually (Poor=0, Average=1, Good=2) in the CSV.  Then pass
the CSV to load_dataset() and you're ready to call MLTrainer.train_all().
"""

from __future__ import annotations

import os
import csv
import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
FEATURE_COLS  = ["elbow_angle", "spine_lean", "head_over_knee", "foot_direction"]
LABEL_COL     = "label"
CLASS_MAP     = {0: "Poor", 1: "Average", 2: "Good"}
CLASS_MAP_INV = {"Poor": 0, "Average": 1, "Good": 2}


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic dataset generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_samples: int = 600,
    seed: int = 42,
    output_path: str = "data/shot_quality_dataset.csv",
    noise_scale: float = 1.5,
) -> pd.DataFrame:
    """
    Generate a synthetic, labeled CSV dataset from biomechanical domain knowledge.

    This is the recommended starting point when no labeled video data exists.
    After generation, a domain expert (coach) can inspect the CSV and correct
    or add labels before re-training the model.

    Feature distributions per class
    --------------------------------
    Good    — metrics close to ideal biomechanical ranges
    Average — metrics near the boundary / partially correct
    Poor    — metrics clearly outside optimal ranges

    Parameters
    ----------
    n_samples   : total number of samples (split equally among classes)
    seed        : random seed for reproducibility
    output_path : where to save the CSV
    noise_scale : Gaussian noise std applied to each feature for diversity

    Returns
    -------
    pd.DataFrame with columns: video_id, frame_index, elbow_angle,
                                spine_lean, head_over_knee, foot_direction, label
    """
    rng = np.random.default_rng(seed)
    per_class = n_samples // 3
    rows = []

    # ── Class 2 — Good shots ─────────────────────────────────────────────────
    for i in range(per_class):
        rows.append({
            "video_id":       f"synthetic_good_{i:04d}",
            "frame_index":    i,
            "elbow_angle":    _noisy(rng, 100, 155, noise_scale),
            "spine_lean":     _noisy(rng, 82,   98, noise_scale),
            "head_over_knee": _noisy_float(rng, 0.00, 0.06, 0.005),
            "foot_direction": _noisy(rng, 82,   98, noise_scale),
            LABEL_COL:        "Good",
        })

    # ── Class 1 — Average shots ───────────────────────────────────────────────
    for i in range(per_class):
        rows.append({
            "video_id":       f"synthetic_avg_{i:04d}",
            "frame_index":    i,
            "elbow_angle":    _noisy(rng, 75,  110, noise_scale),
            "spine_lean":     _noisy(rng, 73,   85, noise_scale),
            "head_over_knee": _noisy_float(rng, 0.05, 0.15, 0.008),
            "foot_direction": _noisy(rng, 70,   88, noise_scale),
            LABEL_COL:        "Average",
        })

    # ── Class 0 — Poor shots ─────────────────────────────────────────────────
    remaining = n_samples - 2 * per_class
    for i in range(remaining):
        rows.append({
            "video_id":       f"synthetic_poor_{i:04d}",
            "frame_index":    i,
            "elbow_angle":    _noisy(rng, 25,   75, noise_scale),
            "spine_lean":     _noisy(rng, 45,   72, noise_scale),
            "head_over_knee": _noisy_float(rng, 0.15, 0.40, 0.010),
            "foot_direction": _noisy(rng, 35,   70, noise_scale),
            LABEL_COL:        "Poor",
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[dataset_utils] Synthetic dataset ({len(df)} samples) saved → {output_path}")
    print(df[LABEL_COL].value_counts().to_string())
    return df


def _noisy(rng, lo, hi, std):
    """Uniform sample in [lo,hi] + Gaussian noise, clipped to ±10% of range."""
    val = rng.uniform(lo, hi)
    return float(np.clip(val + rng.normal(0, std), lo - 10, hi + 10))


def _noisy_float(rng, lo, hi, std):
    val = rng.uniform(lo, hi)
    return float(np.clip(val + rng.normal(0, std), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  CSV → numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    csv_path: str,
    feature_cols: list[str] | None = None,
    label_col: str = LABEL_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a labeled CSV and return (X, y) arrays.

    The label column may contain strings ("Poor", "Average", "Good") or
    integers (0, 1, 2).  Strings are kept as-is; MLTrainer handles encoding.

    Parameters
    ----------
    csv_path     : path to the CSV file
    feature_cols : list of column names to use as features (default: all 4)
    label_col    : name of the target column

    Returns
    -------
    X : np.ndarray, shape (n, 4)
    y : np.ndarray of str labels, shape (n,)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = pd.read_csv(csv_path)

    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df = df.dropna(subset=feature_cols + [label_col])

    X = df[feature_cols].values.astype(float)
    y = df[label_col].values

    # Convert int labels to string if needed
    try:
        if np.issubdtype(y.dtype, np.integer):
            y = np.array([CLASS_MAP[int(v)] for v in y])
        else:
            y = y.astype(str)
    except TypeError:
        # Pandas StringDtype or other extension types — just cast to str
        y = y.astype(str)

    print(f"[dataset_utils] Loaded {len(X)} samples from '{csv_path}'")
    label_counts = pd.Series(y).value_counts()
    print(label_counts.to_string())
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
#  Export features from a processed video
# ─────────────────────────────────────────────────────────────────────────────

def export_features_from_video(
    all_metrics: list[dict],
    video_id: str,
    output_path: str = "data/shot_quality_dataset.csv",
    label: str = "",
) -> None:
    """
    Append frame-level features from an analyzed video to a CSV dataset file.

    After running analyze_video() you have a list of per-frame metrics dicts.
    Call this function to persist them for future training.  Leave `label`
    empty — fill it manually in the CSV, then re-train.

    Parameters
    ----------
    all_metrics : list of dicts from biomechanical_metrics.calculate_metrics()
    video_id    : string identifier for this video (e.g. filename stem)
    output_path : CSV file to append to (created if absent)
    label       : optional quality label — fill in manually if not known yet
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    write_header = not os.path.exists(output_path)

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "frame_index"] + FEATURE_COLS + [LABEL_COL]
        )
        if write_header:
            writer.writeheader()

        for idx, m in enumerate(all_metrics):
            if m is None:
                continue
            writer.writerow({
                "video_id":       video_id,
                "frame_index":    idx,
                "elbow_angle":    round(m.get("elbow_angle", 0), 4),
                "spine_lean":     round(m.get("spine_lean", 0), 4),
                "head_over_knee": round(m.get("head_over_knee", 0), 6),
                "foot_direction": round(m.get("foot_direction", 0), 4),
                LABEL_COL:        label,
            })

    print(f"[dataset_utils] Exported {len(all_metrics)} frames for '{video_id}' → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Describe dataset
# ─────────────────────────────────────────────────────────────────────────────

def describe_dataset(csv_path: str) -> None:
    """Print a statistical summary of a dataset CSV."""
    df = pd.read_csv(csv_path)
    print(f"\n{'='*55}")
    print(f"  Dataset: {csv_path}  ({len(df)} rows)")
    print(f"{'='*55}")
    print(df[FEATURE_COLS].describe().round(3).to_string())
    print("\nClass distribution:")
    print(df[LABEL_COL].value_counts().to_string())
    print("=" * 55 + "\n")