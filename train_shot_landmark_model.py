"""
train_shot_landmark_model.py
============================
Trains the landmark-based cricket shot type classifier.

Run from project root:
    python train_shot_landmark_model.py

What this does
--------------
1. Generates synthetic training data using real biomechanical ranges
   (cover drive has very different elbow/bat-plane/weight distribution
    than a pull shot — so the model can actually learn the difference)
2. Trains Random Forest + XGBoost + Ensemble with cross-validation
3. Saves the best model to models/shot_landmark_model.pkl
4. Saves the class list to models/shot_landmark_classes.json

Why this beats the ResNet18 image approach
------------------------------------------
ResNet18 learned background pixels (pitch, crowd, jersey colour).
This model learns BODY GEOMETRY:
  - Cover drive:     front-foot, high elbow, bat through covers (near vertical plane)
  - Cut shot:        back-foot, horizontal bat, wrist snap
  - Pull shot:       back-foot, horizontal bat, ducking head, rotate to leg side
  - Straight drive:  front-foot, high elbow, bat straight (vertical plane)
  - Leg glance:      minimal bat swing, close to body, wrist rotation

The synthetic ranges below come from cricket coaching biomechanics literature
and MediaPipe angle observations on real cricket videos.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH     = "models/shot_landmark_model.pkl"
CLASS_MAP_PATH = "models/shot_landmark_classes.json"
DATASET_PATH   = "data/shot_landmark_dataset.csv"

FEATURE_COLS = [
    "elbow_angle",       # degrees — swing mechanics
    "wrist_height",      # 0–1+ normalised to shoulder-hip span
    "hip_rotation",      # degrees — body rotation
    "knee_bend",         # degrees — front knee flexion (lower = more bent)
    "spine_lean",        # degrees — forward lean
    "foot_stance_width", # normalised to shoulder width
    "bat_plane",         # 0=horizontal, 90=vertical swing plane
    "weight_forward",    # 0=back foot, 1=front foot
]

CLASSES = [
    "Cover Drive",
    "Cut Shot",
    "Leg Glance",
    "Pull Shot",
    "Straight Drive",
]

# ── Biomechanical ranges per shot type ───────────────────────────────────────
# Each entry: (low, high) — uniform sample then add Gaussian noise
# Sources: coaching biomechanics + MediaPipe observation on real shots

# ── Ranges calibrated against real MediaPipe output from actual cricket video ─
# Cover Drive anchor (from debug_features.py on 8 real cover drive videos):
#   elbow_angle ~120  wrist_height ~0.57  hip_rotation ~20-40 (after fix)
#   knee_bend ~137    spine_lean ~30-50 (after fix)  bat_plane ~38
#   weight_forward ~0.3-0.6 (after fix)   foot_stance_width ~0.5-1.0
#
# Other shot ranges are offset from cover drive based on cricket biomechanics.
# Key separators between shots:
#   bat_plane  : Cut/Pull (5-30) < Cover/Leg (20-55) < Straight Drive (55-85)
#   weight_fwd : Pull/Cut (0.05-0.40) << Cover/Straight (0.40-0.75)
#   wrist_ht   : Cut (low) vs Pull (high) vs Leg Glance (very low)
#   knee_bend  : Pull back-foot (85-120) vs front-foot drives (120-170)

SHOT_PROFILES = {
    "Cover Drive": {
        # Real data anchor: elbow ~120, wrist_ht ~0.57, knee ~137, bat_plane ~38
        "elbow_angle":       (90, 155),
        "wrist_height":      (0.30, 0.90),
        "hip_rotation":      (15, 55),    # moderate body rotation toward off side
        "knee_bend":         (105, 165),  # front knee driving through
        "spine_lean":        (15, 55),    # leaning into the ball
        "foot_stance_width": (0.40, 1.50),
        "bat_plane":         (20, 58),    # angled toward covers, not straight
        "weight_forward":    (0.40, 0.75),# weight transferring to front foot
    },
    "Cut Shot": {
        # Back foot, horizontal bat, wrist low — very different bat_plane & weight
        "elbow_angle":       (70, 130),
        "wrist_height":      (0.15, 0.55),# wrist much lower than cover drive
        "hip_rotation":      (5, 35),     # less hip turn, more lateral movement
        "knee_bend":         (90, 145),   # back foot dominant
        "spine_lean":        (8, 40),
        "foot_stance_width": (0.30, 1.10),
        "bat_plane":         (5, 28),     # near-horizontal — key separator
        "weight_forward":    (0.05, 0.38),# weight back — key separator
    },
    "Leg Glance": {
        # Deflection, minimal bat swing, wrist very low, close to body
        "elbow_angle":       (95, 150),
        "wrist_height":      (0.05, 0.40),# lowest wrist — key separator
        "hip_rotation":      (8, 40),
        "knee_bend":         (100, 160),
        "spine_lean":        (10, 45),
        "foot_stance_width": (0.35, 1.20),
        "bat_plane":         (15, 50),    # mid-plane deflection
        "weight_forward":    (0.30, 0.65),
    },
    "Pull Shot": {
        # Short ball, horizontal swing, arms high, weight back, back knee bent
        "elbow_angle":       (75, 135),
        "wrist_height":      (0.60, 1.10),# wrist highest of all shots — key separator
        "hip_rotation":      (25, 70),    # strong rotation to leg side
        "knee_bend":         (80, 125),   # deep back-knee bend — key separator
        "spine_lean":        (8, 42),
        "foot_stance_width": (0.50, 1.60),
        "bat_plane":         (5, 28),     # very horizontal like cut — but weight back
        "weight_forward":    (0.05, 0.38),# weight back — same as cut
    },
    "Straight Drive": {
        # Front foot, bat very vertical (straight down the pitch), strong weight transfer
        "elbow_angle":       (105, 160),
        "wrist_height":      (0.50, 1.00),
        "hip_rotation":      (18, 52),
        "knee_bend":         (115, 168),
        "spine_lean":        (18, 58),    # strong forward lean
        "foot_stance_width": (0.45, 1.40),
        "bat_plane":         (58, 88),    # most vertical — key separator
        "weight_forward":    (0.55, 0.85),# strong front-foot weight — key separator
    },
}

NOISE_STD = {
    "elbow_angle":       5.0,
    "wrist_height":      0.05,
    "hip_rotation":      5.0,
    "knee_bend":         6.0,
    "spine_lean":        3.0,
    "foot_stance_width": 0.10,
    "bat_plane":         5.0,
    "weight_forward":    0.04,
}


# ── Synthetic dataset generator ───────────────────────────────────────────────

def generate_dataset(n_per_class: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for shot, profile in SHOT_PROFILES.items():
        for i in range(n_per_class):
            row = {"shot_type": shot}
            for feat, (lo, hi) in profile.items():
                val = rng.uniform(lo, hi) + rng.normal(0, NOISE_STD[feat])
                if feat == "wrist_height" or feat == "weight_forward":
                    val = float(np.clip(val, 0.0, 1.5))
                else:
                    val = float(np.clip(val, 0.0, 200.0))
                row[feat] = round(val, 4)
            rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(DATASET_PATH) if os.path.dirname(DATASET_PATH) else ".", exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"[train] Synthetic dataset ({len(df)} samples) → {DATASET_PATH}")
    print(df["shot_type"].value_counts().to_string())
    return df


# ── Model builders ────────────────────────────────────────────────────────────

def _make_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_split=4, random_state=42, n_jobs=-1
        )),
    ])


def _make_gb():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, random_state=42,
        )),
    ])


def _make_xgb():
    try:
        from xgboost import XGBClassifier
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    XGBClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=5, eval_metric="mlogloss",
                random_state=42,
            )),
        ])
    except ImportError:
        print("[train] XGBoost not installed — skipping XGB model")
        return None


# ── Main training ──────────────────────────────────────────────────────────────

def train():
    print("\n" + "="*60)
    print("  Cricket Shot Landmark Classifier — Training")
    print("="*60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report

    # 1 ── Dataset
    df = generate_dataset(n_per_class=600)  # more samples = more robust model
    X     = df[FEATURE_COLS].values
    y_str = df["shot_type"].values   # string labels e.g. "Cover Drive"

    # Encode strings → integers — XGBoost requires numeric labels
    le = LabelEncoder()
    y  = le.fit_transform(y_str)     # 0,1,2,3,4
    encoded_classes = list(le.classes_)

    print(f"\nFeatures : {FEATURE_COLS}")
    print(f"Classes  : {encoded_classes}")
    print(f"Samples  : {len(X)}\n")

    # 2 ── Cross-validation per model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {"RandomForest": _make_rf(), "GradientBoosting": _make_gb()}

    xgb = _make_xgb()
    if xgb is not None:
        models["XGBoost"] = xgb

    results = {}
    print(f"{'Model':<20} {'CV Acc':>8} {'±':>6}")
    print("-" * 38)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        results[name] = (scores.mean(), scores.std(), model)
        print(f"{name:<20} {scores.mean():.3f}   ±{scores.std():.3f}")

    # 3 ── Ensemble
    estimators = [(n, m) for n, (_, _, m) in results.items()]
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ens_scores = cross_val_score(ensemble, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    results["Ensemble"] = (ens_scores.mean(), ens_scores.std(), ensemble)
    print(f"{'Ensemble':<20} {ens_scores.mean():.3f}   ±{ens_scores.std():.3f}  ★")

    # 4 ── Pick best, fit on full data
    best_name  = max(results, key=lambda n: results[n][0])
    best_model = results[best_name][2]
    best_model.fit(X, y)

    print(f"\n[train] Best model: {best_name}  (CV acc {results[best_name][0]:.3f})")

    # 5 ── Save model, label encoder, and class map
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, MODEL_PATH.replace(".pkl", "_label_encoder.pkl"))
    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(encoded_classes, f, indent=2)

    print(f"[train] Model saved        → {MODEL_PATH}")
    print(f"[train] Label encoder saved→ {MODEL_PATH.replace('.pkl', '_label_encoder.pkl')}")
    print(f"[train] Classes saved      → {CLASS_MAP_PATH}")

    # 6 ── Per-class accuracy report
    preds     = best_model.predict(X)
    preds_str = le.inverse_transform(preds)
    print("\nIn-sample classification report:")
    print(classification_report(y_str, preds_str, target_names=encoded_classes))

    print("\n[train] Done. You can now run:  python main.py")
    print("        The landmark classifier will be loaded automatically.")

if __name__ == "__main__":
    train()