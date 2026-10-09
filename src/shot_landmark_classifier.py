"""
src/shot_landmark_classifier.py
================================
Upgraded 10-Feature Landmark-Based Cricket Shot Type Classifier.
Extracts 10 biomechanical features from MediaPipe 3D joint positions to accurately
discriminate between subtle stroke variations (e.g. Straight Drive vs Cover Drive).

Features extracted:
------------------
  1. elbow_angle             — Lead elbow angle (swing mechanics & elevation)
  2. wrist_height            — Normalised wrist height relative to shoulder-hip span
  3. hip_rotation            — Horizontal hip-to-shoulder twist (rotation plane)
  4. knee_bend               — Front knee flexion (weight transfer & stride)
  5. spine_lean              — Lean angle of the spine from vertical plumb
  6. foot_stance_width       — Normalised feet separation
  7. bat_plane               — Estimated swing plane angle (horizontal=0, vertical=90)
  8. weight_forward          — Ratio of weight shifted forward (0=back foot, 1=front foot)
  9. wrist_centerline_offset — Lateral distance between wrist and head/nose center line
                             (Straight Drive: < 0.06; Cover Drive: > 0.12)
  10. foot_stride_angle      — Stride direction angle (0° = straight down pitch, 45° = extra cover)
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import joblib
import numpy as np

MODEL_PATH     = "models/shot_landmark_model.pkl"
CLASS_MAP_PATH = "models/shot_landmark_classes.json"

LOW_CONFIDENCE_THRESHOLD = 0.40

# MediaPipe landmark indices
_MP = {
    "nose":           0,
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
    "left_heel":      29, "right_heel":     30,
    "left_foot":      31, "right_foot":     32,
}

FEATURE_COLS = [
    "elbow_angle",
    "wrist_height",
    "hip_rotation",
    "knee_bend",
    "spine_lean",
    "foot_stance_width",
    "bat_plane",
    "weight_forward",
    "wrist_centerline_offset",
    "foot_stride_angle",
]


def _angle(a, b, c) -> float:
    """Angle (degrees) at vertex b, between rays b->a and b->c."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = (math.hypot(*ba) * math.hypot(*bc)) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _pt(lm, idx) -> tuple[float, float]:
    """Return (x, y) for a landmark index, normalised to [0,1]."""
    l = lm[idx]
    return (l.x, l.y)


def extract_features(landmarks) -> Optional[dict]:
    """
    Extract 10 discriminating biomechanical features from a MediaPipe landmark list.
    """
    if landmarks is None or len(landmarks) < 33:
        return None

    try:
        nose = _pt(landmarks, _MP["nose"])
        ls = _pt(landmarks, _MP["left_shoulder"])
        rs = _pt(landmarks, _MP["right_shoulder"])
        le = _pt(landmarks, _MP["left_elbow"])
        re = _pt(landmarks, _MP["right_elbow"])
        lw = _pt(landmarks, _MP["left_wrist"])
        rw = _pt(landmarks, _MP["right_wrist"])
        lh = _pt(landmarks, _MP["left_hip"])
        rh = _pt(landmarks, _MP["right_hip"])
        lk = _pt(landmarks, _MP["left_knee"])
        rk = _pt(landmarks, _MP["right_knee"])
        la = _pt(landmarks, _MP["left_ankle"])
        ra = _pt(landmarks, _MP["right_ankle"])
        lf = _pt(landmarks, _MP["left_foot"])
        rf = _pt(landmarks, _MP["right_foot"])

        # Determine dominant lead side (based on wrist extension)
        left_ext  = abs(lw[0] - ls[0])
        right_ext = abs(rw[0] - rs[0])
        if left_ext >= right_ext:
            shoulder, elbow, wrist = ls, le, lw
            front_ankle, front_foot = la, lf
        else:
            shoulder, elbow, wrist = rs, re, rw
            front_ankle, front_foot = ra, rf

        # ── Feature 1: elbow_angle ───────────────────────────────────────────
        elbow_angle = _angle(shoulder, elbow, wrist)

        # ── Feature 2: wrist_height ──────────────────────────────────────────
        hip_mid_y = (lh[1] + rh[1]) / 2
        sh_mid_y  = (ls[1] + rs[1]) / 2
        span      = abs(hip_mid_y - sh_mid_y) + 1e-9
        wrist_height = float(np.clip(1.0 - (wrist[1] - sh_mid_y) / span, 0.0, 1.5))

        # ── Feature 3: hip_rotation ──────────────────────────────────────────
        sh_angle  = math.degrees(math.atan2(rs[1]-ls[1], rs[0]-ls[0]))
        hip_angle = math.degrees(math.atan2(rh[1]-lh[1], rh[0]-lh[0]))
        diff = abs(sh_angle - hip_angle) % 360
        hip_rotation = min(diff, 360 - diff)

        # ── Feature 4: knee_bend ─────────────────────────────────────────────
        left_knee_angle  = _angle(lh, lk, la)
        right_knee_angle = _angle(rh, rk, ra)
        knee_bend = min(left_knee_angle, right_knee_angle)

        # ── Feature 5: spine_lean ────────────────────────────────────────────
        sh_mid  = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        hip_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
        dx = sh_mid[0] - hip_mid[0]
        dy = sh_mid[1] - hip_mid[1]
        spine_lean = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))

        # ── Feature 6: foot_stance_width ─────────────────────────────────────
        sh_width     = abs(rs[0] - ls[0]) + 1e-9
        stance_width = float(np.clip(abs(ra[0] - la[0]) / sh_width, 0.0, 3.0))

        # ── Feature 7: bat_plane ─────────────────────────────────────────────
        dx2 = wrist[0] - elbow[0]
        dy2 = wrist[1] - elbow[1]
        bat_plane = abs(math.degrees(math.atan2(abs(dy2), abs(dx2) + 1e-9)))

        # ── Feature 8: weight_forward ────────────────────────────────────────
        front_knee = min(left_knee_angle, right_knee_angle)
        weight_forward = float(np.clip(1.0 - (front_knee - 90.0) / 90.0, 0.0, 1.0))

        # ── Feature 9: wrist_centerline_offset (KEY DISCRIMINATOR) ──────────
        # Straight Drive hands pass directly through body centerline (nose/sternum)
        # Cover Drive reaches laterally out towards the off side
        wrist_centerline_offset = float(abs(wrist[0] - nose[0]))

        # ── Feature 10: foot_stride_angle (KEY DISCRIMINATOR) ───────────────
        # Measures front foot angle relative to vertical pitch line
        dx_foot = front_foot[0] - front_ankle[0]
        dy_foot = front_foot[1] - front_ankle[1]
        foot_stride_angle = abs(math.degrees(math.atan2(abs(dx_foot), abs(dy_foot) + 1e-9)))

        return {
            "elbow_angle":             float(elbow_angle),
            "wrist_height":            float(wrist_height),
            "hip_rotation":            float(hip_rotation),
            "knee_bend":               float(knee_bend),
            "spine_lean":              float(spine_lean),
            "foot_stance_width":       float(stance_width),
            "bat_plane":               float(bat_plane),
            "weight_forward":          float(weight_forward),
            "wrist_centerline_offset": float(wrist_centerline_offset),
            "foot_stride_angle":       float(foot_stride_angle),
        }

    except Exception as e:
        return None


class ShotLandmarkClassifier:
    """
    10-Feature Landmark-based shot type classifier.
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Shot landmark model not found at '{MODEL_PATH}'.\n"
                "Run:  python train_shot_landmark_model.py  first."
            )

        self.model   = joblib.load(MODEL_PATH)
        self.classes = self._load_classes()

        le_path = MODEL_PATH.replace(".pkl", "_label_encoder.pkl")
        self.label_encoder = joblib.load(le_path) if os.path.exists(le_path) else None

        print(f"[ShotLandmarkClassifier] Loaded model  |  classes: {self.classes}")

    def _load_classes(self) -> list[str]:
        if os.path.exists(CLASS_MAP_PATH):
            with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return ["Cover Drive", "Cut Shot", "Leg Glance", "Pull Shot", "Straight Drive"]

    def predict_from_landmarks(self, all_landmarks: list) -> dict:
        features_list = []
        for lm in all_landmarks:
            f = extract_features(lm)
            if f is not None:
                features_list.append([f[k] for k in FEATURE_COLS])

        if not features_list:
            return {
                "shot_type":        "Unknown",
                "confidence":       0.0,
                "all_probabilities": {c: 0.0 for c in self.classes},
                "low_confidence":   True,
                "mean_features":    {},
            }

        X = np.mean(features_list, axis=0).reshape(1, -1)
        mean_features = dict(zip(FEATURE_COLS, X[0].tolist()))

        return self._predict_array(X, mean_features)

    def predict_from_features(self, feature_dict: dict) -> dict:
        X = np.array([[feature_dict.get(k, 0.0) for k in FEATURE_COLS]])
        return self._predict_array(X, feature_dict)

    def _predict_array(self, X: np.ndarray, mean_features: dict) -> dict:
        # Compatibility check: if old model trained on 8 features, slice X
        if hasattr(self.model, "n_features_in_") and self.model.n_features_in_ == 8 and X.shape[1] > 8:
            X = X[:, :8]

        proba = self.model.predict_proba(X)[0]
        idx   = int(np.argmax(proba))

        confidence     = float(proba[idx])
        low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

        if self.label_encoder is not None:
            shot_name = self.label_encoder.inverse_transform([idx])[0]
            class_names = list(self.label_encoder.classes_)
        else:
            shot_name   = self.classes[idx]
            class_names = self.classes

        return {
            "shot_type":        shot_name,
            "confidence":       round(confidence, 4),
            "all_probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(class_names, proba)
            },
            "low_confidence":  low_confidence,
            "mean_features":   {k: round(v, 3) for k, v in mean_features.items()},
        }