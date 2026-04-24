"""
ml_model.py
===========
Supervised Machine-Learning module for Cricket Cover Drive Shot Quality Analysis.

Replaces the previous fuzzy-logic / hand-coded soft-computing approach with
a proper ML pipeline featuring:
  - SVM (Support Vector Machine)
  - Random Forest
  - Gradient Boosting (scikit-learn)
  - XGBoost
  - Voting Ensemble (soft voting over all four)

Each model is trained, cross-validated, and compared. The best model (or the
ensemble) is saved to disk via joblib for fast inference at video-analysis time.

Classes
-------
  MLTrainer    — training, evaluation, model comparison, model saving
  MLPredictor  — load a saved model and run inference on new frame/video data

Usage (training)
----------------
    from src.ml_model import MLTrainer
    from src.dataset_utils import load_dataset

    X, y = load_dataset("data/shot_quality_dataset.csv")
    trainer = MLTrainer()
    trainer.train_all(X, y)
    trainer.save_best(path="models/best_model.pkl")

Usage (inference)
-----------------
    from src.ml_model import MLPredictor

    predictor = MLPredictor(model_path="models/best_model.pkl")
    result = predictor.predict_from_metrics(metrics_dict)
    # result → {"label": "Good", "probabilities": {"Poor":0.05,"Average":0.10,"Good":0.85}}
"""

from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
CLASSES       = ["Poor", "Average", "Good"]
FEATURE_NAMES = ["elbow_angle", "spine_lean", "head_over_knee", "foot_direction"]
RANDOM_STATE  = 42


# ─────────────────────────────────────────────────────────────────────────────
#  MLTrainer
# ─────────────────────────────────────────────────────────────────────────────

class MLTrainer:
    """
    Trains, evaluates, and compares four ML models on biomechanical features.

    The four models are wrapped in scikit-learn Pipelines that include
    StandardScaler (required by SVM; harmless for tree methods).

    Attributes
    ----------
    models_       : dict  — trained Pipeline objects keyed by model name
    cv_scores_    : dict  — mean ± std cross-validation accuracy per model
    test_reports_ : dict  — full classification_report per model
    best_name_    : str   — name of the best-performing model on the test set
    label_encoder_: LabelEncoder — fitted encoder (str ↔ int)
    """

    def __init__(self, cv_folds: int = 5, test_size: float = 0.2):
        self.cv_folds  = cv_folds
        self.test_size = test_size
        self.models_        = {}
        self.cv_scores_     = {}
        self.test_reports_  = {}
        self.best_name_     = None
        self.label_encoder_ = LabelEncoder().fit(CLASSES)

    # ── Model definitions ────────────────────────────────────────────────────

    def _build_pipelines(self) -> dict:
        """Return a dict of name → sklearn Pipeline (un-fitted)."""
        label_map = {c: i for i, c in enumerate(CLASSES)}

        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, random_state=RANDOM_STATE)),
        ])

        rf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, max_depth=None,
                                              min_samples_split=4,
                                              random_state=RANDOM_STATE,
                                              n_jobs=-1)),
        ])

        gb = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=200,
                                                  learning_rate=0.05,
                                                  max_depth=4,
                                                  random_state=RANDOM_STATE)),
        ])

        xgb = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    XGBClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=4,
                                     eval_metric="mlogloss",
                                     random_state=RANDOM_STATE,
                                     n_jobs=-1)),
        ])

        # Soft-voting ensemble over all four
        ensemble = VotingClassifier(
            estimators=[("svm", svm), ("rf", rf), ("gb", gb), ("xgb", xgb)],
            voting="soft",
            n_jobs=-1,
        )

        return {
            "SVM":              svm,
            "RandomForest":     rf,
            "GradientBoosting": gb,
            "XGBoost":          xgb,
            "Ensemble":         ensemble,
        }

    # ── Train & evaluate ─────────────────────────────────────────────────────

    def train_all(self, X: np.ndarray, y: np.ndarray,
                  verbose: bool = True) -> None:
        """
        Train all models, run CV, hold-out evaluation, and pick the best.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 4)
            Feature matrix: [elbow_angle, spine_lean, head_over_knee, foot_direction]
        y : np.ndarray or list of str
            Class labels: "Poor" | "Average" | "Good"
        verbose : bool
            Print training progress and results table.
        """
        # Encode string labels → int for XGBoost compatibility
        if isinstance(y[0], str):
            y_enc = self.label_encoder_.transform(y)
        else:
            y_enc = np.asarray(y, dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=self.test_size,
            stratify=y_enc, random_state=RANDOM_STATE
        )

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=RANDOM_STATE)

        pipelines = self._build_pipelines()
        best_acc  = -1.0

        if verbose:
            print("\n" + "=" * 60)
            print("  ML TRAINING PIPELINE — Cricket Cover Drive Analysis")
            print("=" * 60)
            print(f"  Dataset  : {len(X)} samples  |  "
                  f"Train: {len(X_train)}  Test: {len(X_test)}")
            print(f"  Features : {FEATURE_NAMES}")
            print(f"  Classes  : {CLASSES}")
            print("-" * 60)

        for name, pipe in pipelines.items():
            if verbose:
                print(f"\n  Training {name} …", end=" ", flush=True)

            # 5-fold cross-validation on training data
            cv_accs = cross_val_score(pipe, X_train, y_train,
                                      cv=cv, scoring="accuracy", n_jobs=-1)
            self.cv_scores_[name] = {
                "mean": float(cv_accs.mean()),
                "std":  float(cv_accs.std()),
            }

            # Final fit on full training split
            pipe.fit(X_train, y_train)
            self.models_[name] = pipe

            # Hold-out test evaluation
            y_pred = pipe.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)

            # Classification report (with decoded string labels)
            report = classification_report(
                y_test, y_pred,
                target_names=CLASSES,
                output_dict=True,
                zero_division=0,
            )
            self.test_reports_[name] = {
                "accuracy":  test_acc,
                "report":    report,
                "confusion": confusion_matrix(y_test, y_pred).tolist(),
            }

            if test_acc > best_acc:
                best_acc       = test_acc
                self.best_name_ = name

            if verbose:
                print(f"CV acc={cv_accs.mean():.3f}±{cv_accs.std():.3f}  "
                      f"Test acc={test_acc:.3f}")

        if verbose:
            print("\n" + "-" * 60)
            print(f"  ★  Best model: {self.best_name_}  "
                  f"(test accuracy = {best_acc:.3f})")
            print("=" * 60 + "\n")

    # ── Comparison report ────────────────────────────────────────────────────

    def comparison_report(self) -> pd.DataFrame:
        """Return a DataFrame comparing CV and test accuracy for all models."""
        rows = []
        for name in self.cv_scores_:
            cv   = self.cv_scores_[name]
            test = self.test_reports_[name]["accuracy"]
            rows.append({
                "Model":          name,
                "CV Accuracy":    round(cv["mean"], 4),
                "CV Std":         round(cv["std"],  4),
                "Test Accuracy":  round(test,       4),
                "Best":           "★" if name == self.best_name_ else "",
            })
        return pd.DataFrame(rows).set_index("Model")

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save_best(self, path: str = "models/best_model.pkl") -> None:
        """Persist the best model (Pipeline) and the label encoder."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bundle = {
            "model":         self.models_[self.best_name_],
            "label_encoder": self.label_encoder_,
            "model_name":    self.best_name_,
            "cv_scores":     self.cv_scores_,
            "feature_names": FEATURE_NAMES,
            "classes":       CLASSES,
        }
        joblib.dump(bundle, path)
        print(f"[MLTrainer] Best model ({self.best_name_}) saved → {path}")

    def save_all(self, directory: str = "models/") -> None:
        """Persist every trained model as a separate .pkl file."""
        os.makedirs(directory, exist_ok=True)
        for name, pipe in self.models_.items():
            path = os.path.join(directory, f"{name.lower()}_model.pkl")
            bundle = {
                "model":         pipe,
                "label_encoder": self.label_encoder_,
                "model_name":    name,
                "feature_names": FEATURE_NAMES,
                "classes":       CLASSES,
            }
            joblib.dump(bundle, path)
        print(f"[MLTrainer] All {len(self.models_)} models saved → {directory}")

    def save_comparison_json(self, path: str = "output/model_comparison.json") -> None:
        """Write a JSON summary of all model metrics for the API / frontend."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        summary = {}
        for name in self.cv_scores_:
            summary[name] = {
                "cv_accuracy": self.cv_scores_[name],
                "test_accuracy": self.test_reports_[name]["accuracy"],
                "is_best": name == self.best_name_,
                "classification_report": self.test_reports_[name]["report"],
                "confusion_matrix": self.test_reports_[name]["confusion"],
            }
        with open(path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[MLTrainer] Comparison report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MLPredictor
# ─────────────────────────────────────────────────────────────────────────────

class MLPredictor:
    """
    Thin inference wrapper.  Loads a saved model bundle and exposes a simple
    predict interface that accepts raw metrics dicts from biomechanical_metrics.py.

    Parameters
    ----------
    model_path : str
        Path to the .pkl bundle saved by MLTrainer.save_best() or save_all().
    """

    def __init__(self, model_path: str = "models/best_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run `python train_model.py` to generate a trained model first."
            )
        bundle              = joblib.load(model_path)
        self._pipeline      = bundle["model"]
        self._le            = bundle["label_encoder"]
        self._model_name    = bundle["model_name"]
        self._feature_names = bundle.get("feature_names", FEATURE_NAMES)
        self._classes       = bundle.get("classes", CLASSES)

    # ── Core prediction ──────────────────────────────────────────────────────

    def predict_from_vector(self, feature_vector: np.ndarray) -> dict:
        """
        Predict from a raw 1-D numpy array of shape (4,).

        Returns
        -------
        dict with keys:
            label          : str   — "Poor" | "Average" | "Good"
            probabilities  : dict  — {class: float}
            confidence     : float — max probability
            model_used     : str   — name of the underlying model
        """
        x = feature_vector.reshape(1, -1)
        pred_enc  = self._pipeline.predict(x)[0]
        pred_proba = self._pipeline.predict_proba(x)[0]
        label     = str(self._le.inverse_transform([pred_enc])[0])

        proba_dict = {
            cls: round(float(p), 4)
            for cls, p in zip(self._classes, pred_proba)
        }

        return {
            "label":         label,
            "probabilities": proba_dict,
            "confidence":    round(float(pred_proba.max()), 4),
            "model_used":    self._model_name,
        }

    def predict_from_metrics(self, metrics: dict) -> dict:
        """
        Predict from a metrics dict as returned by biomechanical_metrics.py.

        Only the four biomechanical features are used; other keys are ignored.
        """
        x = np.array([
            metrics["elbow_angle"],
            metrics["spine_lean"],
            metrics["head_over_knee"],
            metrics["foot_direction"],
        ], dtype=float)
        return self.predict_from_vector(x)

    def predict_aggregated(self, all_metrics: list[dict]) -> dict:
        """
        Aggregate frame-level metrics into a single video-level prediction.

        Strategy: average all four features across frames, then predict once.
        This is more robust than voting on per-frame predictions (which are
        noisy due to landmark detection jitter).

        Parameters
        ----------
        all_metrics : list of metrics dicts from biomechanical_metrics.py

        Returns
        -------
        dict — same structure as predict_from_metrics(), plus:
            frame_count  : int   — number of frames used
            mean_features: dict  — per-feature averages
        """
        if not all_metrics:
            return {"label": "Unknown", "probabilities": {}, "confidence": 0.0,
                    "model_used": self._model_name, "frame_count": 0}

        arr = np.array([
            [m["elbow_angle"], m["spine_lean"],
             m["head_over_knee"], m["foot_direction"]]
            for m in all_metrics
            if m is not None
        ], dtype=float)

        mean_vec = arr.mean(axis=0)
        result   = self.predict_from_vector(mean_vec)
        result["frame_count"]   = len(arr)
        result["mean_features"] = {
            name: round(float(val), 4)
            for name, val in zip(self._feature_names, mean_vec)
        }
        return result