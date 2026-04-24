"""
soft_computing.py
=================
Soft Computing Extension for Cricket Cover Drive Biomechanical Analysis
------------------------------------------------------------------------
Adds two soft computing techniques to the existing project:
  1. Fuzzy Logic Evaluator  — replaces hard thresholds with fuzzy membership
  2. Neural Network Classifier — learns shot quality from biomechanical features

Drop this file into your project and import it anywhere.
It reads the SAME metrics dict produced by biomechanical_metrics.py and the
SAME thresholds defined in config.py, so NOTHING in your existing code changes.

Usage Example:
    from soft_computing import FuzzyEvaluator, NeuralNetClassifier

    metrics = calculate_metrics(landmarks, mp_pose)   # your existing function

    # --- Fuzzy Logic ---
    fuzzy = FuzzyEvaluator()
    scores = fuzzy.evaluate(metrics)
    print(scores)   # {'elbow': 0.87, 'spine': 0.65, ...}

    # --- Neural Network ---
    nn = NeuralNetClassifier()
    # nn.train(X_train, y_train)   # train once with labelled data
    quality = nn.predict(metrics)  # 'Good', 'Average', or 'Poor'
"""

import numpy as np
import json
import os


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — FUZZY LOGIC EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY FUZZY LOGIC HERE?
# ---------------------
# config.py defines crisp ranges, e.g.:
#   ELBOW_ANGLE_GOOD_THRESHOLD  = (90, 180)
#   SPINE_LEAN_GOOD_RANGE       = (80, 100)
#
# A hard rule says "elbow_angle=89 → Bad". But 89° vs 90° is almost identical.
# Fuzzy logic expresses this as a *degree of membership* in [0, 1]:
#   elbow_angle=135  → membership 1.0  (perfectly inside range)
#   elbow_angle=95   → membership 0.9  (mostly good, near the edge)
#   elbow_angle=60   → membership 0.1  (mostly bad, far from range)
#
# Membership functions used:
#   • Trapezoidal (trapz)  — for ranges with a flat "best zone"
#   • Triangular  (tri)    — for single optimal value targets
# ══════════════════════════════════════════════════════════════════════════════

class FuzzyEvaluator:
    """
    Evaluates biomechanical metrics using fuzzy membership functions.
    Returns a score in [0, 1] for each metric and an overall weighted score.
    """

    # ── Membership Functions ──────────────────────────────────────────────────

    @staticmethod
    def trapz_membership(x, a, b, c, d):
        """
        Trapezoidal membership function.
        
            1       |‾‾‾‾‾‾|
            0   ___/        \___
                a  b        c  d

        Fully 'in' for x in [b, c]; linearly rises from a→b and falls from c→d.
        x: input value
        a, b, c, d: feet and shoulders of the trapezoid (a < b <= c < d)
        """
        if x <= a or x >= d:
            return 0.0
        elif b <= x <= c:
            return 1.0
        elif a < x < b:
            return (x - a) / (b - a)
        else:  # c < x < d
            return (d - x) / (d - c)

    @staticmethod
    def tri_membership(x, a, b, c):
        """
        Triangular membership function.
        
            1       /\
            0  ____/  \____
               a   b   c

        Peaks at x=b; linearly rises from a→b and falls from b→c.
        """
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif a < x < b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)

    # ── Per-metric fuzzy rules ────────────────────────────────────────────────
    #  These numbers are derived from the ranges in config.py but extended into
    #  smooth transition zones on each side.

    def elbow_fuzzy(self, angle):
        """
        Good elbow angle for cover drive: 90°–160° (peak 110°–140°).
        Below 60° → fully collapsed arm (bad). Above 180° → over-extended (bad).
        """
        return self.trapz_membership(angle, 60, 90, 160, 185)

    def spine_fuzzy(self, angle):
        """
        Spine lean 80°–100° is ideal (upright with slight forward lean).
        config: SPINE_LEAN_GOOD_RANGE = (80, 100)
        """
        return self.trapz_membership(angle, 65, 80, 100, 115)

    def head_over_knee_fuzzy(self, distance):
        """
        Smaller distance = head directly over knee = better.
        config: HEAD_OVER_KNEE_GOOD_THRESHOLD = 0.1
        0.0 is perfect; beyond 0.2 starts becoming poor.
        """
        # Invert: closeness to 0 is best → map 0 → 1.0, 0.2+ → 0.0
        return self.trapz_membership(distance, -0.01, 0.0, 0.05, 0.2)

    def foot_direction_fuzzy(self, angle):
        """
        Foot should point towards cover region: 80°–100°.
        config: FOOT_DIRECTION_GOOD_RANGE = (80, 100)
        """
        return self.trapz_membership(angle, 65, 80, 100, 115)

    # ── Evaluate all metrics ──────────────────────────────────────────────────

    def evaluate(self, metrics: dict) -> dict:
        """
        Input:  metrics dict from biomechanical_metrics.calculate_metrics()
        Output: dict with fuzzy score [0,1] per metric + weighted overall score.
        
        Weights reflect coaching importance for a cover drive:
          Swing Control (elbow)  → 30%
          Head Position          → 30%
          Balance (spine)        → 20%
          Footwork               → 20%
        """
        if metrics is None:
            return {}

        scores = {
            "elbow_fuzzy_score":     self.elbow_fuzzy(metrics["elbow_angle"]),
            "spine_fuzzy_score":     self.spine_fuzzy(metrics["spine_lean"]),
            "head_fuzzy_score":      self.head_over_knee_fuzzy(metrics["head_over_knee"]),
            "foot_fuzzy_score":      self.foot_direction_fuzzy(metrics["foot_direction"]),
        }

        weights = {
            "elbow_fuzzy_score": 0.30,
            "head_fuzzy_score":  0.30,
            "spine_fuzzy_score": 0.20,
            "foot_fuzzy_score":  0.20,
        }

        overall = sum(scores[k] * weights[k] for k in weights)
        scores["overall_fuzzy_score"] = round(overall, 4)
        scores["grade"] = self._grade(overall)

        return scores

    @staticmethod
    def _grade(score):
        """Defuzzify overall score into a human-readable grade."""
        if score >= 0.75:
            return "Excellent"
        elif score >= 0.50:
            return "Good"
        elif score >= 0.25:
            return "Needs Improvement"
        else:
            return "Poor"

    def evaluate_to_json(self, metrics: dict, output_dir: str = "output") -> dict:
        """
        Evaluate and also save fuzzy scores alongside the existing evaluation.json.
        Mirrors the pattern used in evaluation.py → save_evaluation().
        """
        result = self.evaluate(metrics)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "fuzzy_evaluation.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=4)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — NEURAL NETWORK CLASSIFIER  (pure NumPy, no PyTorch/TensorFlow)
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY NEURAL NETWORK HERE?
# -------------------------
# biomechanical_metrics.py extracts 4 features per frame:
#   [elbow_angle, spine_lean, head_over_knee, foot_direction]
#
# Instead of manually coding rules for all combinations, a neural network can
# LEARN the mapping:
#   features → shot quality (Good / Average / Poor)
#
# Architecture:  Input(4) → Hidden(8, ReLU) → Hidden(4, ReLU) → Output(3, Softmax)
# Training:      Backpropagation with cross-entropy loss (implemented from scratch)
# This is a classic soft computing / ANN demonstration, perfect for mini-project.
# ══════════════════════════════════════════════════════════════════════════════

class NeuralNetClassifier:
    """
    3-class feedforward neural network for shot quality classification.
    Classes:  0 = Poor | 1 = Average | 2 = Good
    
    Architecture: 4 → 8 → 4 → 3  (all weights initialised with He init)
    """

    CLASSES = ["Poor", "Average", "Good"]

    def __init__(self, hidden1=8, hidden2=4, lr=0.01, epochs=1000, seed=42):
        np.random.seed(seed)
        self.lr     = lr
        self.epochs = epochs

        # He initialisation for ReLU layers
        self.W1 = np.random.randn(4, hidden1) * np.sqrt(2 / 4)
        self.b1 = np.zeros((1, hidden1))

        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2 / hidden1)
        self.b2 = np.zeros((1, hidden2))

        self.W3 = np.random.randn(hidden2, 3) * np.sqrt(2 / hidden2)
        self.b3 = np.zeros((1, 3))

        # Normalization stats (set during training)
        self.mu  = None
        self.std = None

    # ── Activations ──────────────────────────────────────────────────────────

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(float)

    @staticmethod
    def _softmax(z):
        ez = np.exp(z - np.max(z, axis=1, keepdims=True))
        return ez / ez.sum(axis=1, keepdims=True)

    # ── Forward Pass ─────────────────────────────────────────────────────────

    def _forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = self._softmax(self.z3)
        return self.a3

    # ── Backward Pass (Backpropagation) ───────────────────────────────────────

    def _backward(self, X, y_one_hot):
        m = X.shape[0]

        # Output layer gradient (cross-entropy + softmax combined)
        dz3 = self.a3 - y_one_hot                        # (m, 3)
        dW3 = self.a2.T @ dz3 / m
        db3 = dz3.mean(axis=0, keepdims=True)

        # Hidden layer 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self._relu_deriv(self.z2)
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.mean(axis=0, keepdims=True)

        # Hidden layer 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_deriv(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = dz1.mean(axis=0, keepdims=True)

        # Gradient descent update
        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    # ── Loss ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _cross_entropy(y_pred, y_one_hot):
        eps = 1e-9
        return -np.mean(np.sum(y_one_hot * np.log(y_pred + eps), axis=1))

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Train the network.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 4)
            Each row is [elbow_angle, spine_lean, head_over_knee, foot_direction]
        y : np.ndarray, shape (n_samples,)
            Class labels: 0=Poor, 1=Average, 2=Good
        verbose : bool
            Print loss every 100 epochs.
        """
        # Normalise features to zero-mean, unit-variance
        self.mu  = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        X_norm   = (X - self.mu) / self.std

        # One-hot encode labels
        n_classes  = 3
        y_one_hot  = np.eye(n_classes)[y.astype(int)]

        self.loss_history = []
        for epoch in range(1, self.epochs + 1):
            y_pred = self._forward(X_norm)
            loss   = self._cross_entropy(y_pred, y_one_hot)
            self._backward(X_norm, y_one_hot)
            self.loss_history.append(loss)

            if verbose and epoch % 100 == 0:
                acc = self._accuracy(X_norm, y)
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc*100:.1f}%")

        print(f"\nTraining complete. Final accuracy: {self._accuracy(X_norm, y)*100:.1f}%")

    def _accuracy(self, X_norm, y):
        probs = self._forward(X_norm)
        preds = np.argmax(probs, axis=1)
        return (preds == y).mean()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, metrics: dict) -> str:
        """
        Predict shot quality from a metrics dict.
        Returns: 'Good', 'Average', or 'Poor'
        """
        x = self._metrics_to_vector(metrics)
        if self.mu is not None:
            x = (x - self.mu) / self.std
        probs = self._forward(x.reshape(1, -1))
        label = int(np.argmax(probs))
        return self.CLASSES[label]

    def predict_proba(self, metrics: dict) -> dict:
        """Returns probability for each class as a dict."""
        x = self._metrics_to_vector(metrics)
        if self.mu is not None:
            x = (x - self.mu) / self.std
        probs = self._forward(x.reshape(1, -1)).flatten()
        return {cls: round(float(p), 4) for cls, p in zip(self.CLASSES, probs)}

    @staticmethod
    def _metrics_to_vector(metrics: dict) -> np.ndarray:
        """Convert metrics dict → 4-element numpy array (same order as training)."""
        return np.array([
            metrics["elbow_angle"],
            metrics["spine_lean"],
            metrics["head_over_knee"],
            metrics["foot_direction"],
        ], dtype=float)

    # ── Save / Load weights ───────────────────────────────────────────────────

    def save_weights(self, path: str = "output/nn_weights.npz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1,
                       W2=self.W2, b2=self.b2,
                       W3=self.W3, b3=self.b3,
                       mu=self.mu if self.mu is not None else np.zeros(4),
                       std=self.std if self.std is not None else np.ones(4))
        print(f"Weights saved → {path}")

    def load_weights(self, path: str = "output/nn_weights.npz"):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]
        self.mu, self.std = data["mu"], data["std"]
        print(f"Weights loaded ← {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — SYNTHETIC DATA GENERATOR  (for demo / testing without labelled video)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_training_data(n_samples: int = 300, seed: int = 0) -> tuple:
    """
    Generates synthetic labelled training data based on the config thresholds.
    In a real project you would replace this with frames labelled by a coach.

    Rules (mirrors the fuzzy membership logic):
      Good    → all metrics near ideal ranges
      Average → some metrics near boundary
      Poor    → metrics clearly outside ranges

    Returns
    -------
    X : np.ndarray, shape (n_samples, 4)   [elbow, spine, head, foot]
    y : np.ndarray, shape (n_samples,)     [0=Poor, 1=Average, 2=Good]
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    samples_per_class = n_samples // 3

    # Good shots  (class 2)
    for _ in range(samples_per_class):
        X.append([
            rng.uniform(100, 155),   # elbow_angle   in good range
            rng.uniform(82, 98),     # spine_lean     in good range
            rng.uniform(0.0, 0.06),  # head_over_knee small = good
            rng.uniform(82, 98),     # foot_direction in good range
        ])
        y.append(2)

    # Average shots  (class 1)
    for _ in range(samples_per_class):
        X.append([
            rng.uniform(75, 105),    # elbow on boundary
            rng.uniform(75, 85),     # spine slightly off
            rng.uniform(0.05, 0.14), # head moderate
            rng.uniform(75, 88),     # foot slightly off
        ])
        y.append(1)

    # Poor shots  (class 0)
    for _ in range(n_samples - 2 * samples_per_class):
        X.append([
            rng.uniform(30, 75),     # elbow very bent / collapsed
            rng.uniform(50, 72),     # spine heavily leaning
            rng.uniform(0.15, 0.35), # head far from knee
            rng.uniform(40, 70),     # foot pointing wrong way
        ])
        y.append(0)

    return np.array(X, dtype=float), np.array(y, dtype=int)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO — run `python soft_computing.py` to see both systems in action
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SOFT COMPUTING DEMO — Cricket Cover Drive Analysis")
    print("=" * 60)

    # Simulate a metrics dict as produced by biomechanical_metrics.py
    sample_metrics = {
        "elbow_angle":    125.4,
        "spine_lean":     88.7,
        "head_over_knee": 0.04,
        "foot_direction": 91.2,
        "wrist":          [0.5, 0.6],
        "shoulder":       [0.4, 0.3],
    }

    # ── FUZZY LOGIC ───────────────────────────────────────────────────────────
    print("\n[1] FUZZY LOGIC EVALUATOR")
    print("-" * 40)
    fuzzy = FuzzyEvaluator()
    fuzzy_result = fuzzy.evaluate(sample_metrics)

    for key, val in fuzzy_result.items():
        if key == "grade":
            print(f"  {key:<28} {val}")
        else:
            bar = "█" * int(val * 20)
            print(f"  {key:<28} {val:.4f}  {bar}")

    # ── NEURAL NETWORK ────────────────────────────────────────────────────────
    print("\n[2] NEURAL NETWORK CLASSIFIER")
    print("-" * 40)
    X_train, y_train = generate_synthetic_training_data(n_samples=600)
    print(f"  Training on {len(X_train)} synthetic samples...")

    nn = NeuralNetClassifier(lr=0.05, epochs=500)
    nn.train(X_train, y_train, verbose=True)

    print(f"\n  Predicting quality for sample metrics:")
    prediction = nn.predict(sample_metrics)
    proba      = nn.predict_proba(sample_metrics)

    print(f"  → Predicted class : {prediction}")
    print(f"  → Class probabilities:")
    for cls, p in proba.items():
        bar = "█" * int(p * 30)
        print(f"     {cls:<10} {p:.4f}  {bar}")

    print("\n" + "=" * 60)
    print("  Both systems read from biomechanical_metrics.py output.")
    print("  No existing files were modified.")
    print("=" * 60)