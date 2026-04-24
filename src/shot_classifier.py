"""
shot_classifier.py  — FIXED VERSION
=====================================
Key fixes:
  1. Reads num_classes from the saved JSON (no more hardcoded 5)
     → if your dataset has 4 or 6 folders, this still works correctly
  2. Falls back gracefully when model file doesn't exist
  3. Added confidence threshold warning for low-confidence predictions
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

MODEL_PATH     = "models/shot_type_model.pth"
CLASS_MAP_PATH = "models/shot_type_classes.json"

# Warn the user when confidence is below this level
LOW_CONFIDENCE_THRESHOLD = 0.40


class ShotClassifier:
    def __init__(self):
        self.device = torch.device("cpu")

        # FIX 1: load classes FIRST, then build model with correct output size
        self.classes   = self._load_classes()
        num_classes    = len(self.classes)

        self.model = models.resnet18(pretrained=False)
        # FIX 2: use num_classes from JSON, not hardcoded 5
        self.model.fc  = nn.Linear(self.model.fc.in_features, num_classes)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Shot model not found at '{MODEL_PATH}'. "
                "Run  python train_shot_model.py  first."
            )

        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        print(f"[ShotClassifier] Loaded model with {num_classes} classes: {self.classes}")

    def _load_classes(self):
        if os.path.exists(CLASS_MAP_PATH):
            with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
                classes = json.load(f)
            # Strip numeric prefixes like "1. Cover Drive" → "Cover Drive"
            return [c.split(". ", 1)[1] if ". " in c else c for c in classes]

        # Fallback — only used if JSON is missing
        print("[ShotClassifier] WARNING: class map JSON not found, using fallback list.")
        return [
            "Cover Drive",
            "Cut Shot",
            "Leg Glance Shot",
            "Pull Shot",
            "Straight Drive",
        ]

    def predict(self, frames) -> dict:
        """
        Parameters
        ----------
        frames : np.ndarray, shape (N, H, W, 3), values in [0, 1]

        Returns
        -------
        dict with keys: shot_type, confidence, all_probabilities, low_confidence
        """
        if len(frames) == 0:
            return {
                "shot_type":        "Unknown",
                "confidence":       0.0,
                "all_probabilities": {},
                "low_confidence":   True,
            }

        preds = []
        for f in frames:
            img = (f * 255).astype(np.uint8)
            img = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                out  = self.model(img)
                prob = torch.softmax(out, dim=1)
                preds.append(prob.numpy())

        avg = np.mean(preds, axis=0)[0]          # shape: (num_classes,)
        idx = int(np.argmax(avg))

        confidence     = float(avg[idx])
        low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

        if low_confidence:
            print(f"[ShotClassifier] WARNING: low confidence ({confidence:.2%}) "
                  f"— model is unsure. Consider retraining with more data.")

        return {
            "shot_type":        self.classes[idx],
            "confidence":       confidence,
            "all_probabilities": {
                cls: round(float(prob), 4)
                for cls, prob in zip(self.classes, avg)
            },
            "low_confidence": low_confidence,
        }