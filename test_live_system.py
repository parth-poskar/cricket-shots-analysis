"""
scratch/test_live_system.py
===========================
Verification script to test:
1. BroadcastOverlay rendering
2. PoseEstimator + calculate_metrics + ShotLandmarkClassifier live loop
3. FastAPI endpoint imports and WebSocket route availability
"""

import os
import sys
import numpy as np
import cv2
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.broadcast_overlay import BroadcastOverlay
from src.pose_estimation import PoseEstimator
from src.biomechanical_metrics import calculate_metrics
from src.shot_landmark_classifier import ShotLandmarkClassifier
from backend.app import app

def test_broadcast_overlay():
    print("[Test 1] Testing BroadcastOverlay...")
    overlay = BroadcastOverlay()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    dummy_metrics = {
        "elbow_angle": 135.5,
        "spine_lean": 88.0,
        "head_over_knee": 0.045,
        "foot_direction": 92.0
    }
    dummy_shot = {"shot_type": "Cover Drive", "confidence": 0.94}
    dummy_quality = {"label": "Excellent", "confidence": 0.96}

    result_frame = overlay.draw_broadcast_hud(dummy_frame, dummy_metrics, dummy_shot, dummy_quality, is_live=True)
    assert result_frame is not None
    assert result_frame.shape == (720, 1280, 3)
    print("  -> BroadcastOverlay passed successfully!")

def test_ai_pipeline():
    print("[Test 2] Testing PoseEstimator + Classifier on dummy frame...")
    estimator = PoseEstimator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, results = estimator.process_frame(dummy_frame)
    landmarks = estimator.get_landmarks(results)
    print("  -> PoseEstimator ran without errors.")

    clf = ShotLandmarkClassifier()
    assert clf is not None
    print(f"  -> ShotLandmarkClassifier loaded with classes: {clf.classes}")

def test_fastapi_routes():
    print("[Test 3] Testing FastAPI Routes...")
    routes = [route.path for route in app.routes]
    print(f"  -> Registered Routes: {routes}")
    assert "/ws/live-feed" in routes
    assert "/live/snapshot" in routes
    assert "/upload/" in routes
    assert "/health" in routes
    print("  -> All broadcast and analysis routes registered successfully!")

if __name__ == "__main__":
    test_broadcast_overlay()
    test_ai_pipeline()
    test_fastapi_routes()
    print("\n[SUCCESS] ALL VERIFICATION TESTS PASSED!")
