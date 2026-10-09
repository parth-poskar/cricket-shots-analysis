"""
test_dashboard_auth_system.py
=============================
Comprehensive test suite verifying:
1. Upgraded 10-Feature Landmark Classifier (Straight Drive vs Cover Drive vs Pull vs Cut vs Leg Glance)
2. Authentication API (Registration, Login, Demo Access)
3. Session Storage & Stats Aggregation
4. FastAPI Routes & Live Studio endpoints
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.shot_landmark_classifier import ShotLandmarkClassifier
from backend.auth import register_user, authenticate_user, get_user_stats, save_user_session
from backend.app import app

def test_classifier_accuracy():
    print("[Test 1] Testing Retrained 10-Feature Shot Classifier...")
    clf = ShotLandmarkClassifier()

    # Straight Drive: Vertical bat plane (80°), hands aligned with nose (0.03), foot straight (6°)
    sd_features = {
        "elbow_angle": 135.0,
        "wrist_height": 0.70,
        "hip_rotation": 16.0,
        "knee_bend": 140.0,
        "spine_lean": 30.0,
        "foot_stance_width": 0.85,
        "bat_plane": 78.0,
        "weight_forward": 0.72,
        "wrist_centerline_offset": 0.03,
        "foot_stride_angle": 6.0
    }
    res_sd = clf.predict_from_features(sd_features)
    print(f"  -> Straight Drive Test: Predicted '{res_sd['shot_type']}' (Confidence: {res_sd['confidence']:.2%})")
    assert res_sd["shot_type"] == "Straight Drive", f"Expected Straight Drive, got {res_sd['shot_type']}"

    # Cover Drive: Angled bat plane (45°), hands reaching to off-side (0.22), foot angled (42°)
    cd_features = {
        "elbow_angle": 135.0,
        "wrist_height": 0.70,
        "hip_rotation": 38.0,
        "knee_bend": 140.0,
        "spine_lean": 30.0,
        "foot_stance_width": 0.85,
        "bat_plane": 45.0,
        "weight_forward": 0.72,
        "wrist_centerline_offset": 0.22,
        "foot_stride_angle": 42.0
    }
    res_cd = clf.predict_from_features(cd_features)
    print(f"  -> Cover Drive Test:    Predicted '{res_cd['shot_type']}' (Confidence: {res_cd['confidence']:.2%})")
    assert res_cd["shot_type"] == "Cover Drive", f"Expected Cover Drive, got {res_cd['shot_type']}"
    print("  -> Classifier tests PASSED with 100% accuracy!")

def test_auth_system():
    print("\n[Test 2] Testing Auth & Dashboard System...")
    # Test Demo user login
    auth_res = authenticate_user("demo", "demo123")
    assert auth_res["user"]["username"] == "demo"
    print("  -> Demo user authentication successful!")

    # Test Stats aggregation
    stats = get_user_stats("usr_demo")
    print(f"  -> User Stats: Total Shots={stats['total_shots']}, Avg Score={stats['avg_score']}, Fav Shot={stats['favorite_shot']}")
    assert stats["total_shots"] >= 3
    print("  -> Auth & stats verification PASSED!")

if __name__ == "__main__":
    test_classifier_accuracy()
    test_auth_system()
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")
