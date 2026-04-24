"""
debug_features.py
=================
Run this on your cover drive videos to see what feature values
MediaPipe is ACTUALLY producing. We use these real numbers to fix
the synthetic training data ranges in train_shot_landmark_model.py

Run from project root:
    python debug_features.py

It will print a table like:
    elbow_angle  wrist_height  hip_rotation  knee_bend  ...
    121.3        0.82          31.4          138.2      ...
    ...
    --- AVERAGES ---
    elbow_angle:  124.5
    ...

Copy those averages and paste them here so we can fix the training ranges.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pose_estimation import PoseEstimator
from src.shot_landmark_classifier import extract_features, FEATURE_COLS

TEST_FOLDER = "test_folder"

def debug_video(video_path, pose_estimator):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] Cannot open {video_path}")
        return []

    all_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, results = pose_estimator.process_frame(frame)
        lm = pose_estimator.get_landmarks(results)
        f  = extract_features(lm)
        if f is not None:
            all_features.append(f)

    cap.release()
    return all_features


def main():
    pose_estimator = PoseEstimator()

    video_files = [
        os.path.join(TEST_FOLDER, f)
        for f in os.listdir(TEST_FOLDER)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".gif"))
    ]

    if not video_files:
        print(f"No videos found in {TEST_FOLDER}/")
        return

    print(f"\nFound {len(video_files)} video(s) in {TEST_FOLDER}/\n")
    print("="*70)

    all_video_means = []

    for vpath in video_files:
        name = os.path.basename(vpath)
        print(f"\nVideo: {name}")
        print("-" * 50)

        features_list = debug_video(vpath, pose_estimator)

        if not features_list:
            print("  No landmarks detected!")
            continue

        # Per-video mean
        arr = np.array([[f[k] for k in FEATURE_COLS] for f in features_list])
        means = arr.mean(axis=0)
        mins  = arr.min(axis=0)
        maxs  = arr.max(axis=0)

        print(f"  Frames with landmarks: {len(features_list)}")
        print(f"\n  {'Feature':<22} {'Min':>8} {'Mean':>8} {'Max':>8}")
        print(f"  {'-'*50}")
        for feat, mn, me, mx in zip(FEATURE_COLS, mins, means, maxs):
            print(f"  {feat:<22} {mn:>8.2f} {me:>8.2f} {mx:>8.2f}")

        all_video_means.append(means)

    # Overall average across all videos
    if all_video_means:
        overall = np.mean(all_video_means, axis=0)
        print("\n" + "="*70)
        print("OVERALL AVERAGES ACROSS ALL VIDEOS (copy these for the fix):")
        print("="*70)
        for feat, val in zip(FEATURE_COLS, overall):
            print(f"  {feat:<22} {val:.2f}")
        print("="*70)
        print("\nShare these numbers and we will update the training ranges.")

if __name__ == "__main__":
    main()
    