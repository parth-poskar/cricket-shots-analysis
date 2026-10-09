"""
live_broadcast.py
=================
Desktop TV Broadcast Live Cricket Analysis Application.
Connects directly to your webcam / camera / video capture card and renders
real-time Star Sports / Hawk-Eye style HUD telemetry and shot classification.

Usage:
  python live_broadcast.py
  python live_broadcast.py --camera 0
  python live_broadcast.py --camera 1 --width 1920 --height 1080

Controls:
  [SPACE] or [S] : Capture high-res snapshot & export instant biomechanical report
  [H]           : Toggle HUD overlays (Clean Feed vs Full Broadcast Graphics)
  [R]           : Record live stroke sequence (5 seconds)
  [Q] or [ESC]  : Quit application
"""

import os
import sys
import cv2
import time
import json
import argparse
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pose_estimation import PoseEstimator
from src.biomechanical_metrics import calculate_metrics
from src.broadcast_overlay import BroadcastOverlay
from src.evaluation import save_evaluation
import src.config as config

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose if hasattr(mp, "solutions") else None
except Exception:
    mp_pose = None


def get_shot_classifier():
    try:
        from src.shot_landmark_classifier import ShotLandmarkClassifier
        return ShotLandmarkClassifier()
    except Exception as e:
        print(f"[live_broadcast] Shot classifier unavailable: {e}")
        return None


def run_live_broadcast(camera_id=0, width=1280, height=720, fps=30):
    print("=" * 65)
    print("  🏏 CRICPOSE AI — TV BROADCAST LIVE CAMERA STUDIO")
    print("=" * 65)
    print(f"[*] Initializing camera device index: {camera_id}")
    print("[*] Controls:")
    print("    [SPACE] / [S] : Save stroke snapshot & analysis report")
    print("    [H]           : Toggle HUD telemetry graphics")
    print("    [R]           : Record 5-second stroke clip")
    print("    [Q] / [ESC]   : Exit live broadcast")
    print("-" * 65)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Initialize Camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[-] ERROR: Could not open camera source [{camera_id}].")
        print("    Please ensure your webcam is connected or try '--camera 1'.")
        return

    # Set requested resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Camera feed active: {actual_w}x{actual_h} @ {fps} FPS")

    # Initialize AI Engines
    pose_estimator = PoseEstimator()
    shot_classifier = get_shot_classifier()
    overlay_engine = BroadcastOverlay()

    # State variables
    show_hud = True
    is_recording = False
    record_writer = None
    record_frames_remaining = 0
    recorded_metrics = []

    # Sliding window of landmarks for smooth shot classification
    landmark_window = deque(maxlen=24)
    last_shot_result = {"shot_type": "Detecting Stance...", "confidence": 0.0}
    last_classify_time = 0

    window_name = "CricPose AI — Live Broadcast Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_w, actual_h)

    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[-] Failed to grab frame from camera.")
                break

            frame_idx += 1
            # Mirror horizontally for natural webcam self-view
            display_frame = cv2.flip(frame, 1)

            # 1. Pose Tracking
            processed_img, results = pose_estimator.process_frame(display_frame)
            landmarks = pose_estimator.get_landmarks(results)

            metrics = None
            if landmarks:
                metrics = calculate_metrics(landmarks, mp_pose)
                landmark_window.append(landmarks)

            # 2. Periodic Shot Classification (every 8 frames)
            current_time = time.time()
            if shot_classifier and len(landmark_window) >= 8 and (current_time - last_classify_time > 0.25):
                try:
                    last_shot_result = shot_classifier.predict_from_landmarks(list(landmark_window))
                    last_classify_time = current_time
                except Exception:
                    pass

            # 3. Recording Live Stroke Sequence
            if is_recording:
                recorded_metrics.append(metrics)
                if record_writer:
                    record_writer.write(display_frame)
                record_frames_remaining -= 1

                # Draw recording badge on screen
                cv2.circle(display_frame, (actual_w // 2 - 80, 80), 8, (0, 0, 255), -1)
                cv2.putText(display_frame, f"REC STROKE ({record_frames_remaining})", 
                            (actual_w // 2 - 65, 85), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

                if record_frames_remaining <= 0:
                    is_recording = False
                    if record_writer:
                        record_writer.release()
                        record_writer = None
                    print("[+] Live stroke recording complete! Report saved.")

            # 4. Render TV-Broadcast HUD & Skeleton
            if show_hud:
                if landmarks:
                    display_frame = overlay_engine.draw_skeleton(display_frame, landmarks, metrics)
                display_frame = overlay_engine.draw_broadcast_hud(
                    display_frame, metrics, last_shot_result, is_live=True
                )

            # 5. Display Window
            cv2.imshow(window_name, display_frame)

            # Key Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[*] Exiting Live Broadcast...")
                break
            elif key == ord('h') or key == ord('H'):
                show_hud = not show_hud
                print(f"[*] HUD Overlays: {'ON' if show_hud else 'OFF (Clean Feed)'}")
            elif key == ord('s') or key == ord('S') or key == 32:  # 's' or SPACE
                # Save snapshot + JSON report
                ts = int(time.time())
                snap_path = os.path.join(config.OUTPUT_DIR, f"live_snapshot_{ts}.jpg")
                json_path = os.path.join(config.OUTPUT_DIR, f"live_snapshot_{ts}_result.json")
                cv2.imwrite(snap_path, display_frame)

                report = {
                    "timestamp": ts,
                    "shot_classification": last_shot_result,
                    "biomechanical_metrics": metrics,
                    "snapshot_image": os.path.basename(snap_path)
                }
                with open(json_path, "w") as jf:
                    json.dump(report, jf, indent=4)
                print(f"[+] Instant snapshot saved -> {snap_path}")
                print(f"[+] Biomechanical report saved -> {json_path}")
            elif key == ord('r') or key == ord('R'):
                if not is_recording:
                    ts = int(time.time())
                    rec_path = os.path.join(config.OUTPUT_DIR, f"live_stroke_{ts}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    record_writer = cv2.VideoWriter(rec_path, fourcc, fps, (actual_w, actual_h))
                    record_frames_remaining = int(fps * 5)  # 5-second burst
                    is_recording = True
                    recorded_metrics = []
                    print(f"[*] Started 5s stroke recording -> {rec_path}")

    finally:
        cap.release()
        if record_writer:
            record_writer.release()
        cv2.destroyAllWindows()
        print("[+] Live broadcast session finished cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CricPose AI Live TV Broadcast Application")
    parser.add_argument("--camera", type=str, default="0", 
                        help="Camera device index (e.g. 0, 1, 2) or IP Webcam URL (e.g. http://192.168.1.50:8080/video)")
    parser.add_argument("--width", type=int, default=1280, help="Stream width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Stream height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS target (default: 30)")
    args = parser.parse_args()

    # Parse camera input: if all digits, treat as integer device index; otherwise treat as stream URL
    camera_source = int(args.camera) if args.camera.isdigit() else args.camera

    run_live_broadcast(camera_id=camera_source, width=args.width, height=args.height, fps=args.fps)
