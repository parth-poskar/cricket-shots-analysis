# config.py

VIDEO_URL = "https://youtube.com/shorts/vSX3IRxGnNY"
OUTPUT_DIR = "output"

# MediaPipe Pose Estimation
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Biomechanical Metrics Thresholds (example values)
ELBOW_ANGLE_GOOD_THRESHOLD = (90, 180) # Changed to a range
HEAD_OVER_KNEE_GOOD_THRESHOLD = 0.1
FOOT_DIRECTION_GOOD_RANGE = (80, 100)
SPINE_LEAN_GOOD_RANGE = (80, 100)

# Phase Segmentation Thresholds (example values)
HIP_VELOCITY_THRESHOLD = 0.02
ARM_VELOCITY_THRESHOLD = 0.03
IMPACT_WRIST_VELOCITY_THRESHOLD = 0.05

# Contact Detection Thresholds
CONTACT_WRIST_VELOCITY_THRESHOLD = 0.05

# Bat Tracking Thresholds (example values)
BAT_MIN_AREA = 100
BAT_ASPECT_RATIO_THRESHOLD = 3

# Skill Grade Thresholds
ADVANCED_SCORE = 8
INTERMEDIATE_SCORE = 5