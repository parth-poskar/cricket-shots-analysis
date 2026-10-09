import cv2

POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm                      
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg & foot
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # Right leg & foot
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10)
]


def draw_pose_landmarks(image, results):
    if not results:
        return image

    # Extract landmark list
    landmarks = None
    if hasattr(results, "pose_landmarks") and results.pose_landmarks is not None:
        pl = results.pose_landmarks
        if hasattr(pl, "landmark"):
            landmarks = pl.landmark
        elif isinstance(pl, list):
            if len(pl) > 0 and isinstance(pl[0], list):
                landmarks = pl[0]
            else:
                landmarks = pl
    elif isinstance(results, list):
        landmarks = results

    if not landmarks:
        return image

    h, w = image.shape[:2]
    points = {}
    for idx, lm in enumerate(landmarks):
        if idx > 32:
            break
        x, y = int(lm.x * w), int(lm.y * h)
        vis = getattr(lm, "visibility", 1.0)
        if vis is None or vis > 0.25:
            points[idx] = (x, y)

    # Draw connection lines (glowing cyan / purple)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(image, points[start_idx], points[end_idx], (245, 66, 230), 2, cv2.LINE_AA)

    # Draw joint keypoints
    for idx, pt in points.items():
        cv2.circle(image, pt, 4, (245, 117, 66), -1, cv2.LINE_AA)
        cv2.circle(image, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

    return image


def display_metrics_on_frame(image, metrics):
    if not metrics:
        return image

    cv2.putText(image, f"Elbow Angle: {metrics.get('elbow_angle', 0):.1f} deg", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Spine Lean: {metrics.get('spine_lean', 0):.1f} deg", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Head/Knee: {metrics.get('head_over_knee', 0):.3f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Foot Dir: {metrics.get('foot_direction', 0):.1f} deg", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return image
