import cv2
import mediapipe as mp

def draw_pose_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    if results and getattr(results, "pose_landmarks", None):
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    return image

def display_metrics_on_frame(image, metrics):
    if not metrics:
        return image
    cv2.putText(image, f"Elbow Angle: {metrics.get('elbow_angle', 0):.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Spine Lean: {metrics.get('spine_lean', 0):.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Head/Knee: {metrics.get('head_over_knee', 0):.3f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Foot Dir: {metrics.get('foot_direction', 0):.1f}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image