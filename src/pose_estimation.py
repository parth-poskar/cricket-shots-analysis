import cv2

# ✅ FIX: proper mediapipe import handling
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp = None
    mp_pose = None


class PoseEstimator:
    def __init__(self):
        if mp_pose is None:
            raise ImportError(
                "MediaPipe is not installed correctly. "
                "Run: pip uninstall mediapipe -y && pip install mediapipe==0.10.11"
            )

        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = self.pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, results

    def get_landmarks(self, results):
        if results and getattr(results, "pose_landmarks", None):
            return results.pose_landmarks.landmark
        return None