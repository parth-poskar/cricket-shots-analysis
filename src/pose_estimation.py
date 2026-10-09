import os
import sys
import time
import urllib.request
import cv2
import numpy as np

# Try importing mediapipe and detecting which API flavor is available
_MP_API_TYPE = None

try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        _MP_API_TYPE = "solutions"
    elif hasattr(mp, "tasks"):
        _MP_API_TYPE = "tasks"
except Exception:
    mp = None


class LandmarkWrapper:
    """Wrapper to provide consistent access to a single landmark point."""
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=1.0, presence=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility) if visibility is not None else 1.0
        self.presence = float(presence) if presence is not None else 1.0

    def __repr__(self):
        return f"Landmark(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, vis={self.visibility:.2f})"


class LandmarkListWrapper(list):
    """List of landmarks that also exposes .landmark property for backward compatibility."""
    @property
    def landmark(self):
        return self


class PoseResultsWrapper:
    """Standardized result object matching mediapipe pose output schema."""
    def __init__(self, pose_landmarks=None, raw_result=None):
        self.pose_landmarks = pose_landmarks
        self.raw_result = raw_result


class PoseEstimator:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pose_landmarker_full.task")

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_path=None):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.api_type = _MP_API_TYPE
        self.detector = None

        if self.api_type == "solutions":
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        elif self.api_type == "tasks":
            self._init_tasks_pose_landmarker()
        else:
            raise ImportError(
                "MediaPipe is not installed or available. "
                "Please run: pip install mediapipe opencv-python"
            )

    def _ensure_task_model(self):
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000000:
            return self.model_path
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        print(f"[PoseEstimator] Downloading MediaPipe pose task model to {self.model_path}...")
        try:
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
            print(f"[PoseEstimator] Model downloaded successfully ({os.path.getsize(self.model_path)} bytes).")
        except Exception as e:
            raise RuntimeError(f"Failed to download MediaPipe pose landmarker model: {e}")
        return self.model_path

    def _init_tasks_pose_landmarker(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_file = self._ensure_task_model()
        base_options = python.BaseOptions(model_asset_path=model_file)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame):
        """
        Process a BGR OpenCV frame.
        Returns: (image, results) where results has .pose_landmarks
        """
        if self.api_type == "solutions":
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image, results

        elif self.api_type == "tasks":
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)

            pose_landmarks = None
            if detection_result and detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                raw_landmarks = detection_result.pose_landmarks[0]
                landmarks_list = LandmarkListWrapper([
                    LandmarkWrapper(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=getattr(lm, "visibility", 1.0),
                        presence=getattr(lm, "presence", 1.0)
                    ) for lm in raw_landmarks
                ])
                pose_landmarks = landmarks_list

            results = PoseResultsWrapper(pose_landmarks=pose_landmarks, raw_result=detection_result)
            return frame, results

        return frame, None

    def get_landmarks(self, results):
        """Extract list of 33 landmarks from result object."""
        if not results:
            return None

        # Solutions API format: results.pose_landmarks.landmark
        if hasattr(results, "pose_landmarks") and results.pose_landmarks is not None:
            pl = results.pose_landmarks
            if hasattr(pl, "landmark"):
                return list(pl.landmark)
            elif isinstance(pl, list):
                return pl
        return None


def run_standalone_demo():
    """Runs a demonstration of pose estimation on a test video or synthetic sample."""
    print("=" * 60)
    print("[*] Cricket Pose Estimation - Standalone Demo")
    print("=" * 60)

    # Initialize PoseEstimator
    t0 = time.time()
    estimator = PoseEstimator()
    print(f"[+] PoseEstimator initialized successfully using [{estimator.api_type}] backend in {time.time()-t0:.2f}s")

    # Locate a sample video
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_folder = os.path.join(root_dir, "test_folder")
    sample_videos = []
    if os.path.exists(test_folder):
        sample_videos = [
            os.path.join(test_folder, f)
            for f in os.listdir(test_folder)
            if f.lower().endswith((".mp4", ".mov", ".avi"))
        ]

    if sample_videos:
        video_path = sample_videos[0]
        print(f"[*] Processing sample video: {os.path.basename(video_path)}")
    else:
        print("[i] No test video found in test_folder/. Creating synthetic test frame...")
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Pose Estimator Test Frame", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        _, results = estimator.process_frame(dummy_frame)
        print(f"[+] Processed test frame successfully. Landmarks detected: {results.pose_landmarks is not None}")
        return

    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[-] Failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pose_estimation_demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Import overlay and metrics if available
    try:
        from src.overlay_utils import draw_pose_landmarks, display_metrics_on_frame
        from src.biomechanical_metrics import calculate_metrics
    except ImportError:
        draw_pose_landmarks = None
        display_metrics_on_frame = None
        calculate_metrics = None

    frame_idx = 0
    detected_count = 0
    start_proc_time = time.time()

    print(f"[*] Video Info: {width}x{height} @ {fps:.1f} FPS ({total_frames} frames)")
    print("[*] Processing frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        processed_frame, results = estimator.process_frame(frame)
        landmarks = estimator.get_landmarks(results)

        if landmarks is not None and len(landmarks) >= 33:
            detected_count += 1
            if calculate_metrics:
                metrics = calculate_metrics(landmarks)
                if metrics and display_metrics_on_frame:
                    display_metrics_on_frame(processed_frame, metrics)

            if draw_pose_landmarks:
                draw_pose_landmarks(processed_frame, results)

        writer.write(processed_frame)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"  Frame {frame_idx}/{total_frames} (Detections: {detected_count})")

    cap.release()
    writer.release()
    elapsed = time.time() - start_proc_time

    print("\n" + "=" * 60)
    print("[+] Pose Estimation Complete!")
    print(f"  Total Frames:    {frame_idx}")
    print(f"  Detected Frames: {detected_count} ({detected_count/max(frame_idx, 1)*100:.1f}%)")
    print(f"  Processing Time: {elapsed:.2f}s ({frame_idx/max(elapsed, 0.001):.1f} FPS)")
    print(f"  Output Saved:    {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_standalone_demo()