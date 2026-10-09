import numpy as np

# Standard 33-point MediaPipe Pose topology index constants
class PoseLandmarksEnum:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return float(angle)


def get_joint_coordinates(landmarks, mp_pose=None):
    if landmarks is None or len(landmarks) < 32:
        return None

    coords = {}
    try:
        def _get_pt(idx):
            lm = landmarks[idx]
            if hasattr(lm, "x") and hasattr(lm, "y"):
                return [float(lm.x), float(lm.y)]
            elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
                return [float(lm[0]), float(lm[1])]
            return None

        # Resolve index lookups
        shoulder_idx = getattr(getattr(mp_pose, "PoseLandmark", None), "LEFT_SHOULDER", PoseLandmarksEnum).LEFT_SHOULDER
        shoulder_val = getattr(shoulder_idx, "value", shoulder_idx)
        elbow_idx = getattr(getattr(mp_pose, "PoseLandmark", None), "LEFT_ELBOW", PoseLandmarksEnum).LEFT_ELBOW
        elbow_val = getattr(elbow_idx, "value", elbow_idx)
        wrist_idx = getattr(getattr(mp_pose, "PoseLandmark", None), "LEFT_WRIST", PoseLandmarksEnum).LEFT_WRIST
        wrist_val = getattr(wrist_idx, "value", wrist_idx)
        hip_idx = getattr(getattr(mp_pose, "PoseLandmark", None), "LEFT_HIP", PoseLandmarksEnum).LEFT_HIP
        hip_val = getattr(hip_idx, "value", hip_idx)
        knee_idx = getattr(getattr(mp_pose, "PoseLandmark", None), "LEFT_KNEE", PoseLandmarksEnum).LEFT_KNEE
        knee_val = getattr(knee_idx, "value", knee_idx)
        ankle_idx = getattr(getattr(mp_pose, "PoseLandmark", None), "LEFT_ANKLE", PoseLandmarksEnum).LEFT_ANKLE
        ankle_val = getattr(ankle_idx, "value", ankle_idx)
        nose_idx = getattr(getattr(mp_pose, "PoseLandmark", None), "NOSE", PoseLandmarksEnum).NOSE
        nose_val = getattr(nose_idx, "value", nose_idx)
        foot_idx = getattr(getattr(mp_pose, "PoseLandmark", None), "LEFT_FOOT_INDEX", PoseLandmarksEnum).LEFT_FOOT_INDEX
        foot_val = getattr(foot_idx, "value", foot_idx)

        coords['shoulder'] = _get_pt(shoulder_val)
        coords['elbow'] = _get_pt(elbow_val)
        coords['wrist'] = _get_pt(wrist_val)
        coords['hip'] = _get_pt(hip_val)
        coords['knee'] = _get_pt(knee_val)
        coords['ankle'] = _get_pt(ankle_val)
        coords['nose'] = _get_pt(nose_val)
        coords['foot_index'] = _get_pt(foot_val)

        if any(v is None for v in coords.values()):
            return None
    except Exception:
        return None
    return coords


def calculate_metrics(landmarks, mp_pose=None):
    coords = get_joint_coordinates(landmarks, mp_pose)
    if not coords:
        return None

    elbow_angle = calculate_angle(coords['shoulder'], coords['elbow'], coords['wrist'])
    spine_lean = calculate_angle(coords['shoulder'], coords['hip'], [coords['hip'][0], coords['hip'][1] - 1.0]) 
    head_over_knee = abs(coords['nose'][0] - coords['knee'][0])
    foot_direction = calculate_angle([coords['ankle'][0] + 1.0, coords['ankle'][1]], coords['ankle'], coords['foot_index'])

    return {
        "elbow_angle": float(elbow_angle),
        "spine_lean": float(spine_lean),
        "head_over_knee": float(head_over_knee),
        "foot_direction": float(foot_direction),
        "wrist": coords['wrist'],
        "shoulder": coords['shoulder']
    }

