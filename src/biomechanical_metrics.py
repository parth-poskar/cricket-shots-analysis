import numpy as np
import mediapipe as mp

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def get_joint_coordinates(landmarks, mp_pose):
    coords = {}
    try:
        coords['shoulder'] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        coords['elbow'] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        coords['wrist'] = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        coords['hip'] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        coords['knee'] = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        coords['ankle'] = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        coords['nose'] = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        coords['foot_index'] = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    except:
        return None
    return coords

def calculate_metrics(landmarks, mp_pose):
    coords = get_joint_coordinates(landmarks, mp_pose)
    if not coords:
        return None

    elbow_angle = calculate_angle(coords['shoulder'], coords['elbow'], coords['wrist'])
    spine_lean = calculate_angle(coords['shoulder'], coords['hip'], [coords['hip'][0], coords['hip'][1] - 1]) 
    head_over_knee = abs(coords['nose'][0] - coords['knee'][0])
    foot_direction = calculate_angle([coords['ankle'][0] + 1, coords['ankle'][1]], coords['ankle'], coords['foot_index'])

    return {
        "elbow_angle": elbow_angle,
        "spine_lean": spine_lean,
        "head_over_knee": head_over_knee,
        "foot_direction": foot_direction,
        "wrist": coords['wrist'],
        "shoulder": coords['shoulder']
    }
