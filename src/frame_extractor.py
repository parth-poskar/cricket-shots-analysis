import cv2
import numpy as np

def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total//max_frames,1)

    i=0
    while cap.isOpened() and len(frames)<max_frames:
        ret,frame=cap.read()
        if not ret: break
        if i%step==0:
            frame=cv2.resize(frame,(224,224))/255.0
            frames.append(frame)
        i+=1

    cap.release()
    return np.array(frames)