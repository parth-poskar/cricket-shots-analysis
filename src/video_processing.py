import cv2
import os
import yt_dlp

def download_video(url, output_path):
    if os.path.exists(output_path):
        print("Video already exists.")
        return
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def setup_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None, None, None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, width, height, fps

def setup_video_writer(output_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Use H.264 codec for better browser compatibility
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return out
