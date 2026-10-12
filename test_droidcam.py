"""
test_droidcam.py
================
Quick diagnostic & verification utility to test your DroidCam stream
with OpenCV and the CricPose AI live broadcasting pipeline.

Usage:
  # Test with phone IP:
  python test_droidcam.py --ip 192.168.1.45

  # Test with USB / Virtual Webcam client:
  python test_droidcam.py --camera 1

  # Test auto-scan:
  python test_droidcam.py
"""

import sys
import time
import argparse
import cv2

def format_url(target: str) -> str:
    target = target.strip()
    if target.startswith("http://") or target.startswith("https://"):
        return target
    if ":" in target:
        host, port = target.split(":", 1)
        return f"http://{host}:{port}/video"
    return f"http://{target}:4747/video"

def test_camera_stream(source, num_frames=30):
    print("=" * 60)
    print(f"[*] Testing stream connection to: {source}")
    print("=" * 60)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[-] FAILED: Could not open camera source [{source}].")
        print("\nTroubleshooting:")
        print("1. Ensure your phone and PC are connected to the same Wi-Fi.")
        print("2. Ensure DroidCam app is running on your phone.")
        print("3. Check the Wi-Fi IP and Port (default 4747) shown on the DroidCam app.")
        print("4. In your phone's browser, you can also verify: http://<PHONE_IP>:4747")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[+] Successfully connected!")
    print(f"    Resolution : {w}x{h}")
    print(f"    Source FPS : {fps if fps > 0 else 'Variable (Network)'}")

    print(f"\n[*] Reading {num_frames} test frames...")
    start_time = time.time()
    frames_read = 0

    for i in range(num_frames):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames_read += 1
        else:
            print(f"[-] Warning: Frame {i+1} dropped or empty.")

    elapsed = time.time() - start_time
    cap.release()

    if frames_read > 0:
        actual_fps = frames_read / elapsed
        print(f"[+] Stream Healthy! Decoded {frames_read}/{num_frames} frames in {elapsed:.2f}s ({actual_fps:.1f} FPS)")
        return True
    else:
        print("[-] Stream opened but no frames could be read.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DroidCam stream for CricPose AI")
    parser.add_argument("--ip", type=str, default=None, help="DroidCam Phone IP (e.g. 192.168.1.45)")
    parser.add_argument("--camera", type=str, default=None, help="Webcam device index (e.g. 0, 1, 2)")
    args = parser.parse_args()

    if args.ip:
        url = format_url(args.ip)
        test_camera_stream(url)
    elif args.camera is not None:
        idx = int(args.camera) if args.camera.isdigit() else args.camera
        test_camera_stream(idx)
    else:
        print("No source provided. Running quick check of available webcam indices:")
        found = False
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"  [+] Camera index {i}: Available")
                cap.release()
                found = True
        if not found:
            print("  [-] No physical webcams detected.")
        print("\nTo test your DroidCam phone feed, run:")
        print("  python test_droidcam.py --ip <YOUR_PHONE_IP>")
