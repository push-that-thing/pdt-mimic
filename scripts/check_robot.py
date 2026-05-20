"""
Live overlay: blend the camera feed with a reference image to check alignment.

Setup (conda env wm_cam already has these):
    conda create -n wm_cam python=3.11 -y
    conda activate wm_cam
    pip install opencv-python numpy

Usage:
    python live_overlay.py <reference_image>
    python live_overlay.py /home/sebastien/ETH/wm_project/pcs/overlay.png

Keys / controls:
    trackbar 'alpha'  - blend weight (0 = only reference, 100 = only camera)
    'b'   - cycle modes: blend -> camera only -> reference only -> diff -> blend
    'd'   - toggle absolute-difference view (highlights misalignment)
    's'   - save current view to live_overlay_snapshot_<timestamp>.png
    'q'   - quit
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

WIDTH = 640
HEIGHT = 480
FPS = 30
CAMERA_INDEX = 4  # matches live_camera.py; change if needed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="Path to the reference image to overlay")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    args = parser.parse_args()

    if not os.path.isfile(args.reference):
        sys.exit(f"Reference image not found: {args.reference}")

    ref = cv2.imread(args.reference)
    if ref is None:
        sys.exit(f"Could not read image: {args.reference}")
    ref = cv2.resize(ref, (args.width, args.height))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    win = "Live overlay (q=quit, b=mode, d=diff, s=save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("alpha", win, 50, 100, lambda v: None)

    modes = ["blend", "camera", "reference", "diff"]
    mode_idx = 0

    print("Live overlay running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        if frame.shape[1] != args.width or frame.shape[0] != args.height:
            frame = cv2.resize(frame, (args.width, args.height))

        mode = modes[mode_idx]
        if mode == "camera":
            view = frame
        elif mode == "reference":
            view = ref
        elif mode == "diff":
            view = cv2.absdiff(frame, ref)
        else:  # blend
            a = cv2.getTrackbarPos("alpha", win) / 100.0
            view = cv2.addWeighted(frame, a, ref, 1.0 - a, 0)

        label = f"mode: {mode}"
        cv2.putText(view, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('b'):
            mode_idx = (mode_idx + 1) % len(modes)
        elif k == ord('d'):
            mode_idx = modes.index("diff") if mode != "diff" else modes.index("blend")
        elif k == ord('s'):
            fname = f"live_overlay_snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, view)
            print(f"Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "_main_":
    main()