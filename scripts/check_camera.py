"""Compare the SO-101 front camera to ALL training-episode first frames.

Writes /tmp/cam_compare.png as a grid: the LIVE camera frame in cell 0 (green
border) and the first frame of every training MP4 from push-that-thing/task_1
in the remaining cells. Use to spot viewpoint / framing / lighting drift
between deployment and training.

Also writes /tmp/cam_live.png on its own.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Silence the OpenCV/FFmpeg AV1-decode error spam — we try alternative decoders.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("AV_LOG_LEVEL", "quiet")

import cv2
import numpy as np
from huggingface_hub import hf_hub_download


def grab_live(idx: int, width: int, height: int, fps: int) -> np.ndarray:
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {idx}.")
    # Read a few frames — first frames are often dark / wrong exposure.
    frame = None
    for _ in range(5):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Camera {idx} opened but read() returned False.")
    cap.release()
    return frame  # BGR uint8


def grab_training_first_frame(video_path: str) -> np.ndarray:
    # Training MP4s on push-that-thing/task_1 are AV1-encoded — OpenCV's bundled
    # decoder can't handle them on this box. Try PyAV first (uses ffmpeg's
    # libdav1d if present), then imageio, then OpenCV as a last resort.
    errors = []
    try:
        import av  # type: ignore
        with av.open(video_path) as container:
            for frame in container.decode(video=0):
                rgb = frame.to_ndarray(format="rgb24")
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        errors.append(f"pyav: {e!r}")
    try:
        import imageio.v3 as iio  # type: ignore
        frame_rgb = iio.imread(video_path, plugin="pyav", index=0)
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        errors.append(f"imageio: {e!r}")
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if ok:
        return frame  # BGR
    errors.append("cv2.VideoCapture.read returned False")
    raise RuntimeError(
        f"Could not decode {video_path} with any backend. Tried:\n  - " + "\n  - ".join(errors)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-idx", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--training-video", default="videos/observation.images.front/chunk-000/file-000.mp4",
                    help="Path of an MP4 inside push-that-thing/task_1 to use as the reference frame.")
    args = ap.parse_args()

    out_dir = Path("/tmp")
    print(f"[1/3] Grabbing live frame from /dev/video{args.camera_idx}...")
    live = grab_live(args.camera_idx, args.width, args.height, args.fps)
    print(f"      shape={live.shape}  dtype={live.dtype}")
    cv2.imwrite(str(out_dir / "cam_live.png"), live)

    print(f"[2/3] Downloading training video from push-that-thing/task_1: {args.training_video}")
    token = os.environ.get("HF_TOKEN")
    vid_path = hf_hub_download(
        repo_id="push-that-thing/task_1",
        filename=args.training_video,
        repo_type="dataset",
        token=token,
    )
    training = grab_training_first_frame(vid_path)
    print(f"      shape={training.shape}  dtype={training.dtype}")
    cv2.imwrite(str(out_dir / "cam_training.png"), training)

    # Resize to common height for side-by-side
    target_h = max(live.shape[0], training.shape[0])
    def to_h(img):
        h, w = img.shape[:2]
        if h == target_h:
            return img
        new_w = int(round(w * target_h / h))
        return cv2.resize(img, (new_w, target_h))

    live_r = to_h(live)
    training_r = to_h(training)
    gap = np.full((target_h, 8, 3), 32, dtype=np.uint8)
    panel = np.hstack([live_r, gap, training_r])
    # label
    cv2.putText(panel, "LIVE  (now)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(panel, "TRAINING (push-that-thing/task_1)", (live_r.shape[1] + 18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / "cam_compare.png"), panel)

    print("[3/3] Wrote:")
    for p in ["cam_live.png", "cam_training.png", "cam_compare.png"]:
        f = out_dir / p
        print(f"  {f}  ({f.stat().st_size // 1024} KiB)")
    print("\nOpen /tmp/cam_compare.png to eyeball the match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
