"""Render the cosmos backbone's predicted future video and save it.

During normal policy inference the wrapper stops the diffusion early
(`stop_after_step=30`) and pulls cross-attention hidden states for the action
decoder — the actual predicted video is never decoded to RGB. This script
forces the full 35-step diffusion AND a tokenizer decode, so we can SEE what
the cosmos backbone is hallucinating.

Reads one frame from the SO-101 front camera (or from a saved PNG), runs the
full diffusion, decodes the latents back to RGB frames, and writes:
  /tmp/cosmos_input_frame.png             — the single input observation
  /tmp/cosmos_future_frames/frame_NN.png  — every predicted frame
  /tmp/cosmos_future.mp4                  — stitched at 5 FPS (slow playback)

Cost: ~3-4 min on the 3060 (one full diffusion). Run via:
    source scripts/mimic_env.sh
    python scripts/inspect_video_backbone.py             # live camera
    python scripts/inspect_video_backbone.py --input /tmp/cam_live.png
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("AV_LOG_LEVEL", "quiet")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot_policy_mimic_video import MimicVideoConfig, MimicVideoPolicy


def load_input_frame(path: str | None, camera_idx: int) -> np.ndarray:
    """Return one HxWxC RGB uint8 frame at 480x640."""
    if path:
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"Could not read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_idx}")
        frame = None
        for _ in range(5):  # warm-up
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"camera {camera_idx} read returned False")
        cap.release()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if rgb.shape[:2] != (480, 640):
        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Path to a starting RGB frame PNG. Defaults to live camera.")
    ap.add_argument("--camera-idx", type=int, default=0)
    ap.add_argument("--num-sampling-step", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/tmp/cosmos_future_frames")
    ap.add_argument("--out-mp4", default="/tmp/cosmos_future.mp4")
    ap.add_argument("--out-mp4-fps", type=int, default=5, help="Playback FPS for the MP4 (5 = slow, easy to watch).")
    args = ap.parse_args()

    print("[1/5] Loading policy (cosmos backbone + decoder; we only use the backbone here)...")
    cfg = MimicVideoConfig(use_text_encoder=False, load_on_init=False, n_action_steps=90)
    cfg.input_features = cfg.default_input_features
    cfg.output_features = cfg.default_output_features
    policy = MimicVideoPolicy(cfg)
    policy.load_mimic_pipeline()
    assert policy.model is not None
    video_pipe = policy.model.video2world_pipeline
    print("       policy loaded.")

    print("[2/5] Loading input frame...")
    frame_rgb = load_input_frame(args.input, args.camera_idx)
    print(f"       frame shape={frame_rgb.shape} dtype={frame_rgb.dtype}")
    cv2.imwrite("/tmp/cosmos_input_frame.png", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    # Build vid_input shape (1, C=3, T=5, H, W) uint8 — duplicate single frame
    # across the 5-frame history (same pad the live wrapper does).
    chw = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()  # (3, 480, 640) uint8
    cthw = chw.unsqueeze(1).expand(-1, 5, -1, -1).contiguous()       # (3, 5, 480, 640)
    vid_input = cthw.unsqueeze(0).to("cuda")                          # (1, 3, 5, 480, 640)

    print(f"[3/5] Running cosmos diffusion for {args.num_sampling_step} steps + tokenizer decode...")
    print("       (~3-4 minutes on the 3060)")
    t0 = time.time()
    with torch.no_grad():
        decoded = video_pipe.generate_video(
            vid_input=vid_input,
            num_latent_conditional_frames=2,  # matches Video2World2ActionPipeline's choice for T=5
            prompt="",
            prompt_embedding=policy._prompt_embedding,
            negative_prompt="",
            guidance=0.0,
            num_sampling_step=args.num_sampling_step,
            seed=args.seed,
            use_cuda_graphs=False,
            return_context_at_step=None,  # decode the full video to RGB instead of stopping early
        )
    print(f"       generated in {time.time() - t0:.1f}s   shape={tuple(decoded.shape)}   range=[{decoded.min().item():.3f}, {decoded.max().item():.3f}]")

    # decoded is (B, C=3, T, H, W) in roughly [-1, 1]. Convert to uint8 RGB.
    vid = decoded[0].detach().cpu().float().clamp(-1, 1)
    vid = ((vid + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)  # (3, T, H, W)
    T = vid.shape[1]
    print(f"       got {T} frames at {vid.shape[-2]}x{vid.shape[-1]}")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print(f"[4/5] Saving {T} frames to {out_dir}/ ...")
    for t in range(T):
        rgb = vid[:, t, :, :].permute(1, 2, 0).numpy()           # (H, W, 3) RGB
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"frame_{t:03d}.png"), bgr)

    # Stitch as MP4 at slow playback speed so it's watchable.
    print(f"[5/5] Stitching MP4 to {args.out_mp4} at {args.out_mp4_fps} fps ...")
    h, w = vid.shape[-2:]
    writer = cv2.VideoWriter(args.out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.out_mp4_fps, (w, h))
    for t in range(T):
        rgb = vid[:, t, :, :].permute(1, 2, 0).numpy()
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"       saved {T} frames + {args.out_mp4}")

    print()
    print("Done. To eyeball:")
    print("  cp /tmp/cosmos_future.mp4 ./cosmos_future.mp4   # open in Cursor/file manager")
    print("  or view PNG sequence under /tmp/cosmos_future_frames/")
    print()
    print("What to look for:")
    print("  COHERENT: workspace stays put, white goal circle stays put, arm sweeps toward the object,")
    print("            gripper closes on the object, arm pushes object to circle.")
    print("            → cosmos backbone is fine, action decoder is at fault.")
    print("  HALLUCINATED: scene blurs/morphs, objects appear/disappear, arm teleports,")
    print("                colors drift, lighting changes.")
    print("                → cosmos backbone is undertrained — this is your culprit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
