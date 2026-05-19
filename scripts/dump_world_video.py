"""Run only the video2world model to full denoising and dump the predicted
future-frames video to disk. Useful for inspecting what the world model is
imagining, separate from the action head.

Usage (from repo root, after sourcing scripts/mimic_env.sh):
  python scripts/dump_world_video.py --image snapshot.jpg --out /tmp/world.mp4
  python scripts/dump_world_video.py --camera 0 --out /tmp/world.mp4
  python scripts/dump_world_video.py --camera 0 --steps 35 --prompt "..."

`--steps` is the number of diffusion sampling steps. Default 35 = full quality
(takes ~2-3 minutes on Thor). Lower for faster but blurrier previews.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import lerobot_policy_mimic_video  # noqa: F401 — registers the policy
from lerobot_policy_mimic_video import MimicVideoConfig, MimicVideoPolicy


def grab_frame(camera_idx: int, width: int, height: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(3):  # discard warmup frames
        cap.read()
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read from /dev/video{camera_idx}.")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def load_frame(path: Path, width: int, height: int) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))
    return img


def write_mp4(frames_uint8_thwc: np.ndarray, out_path: Path, fps: int) -> None:
    import imageio.v3 as iio
    iio.imwrite(out_path, frames_uint8_thwc, fps=fps, codec="libx264")


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Input snapshot (jpg/png).")
    src.add_argument("--camera", type=int, help="Capture from /dev/videoN.")
    prompt_src = p.add_mutually_exclusive_group()
    prompt_src.add_argument("--prompt", default=None, help="Task prompt. Default = config default (task1).")
    prompt_src.add_argument(
        "--task", choices=["task1", "task2"],
        help="Load the prompt from the cached T5 embedding for the named task. "
             "Avoids unicode apostrophe mismatch.",
    )
    p.add_argument("--t5-cache-dir", type=Path,
                   default=Path.home() / ".cache" / "mimic_lerobot_t5" / ".t5_embeddings",
                   help="Where the cached *.npz embeddings live.")
    p.add_argument("--steps", type=int, default=35, help="Diffusion sampling steps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("/tmp/world.mp4"))
    p.add_argument("--fps", type=int, default=30, help="Output mp4 fps.")
    args = p.parse_args()

    cfg = MimicVideoConfig()
    # Skip loading the T5-11B text encoder; we rely on the precomputed prompt
    # embedding cache (same flow as the deploy script).
    cfg.use_text_encoder = False

    TASK_NPZ_HASHES = {
        "task1": "aaf73a5d3f7ba6e1bc918cf961c86791aa61ee83397d05bd69759262d861c508",
        "task2": "5c53708f764ce3095860f6b20d62b352473c79d86d6e78cbd556789b4230c0f9",
    }
    if args.task:
        npz_path = args.t5_cache_dir / f"{TASK_NPZ_HASHES[args.task]}.npz"
        with np.load(npz_path, allow_pickle=False) as d:
            cfg.task = d.files[0]
        print(f"[dump] loaded prompt for {args.task}: {cfg.task!r}")
    elif args.prompt:
        cfg.task = args.prompt

    print(f"[dump] loading policy …", flush=True)
    t0 = time.time()
    policy = MimicVideoPolicy(cfg)
    print(f"[dump] policy loaded in {time.time() - t0:.1f}s", flush=True)

    if args.image:
        rgb = load_frame(args.image, cfg.image_width, cfg.image_height)
    else:
        rgb = grab_frame(args.camera, cfg.image_width, cfg.image_height)
    print(f"[dump] input frame shape={rgb.shape}", flush=True)

    # Build vid_input with T=1 (single conditioning frame), shape (1, C, 1, H, W),
    # values in [0, 1].
    frame_chw = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    vid_input = frame_chw.unsqueeze(0).unsqueeze(2).to(
        next(policy.model.parameters()).device
    )

    video_pipe = policy.model.video2world_pipeline
    print(f"[dump] running {args.steps}-step video diffusion …", flush=True)
    t0 = time.time()
    decoded = video_pipe.generate_video(
        vid_input=vid_input,
        num_latent_conditional_frames=1,
        prompt=cfg.task,
        prompt_embedding=policy._prompt_embedding,
        negative_prompt="",
        guidance=0.0,
        num_sampling_step=args.steps,
        seed=args.seed,
        use_cuda_graphs=False,
        return_context_at_step=None,
        hidden_state_layer_idx=None,
    )
    print(f"[dump] diffusion done in {time.time() - t0:.1f}s, shape={tuple(decoded.shape)}", flush=True)

    # decoded: (B, C, T, H, W) roughly in [-1, 1]. Convert to (T, H, W, C) uint8.
    video = decoded[0].clamp(-1, 1).add(1).mul(127.5).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).contiguous().cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(video, args.out, args.fps)
    print(f"[dump] wrote {args.out} ({video.shape[0]} frames @ {args.fps} fps)", flush=True)


if __name__ == "__main__":
    main()
