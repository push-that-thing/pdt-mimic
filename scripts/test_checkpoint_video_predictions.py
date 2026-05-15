"""
Sample r random episodes from task1 and task2, run video2world predictions
with a given DiT checkpoint, and save the generated videos to an output folder.

Output layout:
  <output_dir>/task1/ep_<NNN>.mp4
  <output_dir>/task1/ep_<NNN>.prompt.txt
  <output_dir>/task2/ep_<NNN>.mp4
  <output_dir>/task2/ep_<NNN>.prompt.txt

Usage:
  python scripts/test_checkpoint_video_predictions.py \
    --dit-path /path/to/checkpoint.pt \
    --output-dir ./predictions/run_xyz \
    --num-episodes 5
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "mimic-video" / "model"
sys.path.insert(0, str(MODEL_DIR))

from scripts.run_video2world import (  # noqa: E402
    _DEFAULT_NEGATIVE_PROMPT,
    cleanup_distributed,
    process_single_generation,
    setup_pipeline,
)

NUM_CONDITIONAL_FRAMES = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dit-path", type=str, required=True, help="Path to DiT checkpoint.")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to save generated videos.")
    p.add_argument("--num-episodes", "-r", type=int, required=True, help="Episodes to sample per task.")
    p.add_argument(
        "--task1-dir",
        type=Path,
        default=REPO_ROOT / "data" / "push-mimic-task1",
        help="Root of task1 dataset (expects video/ and metas/ subdirs).",
    )
    p.add_argument(
        "--task2-dir",
        type=Path,
        default=REPO_ROOT / "data" / "push-mimic-task2",
        help="Root of task2 dataset (expects video/ and metas/ subdirs).",
    )
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load-ema", action="store_true")
    p.add_argument("--disable-guardrail", action="store_true")
    p.add_argument("--offload-guardrail", action="store_true")
    p.add_argument("--offload-text-encoder", action="store_true")
    p.add_argument("--downcast-text-encoder", action="store_true")
    p.add_argument("--use-cuda-graphs", action="store_true")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument(
        "--negative-prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
    )
    p.add_argument("--aspect-ratio", type=str, default="4:3")
    return p.parse_args()


def extract_first_frames(src: Path, dst: Path, n_frames: int) -> None:
    """Write a clip containing exactly the first n_frames of `src` to `dst`."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        cap.release()
        raise RuntimeError(f"{src}: has {total} frames, need {n_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"{src}: failed reading frame {i} (need first {n_frames})")
        frames.append(frame)
    cap.release()

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {dst}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def sample_episodes(task_dir: Path, n: int, rng: random.Random) -> list[tuple[Path, Path]]:
    video_dir = task_dir / "video"
    metas_dir = task_dir / "metas"
    candidates = []
    for video_path in sorted(video_dir.glob("ep_*.mp4")):
        meta_path = metas_dir / f"{video_path.stem}.txt"
        if meta_path.exists():
            candidates.append((video_path, meta_path))
    if not candidates:
        raise FileNotFoundError(f"No paired episodes under {task_dir}")
    if n > len(candidates):
        raise ValueError(f"Requested {n} episodes but only {len(candidates)} available in {task_dir}")
    return rng.sample(candidates, n)


def build_pipeline_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dit_path=args.dit_path,
        load_ema=args.load_ema,
        seed=args.seed,
        num_gpus=args.num_gpus,
        disable_guardrail=args.disable_guardrail,
        offload_guardrail=args.offload_guardrail,
        offload_text_encoder=args.offload_text_encoder,
        downcast_text_encoder=args.downcast_text_encoder,
        save_path=str(args.output_dir / "_pipeline_config.yaml"),
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    tasks = {"task1": args.task1_dir, "task2": args.task2_dir}
    selections = {name: sample_episodes(path, args.num_episodes, rng) for name, path in tasks.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in tasks:
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    cond_root = args.output_dir / "_conditioning"
    for name in tasks:
        (cond_root / name).mkdir(parents=True, exist_ok=True)

    pipe = setup_pipeline(build_pipeline_args(args))
    try:
        for name, episodes in selections.items():
            for video_path, meta_path in episodes:
                prompt = meta_path.read_text(encoding="utf-8").strip()
                output_path = args.output_dir / name / f"{video_path.stem}.mp4"
                prompt_copy = output_path.with_suffix(".prompt.txt")
                prompt_copy.write_text(prompt + "\n", encoding="utf-8")
                # Condition on the FIRST N frames of the episode. The pipeline
                # internally takes the last N frames of its input clip, so we
                # pre-trim to a clip that contains only frames 0..N-1 — the
                # "last 5" of that 5-frame clip == the first 5 of the episode.
                cond_clip = cond_root / name / f"{video_path.stem}.mp4"
                extract_first_frames(video_path, cond_clip, NUM_CONDITIONAL_FRAMES)
                process_single_generation(
                    pipe=pipe,
                    input_path=str(cond_clip),
                    prompt=prompt,
                    output_path=str(output_path),
                    negative_prompt=args.negative_prompt,
                    aspect_ratio=args.aspect_ratio,
                    num_conditional_frames=NUM_CONDITIONAL_FRAMES,
                    guidance=args.guidance,
                    seed=args.seed,
                    benchmark=False,
                    use_cuda_graphs=args.use_cuda_graphs,
                )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
