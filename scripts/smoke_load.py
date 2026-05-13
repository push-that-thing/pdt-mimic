"""Smoke-test the MIMIC video policy: load weights, run one chunk on a fake frame.

Doesn't touch the SO-101 — use to confirm the env, checkpoints, and the T5
embedding-skip patch all wire up before going to hardware. Run via:

    source scripts/mimic_env.sh
    python scripts/smoke_load.py
"""
from __future__ import annotations

import sys
import time

import torch

from lerobot_policy_mimic_video import MimicVideoConfig, MimicVideoPolicy


def free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / (1024**3)


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False — check LD_LIBRARY_PATH/CUDA_HOME.", file=sys.stderr)
        return 2

    print(f"GPU free before load: {free_gb():.2f} GB")
    print("Building MimicVideoConfig (use_text_encoder=False)...")
    cfg = MimicVideoConfig(
        use_text_encoder=False,
        load_on_init=False,
    )
    cfg.input_features = cfg.default_input_features
    cfg.output_features = cfg.default_output_features

    print("Instantiating policy...")
    policy = MimicVideoPolicy(cfg)

    print("Loading mimic pipeline (cosmos backbone + decoder)...")
    t0 = time.time()
    policy.load_mimic_pipeline()
    print(f"  loaded in {time.time() - t0:.1f}s   GPU free after: {free_gb():.2f} GB")

    print("\nRunning one forward pass with a fake observation...")
    H, W = cfg.image_height, cfg.image_width
    fake_image = (torch.rand(3, H, W) * 255).to(torch.uint8)
    fake_state = torch.zeros(cfg.action_dim)
    batch = {
        cfg.camera_key: fake_image,
        cfg.state_key: fake_state,
        "task": cfg.task,
    }
    t0 = time.time()
    chunk = policy.predict_action_chunk(batch)
    print(f"  predict_action_chunk returned shape={tuple(chunk.shape)} in {time.time() - t0:.2f}s")
    print(f"  first action: {chunk[0, 0].tolist()}")
    print(f"  GPU free after forward: {free_gb():.2f} GB")
    print("\nSmoke test OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
