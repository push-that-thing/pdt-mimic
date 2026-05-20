"""No-arm smoke test: load the MIMIC pipeline and run one inference on a fake frame.

Reads the same env vars the deploy flow exports, so run it via:
    SMOKE=1 bash run_eval_task1.sh
or standalone after `source scripts/mimic_env.sh` with those vars exported.

This is the real check that the cu128 torch build + cosmos kernels run on the
GPU (e.g. Blackwell) — the deploy dry-run never loads the model.
"""
from __future__ import annotations

import os
import sys
import time

import torch

from lerobot_policy_mimic_video import MimicVideoConfig, MimicVideoPolicy

_ENV_TO_FIELD = {
    "VIDEO_CKPT_PATH": "video_checkpoint_path",
    "ACTION_CKPT": "action_checkpoint_path",
    "EXPERIMENT_NAME": "experiment_name",
    "STATS_PATH": "dataset_statistics_path",
    "PROMPT_EMB_DIR": "prompt_embedding_cache_dir",
    "TASK_PROMPT": "task",
}


def free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / (1024**3)


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False — check the venv/driver.", file=sys.stderr)
        return 2

    print(f"torch {torch.__version__} (cuda {torch.version.cuda})")
    print(f"device: {torch.cuda.get_device_name(0)}  cap: {torch.cuda.get_device_capability()}")
    print(f"arch_list: {torch.cuda.get_arch_list()}")
    print(f"GPU free before load: {free_gb():.2f} GB")

    kwargs = {"use_text_encoder": False, "load_on_init": False}
    for env, field in _ENV_TO_FIELD.items():
        val = os.environ.get(env)
        if val:
            kwargs[field] = val

    cfg = MimicVideoConfig(**kwargs)
    cfg.input_features = cfg.default_input_features
    cfg.output_features = cfg.default_output_features

    print("Loading pipeline (video backbone + action decoder + normalizer)...")
    policy = MimicVideoPolicy(cfg)
    t0 = time.time()
    policy.load_mimic_pipeline()
    print(f"  loaded in {time.time() - t0:.1f}s   GPU free: {free_gb():.2f} GB")

    print("Running one forward pass on a fake observation...")
    fake = {
        cfg.camera_key: (torch.rand(3, cfg.image_height, cfg.image_width) * 255).to(torch.uint8),
        cfg.state_key: torch.zeros(cfg.action_dim),
        "task": cfg.task,
    }
    t0 = time.time()
    chunk = policy.predict_action_chunk(fake)
    print(f"  chunk shape={tuple(chunk.shape)} in {time.time() - t0:.2f}s")
    print(f"  first action: {chunk[0, 0].tolist()}")
    print("\nSmoke test OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
