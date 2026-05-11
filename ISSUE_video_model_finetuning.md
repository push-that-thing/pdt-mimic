# feat: fine-tune video model on push-that-thing dataset (LoRA rank 64)

Fine-tune the mimic-video Cosmos-Predict2 2B video backbone with LoRA rank 64 on bird's-eye-view pushing episodes from a so101 arm. GPU compute via Nvidia Brev.

---

## Dataset

`push-that-thing/rl_world_group_10_task_1` (HuggingFace, private) — LeRobot v3 format.

| | |
|--|--|
| Episodes | 57 |
| Resolution | 480×640 |
| FPS | 30 |
| Camera | Single bird's-eye view (`observation.images.front`) |
| Task | "Push the white object to the right into the goal white circle. Stay within the two white lines." |

**Storage quirk**: LeRobot v3 packs all 57 episodes into 4 multi-episode MP4 files (7,540 / 404 / 10,937 / 7,815 frames). Episodes must be extracted with ffmpeg — simple file copy won't work.

---

## Code Changes

### 1. New: `data_preprocessing/video/convert_lerobot_to_mimic.py`

Reads the parquet index to find per-episode frame ranges within each multi-episode MP4, then extracts one clip per episode using ffmpeg's `select` filter (required for frame-accurate extraction from AV1).

```python
"""
Convert a LeRobot v3 dataset to mimic-video format.

LeRobot v3 packs multiple episodes into each video file.
This script reads the parquet index to find per-episode frame
ranges and uses ffmpeg select filter for frame-accurate AV1 extraction.

Input:
  <lerobot_dir>/videos/<video_key>/chunk-<N>/file-<M>.mp4  (multi-episode)
  <lerobot_dir>/data/chunk-<N>/file-<M>.parquet
  <lerobot_dir>/meta/tasks.parquet

Output:
  <output_dir>/video/ep_<NNN>.mp4
  <output_dir>/metas/ep_<NNN>.txt
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--video-key", default="observation.images.front")
    return p.parse_args()


def read_task_map(lerobot_dir: Path) -> dict[int, str]:
    table = pq.read_table(lerobot_dir / "meta" / "tasks.parquet")
    # Find the string column (instruction text) regardless of internal name
    str_col = next(
        f.name for f in table.schema
        if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)
    )
    return {int(i): str(s) for i, s in zip(
        table["task_index"].to_pylist(), table[str_col].to_pylist()
    )}


def extract_episode(src: Path, start: int, count: int, dst: Path) -> None:
    # select filter is frame-accurate for AV1; fast-seek (-ss before -i) is not
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"select='between(n,{start},{start+count-1})',setpts=PTS-STARTPTS",
        "-vsync", "vfr", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(dst),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    args = parse_args()
    (args.output_dir / "video").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metas").mkdir(parents=True, exist_ok=True)

    task_map = read_task_map(args.lerobot_dir)
    video_root = args.lerobot_dir / "videos" / args.video_key

    for parquet_path in sorted((args.lerobot_dir / "data").rglob("*.parquet")):
        rel = parquet_path.relative_to(args.lerobot_dir / "data")
        video_file = video_root / rel.parent / (rel.stem + ".mp4")

        table = pq.read_table(parquet_path, columns=["episode_index", "index", "task_index"])
        ep_col = table["episode_index"].to_pylist()
        idx_col = table["index"].to_pylist()
        task_col = table["task_index"].to_pylist()

        file_offset = min(idx_col)
        ep_info: dict[int, dict] = {}
        for ep, gidx, task in zip(ep_col, idx_col, task_col):
            local = gidx - file_offset
            if ep not in ep_info:
                ep_info[ep] = {"start": local, "count": 0, "task": task}
            ep_info[ep]["count"] += 1

        for ep_idx, info in sorted(ep_info.items()):
            dst_video = args.output_dir / "video" / f"ep_{ep_idx:03d}.mp4"
            dst_meta = args.output_dir / "metas" / f"ep_{ep_idx:03d}.txt"
            if dst_video.exists():
                continue
            print(f"  ep_{ep_idx:03d}: {info['count']} frames from {video_file.name}")
            extract_episode(video_file, info["start"], info["count"], dst_video)
            dst_meta.write_text(task_map[info["task"]])

    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
```

```bash
# Run locally (no GPU needed). Requires: pip install pyarrow  +  ffmpeg in PATH
python data_preprocessing/video/convert_lerobot_to_mimic.py \
  --lerobot-dir ./data/push-that-thing \
  --output-dir  ./data/push-mimic
```

Output:
```
data/push-mimic/
├── video/   ep_000.mp4 … ep_056.mp4   (57 clips, h264)
├── metas/   ep_000.txt … ep_056.txt   (57 identical instructions)
└── t5_xxl/                             (generated on GPU machine)
```

---

### 2. Edit: `model/cosmos_predict2/configs/defaults/data_video.py`

Add `push` to `train_datasets` after the last libero entry:

```python
"push": L(Dataset)(
    dataset_dir=...,        # passed as CLI override at training time
    num_frames=61,
    video_size=[480, 640],
    data_fps=30.0,
    is_val=False,
    obs_history=5,
),
```

`data_fps=30.0` → sampling step = 30/5 = 6 frames → 5 Hz effective.
No `is_multi_img` (default `False`) → T5 pickle name = `ep_NNN.pickle` matching video stem directly.

---

### 3. Done: `model/cosmos_predict2/configs/experiment/video2world.py`

`ranks = [64, 256]` is already set. With `push` registered above, the loop generates:
```
v2w_push_lora_rank64_lr1.778e-04_bsz32
```

---

## Brev Setup

Recommended: **1× H100 80GB** or A100 80GB — Ubuntu 22.04, CUDA 12.6, 200 GB+ disk.

```bash
curl -sL https://brev.dev/install-cli.sh | bash
brev create <name> --gpu h100_80gb --disk 200
brev shell <name>
```

---

## Steps

### Local (before SSH)

```bash
# 1. Convert dataset
pip install pyarrow
python mimic-video/data_preprocessing/video/convert_lerobot_to_mimic.py \
  --lerobot-dir ./data/push-that-thing \
  --output-dir  ./data/push-mimic

# 2. Upload to Brev
brev scp -r ./data/push-mimic <name>:/home/user/data/push-mimic
```

### On Brev

```bash
# 3. Clone
git clone --recurse-submodules https://github.com/TheoLeFur/robot_learning_wm.git
cd robot_learning_wm/mimic-video/model

# 4. Environment
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.cargo/env
uv sync --extra cu126 && source .venv/bin/activate

# 5. Checkpoints  (needs: text_encoder/t5-11b/, video_backbone/tokenizer/, video_backbone/v2w_pretrained_cosmos.pt)
python scripts/download_checkpoints.py

# 6. T5 embeddings  →  produces push-mimic/t5_xxl/ep_000.pickle … ep_056.pickle
cd ../data_preprocessing/video/
python get_t5_embeddings.py --dataset_path /home/user/data/push-mimic

# 7. VAE latents  →  produces push-mimic/vae_latents/ep_NNN_start*.pt (~12 GB, bfloat16)
#    This pre-computes the frozen tokenizer once so it is never re-run during training.
#    Expected: ~15 clips/episode × 57 episodes ≈ 855 files, ~2.3 MB each.
python get_vae_latents.py \
  --dataset_path /home/user/data/push-mimic \
  --vae_pth      /home/user/robot_learning_wm/mimic-video/model/checkpoints/video_backbone/tokenizer/tokenizer.pth

# 9. Train
cd ~/robot_learning_wm/mimic-video/model
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_FUSED_ATTN=0 \
torchrun --nproc_per_node=1 --master_port=12341 \
  -m scripts.train --config=cosmos_predict2/configs/config.py -- \
  experiment=v2w_push_lora_rank64_lr1.778e-04_bsz32 \
  video_dataset_train.dataset_dir=/home/user/data/push-mimic \
  video_dataset_val.dataset_dir=/home/user/data/push-mimic \
  dataloader_train.batch_size=4 \
  trainer.grad_accum_iter=8 \
  trainer.job.path_local=/home/user/checkpoints/push-v2w
```

> `batch_size=4` × `grad_accum_iter=8` = effective bsz 32 (matching the experiment LR). If OOM: `batch_size=2`, `grad_accum_iter=16`.

---

## Checklist

- [ ] Conversion: 57 `video/` clips + 57 `metas/` files
- [ ] T5: 57 `.pickle` files in `t5_xxl/`
- [ ] VAE latents: `vae_latents/` populated (expect ~855 `.pt` files, ~12 GB)
- [ ] Training starts, loss logged every 1000 steps
- [ ] Checkpoint saved to `/home/user/checkpoints/push-v2w`

---

## Files

| File | Action |
|------|--------|
| `data_preprocessing/video/convert_lerobot_to_mimic.py` | **Create** |
| `model/cosmos_predict2/configs/defaults/data_video.py` | **Edit** — add `push` entry |
| `model/cosmos_predict2/configs/experiment/video2world.py` | **Done** — ranks already updated |
