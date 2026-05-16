#!/usr/bin/env bash
# Fetch the action-decoder + fused video-backbone checkpoints that the
# `mimic_video` LeRobot policy boots from by default.
#
# Defaults track the latest combined-LoRA chain run on HF:
#   action_decoder/chain/w2a_so101_combined_lora32_iter2500_75m_2gpu_bsz2_accum8_val5/
#   checkpoints/model/iter_000003400.pt    (latest_checkpoint.txt)
#
# and the matching fused Cosmos backbone (fused locally because HF only has the
# unfused LoRA adapters for iter_000002500).
#
# Usage:
#   bash scripts/fetch_deploy_checkpoints.sh                     # full fetch + fuse
#   ACTION_ITER=3200 bash scripts/fetch_deploy_checkpoints.sh    # pick a different iter
#   DRY_RUN=1 bash scripts/fetch_deploy_checkpoints.sh           # print actions only
#
# Requires: the mimic-video venv (for the fuse step) and an HF token with
# read access to the push-that-thing repos.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MIMIC_DIR="$ROOT_DIR/mimic-video"
MODEL_DIR="$MIMIC_DIR/model"
CKPT_DIR="$MODEL_DIR/checkpoints"

ACTION_REPO="${ACTION_REPO:-push-that-thing/model}"
ACTION_JOB="${ACTION_JOB:-w2a_so101_combined_lora32_iter2500_75m_2gpu_bsz2_accum8_val5}"
ACTION_ITER="${ACTION_ITER:-3400}"
ACTION_FILE="iter_$(printf '%09d' "$ACTION_ITER").pt"
ACTION_DST_DIR="$CKPT_DIR/vam/so101/$ACTION_JOB/checkpoints/model"
ACTION_DST="$ACTION_DST_DIR/$ACTION_FILE"

VIDEO_REPO="${VIDEO_REPO:-push-that-thing/cosmos-task1-task2}"
VIDEO_UNFUSED_PATH="${VIDEO_UNFUSED_PATH:-posttraining/video2world/v2w_push_combined_lora_rank32_lr1.778e-04_bsz32/checkpoints/model/iter_000002500.pt}"
VIDEO_FUSED_NAME="${VIDEO_FUSED_NAME:-v2w_push_combined_lora_rank32_lr1.778e-04_bsz32_iter_000002500_fused}"
VIDEO_DST="$CKPT_DIR/video_backbone/$VIDEO_FUSED_NAME.pt"

if [[ -z "${HF_TOKEN:-}" && -r "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be set (or saved to ~/.cache/huggingface/token via 'hf auth login')." >&2
  exit 1
fi

# Pick python: prefer the mimic venv (needed for fuse), fall back to user python.
PY="${PYTHON:-}"
if [[ -z "$PY" && -x "$MODEL_DIR/.venv/bin/python" ]]; then
  PY="$MODEL_DIR/.venv/bin/python"
fi
PY="${PY:-python3}"

echo "Using python: $PY"
echo "Action checkpoint: $ACTION_REPO :: action_decoder/chain/$ACTION_JOB/checkpoints/model/$ACTION_FILE"
echo "  -> $ACTION_DST"
echo "Video backbone:    $VIDEO_REPO :: $VIDEO_UNFUSED_PATH (fused locally)"
echo "  -> $VIDEO_DST"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p "$ACTION_DST_DIR" "$(dirname "$VIDEO_DST")"

ACTION_REPO="$ACTION_REPO" \
ACTION_JOB="$ACTION_JOB" \
ACTION_FILE="$ACTION_FILE" \
ACTION_DST="$ACTION_DST" \
"$PY" - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.environ["ACTION_REPO"]
job = os.environ["ACTION_JOB"]
fname = os.environ["ACTION_FILE"]
dst = Path(os.environ["ACTION_DST"])

if dst.exists():
    print(f"Action ckpt already present at {dst}, skipping download.")
else:
    path = hf_hub_download(
        repo_id=repo,
        filename=f"action_decoder/chain/{job}/checkpoints/model/{fname}",
        repo_type="model",
    )
    shutil.copyfile(path, dst)
    print(f"Downloaded {dst}")
PY

if [[ -f "$VIDEO_DST" ]]; then
  echo "Video backbone already present at $VIDEO_DST, skipping fuse."
  exit 0
fi

if [[ ! -x "$MODEL_DIR/.venv/bin/python" ]]; then
  echo "ERROR: fuse step needs the mimic-video venv at $MODEL_DIR/.venv." >&2
  echo "       Run setup first (uv sync --extra cu126 in $MODEL_DIR), or fuse on another box and copy $VIDEO_DST in." >&2
  exit 1
fi

VIDEO_REPO="$VIDEO_REPO" \
VIDEO_UNFUSED_PATH="$VIDEO_UNFUSED_PATH" \
VIDEO_DST="$VIDEO_DST" \
FUSE_SCRIPT="$MODEL_DIR/scripts/fuse_lora_ckpt.py" \
"$MODEL_DIR/.venv/bin/python" - <<'PY'
import os
import shutil
import subprocess
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.environ["VIDEO_REPO"]
unfused = os.environ["VIDEO_UNFUSED_PATH"]
dst = Path(os.environ["VIDEO_DST"])
fuse_script = Path(os.environ["FUSE_SCRIPT"])

print(f"Downloading {repo}::{unfused}")
src = Path(hf_hub_download(repo_id=repo, filename=unfused, repo_type="model"))

print(f"Fusing LoRA adapters via {fuse_script}")
subprocess.run([sys.executable, str(fuse_script), str(src)], check=True)
fused = src.with_name(src.stem + "_fused.pt")
shutil.copyfile(fused, dst)
print(f"Wrote {dst}")
PY
