#!/usr/bin/env bash
# Set up the video model: download the finetuned LoRA checkpoint, sync the uv
# env, fuse adapters into the base weights, and activate the venv.
#
# Usage:  source scripts/setup_video_model.sh

HF_REPO="push-that-thing/cosmos-task1-task2"
EXPERIMENT_PATH="posttraining/video2world/v2w_push_combined_lora_rank32_lr1.778e-04_bsz32/checkpoints"

cat <<'BANNER'
================================================================================
  setup_video_model.sh
================================================================================
  This script prepares the push-that-thing video model for inference. It will:

    1. Download the finetuned LoRA DiT checkpoint from
       push-that-thing/cosmos-task1-task2 into <parent>/checkpoints/video-model/.
    2. Download the Cosmos VAE tokenizer from jonpai/mimic-video into
       <parent>/checkpoints/tokenizer/.
    3. Download precomputed T5 embeddings for task1 and task2 from
       push-that-thing/task_1 and push-that-thing/task_2 into
       <parent>/embeddings/{task1,task2}/.
    4. Install uv (if missing) and sync the cu126 env in mimic-video/model/.
    5. Fuse the LoRA adapters into the base weights, producing *_fused.pt
       next to the downloaded checkpoint.
    6. Activate the venv (only persists when this file is sourced).

  Requirements: hf CLI authenticated (hf auth login), Linux/x86_64 for the
  uv sync step (the lockfile is linux-only).

  Usage:  source scripts/setup_video_model.sh
================================================================================
BANNER

# Run the work in a strict sub-shell so `set -e` does not leak when sourced.
(
  set -euo pipefail

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "$script_dir/.." && pwd)"
  checkpoints_dir="$(cd "$repo_root/.." && pwd)/checkpoints/video-model"
  model_dir="$repo_root/mimic-video/model"

  mkdir -p "$checkpoints_dir"

  # 1. Download latest_checkpoint.txt, read the filename, fetch that .pt.
  echo "==> Downloading latest_checkpoint.txt from $HF_REPO"
  hf download "$HF_REPO" "$EXPERIMENT_PATH/latest_checkpoint.txt" \
    --local-dir "$checkpoints_dir"

  ckpt_name="$(tr -d '[:space:]' < "$checkpoints_dir/$EXPERIMENT_PATH/latest_checkpoint.txt")"
  echo "    latest checkpoint: $ckpt_name"

  echo "==> Downloading $EXPERIMENT_PATH/model/$ckpt_name"
  hf download "$HF_REPO" "$EXPERIMENT_PATH/model/$ckpt_name" \
    --local-dir "$checkpoints_dir"

  dst_ckpt="$checkpoints_dir/$ckpt_name"
  mv -f "$checkpoints_dir/$EXPERIMENT_PATH/model/$ckpt_name" "$dst_ckpt"

  # 2. Download the VAE tokenizer to <parent>/checkpoints/tokenizer/.
  tokenizer_dir="$(cd "$repo_root/.." && pwd)/checkpoints/tokenizer"
  mkdir -p "$tokenizer_dir"
  echo "==> Downloading VAE tokenizer from jonpai/mimic-video"
  hf download jonpai/mimic-video "video_backbone/tokenizer/tokenizer.pth" \
    --local-dir "$tokenizer_dir"
  mv -f "$tokenizer_dir/video_backbone/tokenizer/tokenizer.pth" "$tokenizer_dir/tokenizer.pth"
  rm -rf "$tokenizer_dir/video_backbone"

  # 3. Download T5 embeddings for task1 and task2.
  emb_root="$(cd "$repo_root/.." && pwd)/embeddings"
  mkdir -p "$emb_root"
  for pair in "task1:task_1" "task2:task_2"; do
    local_name="${pair%%:*}"
    repo_name="${pair##*:}"
    echo "==> Downloading T5 embeddings from push-that-thing/$repo_name"
    tmp_dl="$(mktemp -d)"
    hf download "push-that-thing/$repo_name" --repo-type dataset \
      --include ".t5_embeddings/*" --local-dir "$tmp_dl"
    rm -rf "$emb_root/$local_name"
    mv "$tmp_dl/.t5_embeddings" "$emb_root/$local_name"
    rm -rf "$tmp_dl"
  done

  # 4. Install uv (if missing) and sync env.
  if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env"
  fi

  echo "==> uv sync --extra cu126 in $model_dir"
  cd "$model_dir"
  uv sync --extra cu126

  # 5. Fuse LoRA adapters into base weights.
  echo "==> Fusing LoRA adapters: $dst_ckpt"
  uv run python scripts/fuse_lora_ckpt.py "$dst_ckpt"
)
status=$?
[[ $status -ne 0 ]] && { return $status 2>/dev/null || exit $status; }

# 6. Activate the venv in the caller's shell.
__svm_venv="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/mimic-video/model/.venv/bin/activate"
# shellcheck disable=SC1090
source "$__svm_venv"
echo "==> activated venv: $(command -v python)"
unset __svm_venv
