#!/usr/bin/env bash
# Shared driver for run_eval_task1.sh / run_eval_task2.sh.
#
# The caller exports the per-task variables below, then sources this file:
#   TASK_PROMPT        prompt text (must match a key in the task's T5 cache)
#   ACTION_EXP         action-decoder experiment dir name on push-that-thing/model
#   ACTION_ITER        action-decoder iteration to deploy (e.g. 2500)
#   STATS_HASH         sha256 of the dataset-statistics cache on push-that-thing/model
#   EXPERIMENT_NAME    mimic-video hydra experiment the decoder was trained under
#   HF_DATASET_REPO    dataset repo whose .t5_embeddings cache to warm
#   VIDEO_CKPT_FILE    fused video backbone path inside HF_MODEL_REPO
#   VIDEO_CKPT_NAME    local basename for the video backbone (no .pt)
#   HF_MODEL_REPO      repo holding the video backbone (default cosmos-task1-task2)
#
# Flags (env):
#   LIVE=1     actually drive the arm (default: dry-run, prints the command only)
#   ROBOT_PORT, ROBOT_ID, CAMERA_IDX  forwarded to deploy_so101.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PDT_DIR="$REPO_ROOT"
MODEL_DIR="$REPO_ROOT/mimic-video/model"

AD_REPO="push-that-thing/model"
HF_MODEL_REPO="${HF_MODEL_REPO:-push-that-thing/cosmos-task1-task2}"
export HF_MODEL_REPO HF_DATASET_REPO VIDEO_CKPT_FILE VIDEO_CKPT_NAME

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN before running (needed for private HF assets)." >&2
  exit 1
fi

# 1. Environment: create the mimic-video venv + base model downloads once.
if [[ ! -x "$MODEL_DIR/.venv/bin/python" ]]; then
  echo "==> First run: bootstrapping environment via .brev/setup.sh"
  bash "$REPO_ROOT/.brev/setup.sh"
fi

# 2. Activate venv (also sets HF_HOME, CUDA libs, PYTHONPATH).
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/mimic_env.sh"
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"

# 3. Install the BYOP policy plugin + SO-101 hardware deps (idempotent).
if ! python -c "import lerobot_policy_mimic_video" >/dev/null 2>&1; then
  echo "==> Installing lerobot_policy_mimic_video plugin"
  uv pip install -e "$REPO_ROOT/lerobot_policy_mimic_video"
fi
if ! python -c "import feetech_servo_sdk" >/dev/null 2>&1; then
  echo "==> Installing SO-101 hardware deps (feetech-servo-sdk, pyserial)"
  uv pip install -r "$REPO_ROOT/requirements-deploy.txt"
fi

# 4. Download the task's action decoder + statistics cache.
ckpt_name="$(printf 'iter_%09d.pt' "$ACTION_ITER")"
ad_dir="$MODEL_DIR/checkpoints/vam/so101/$ACTION_EXP/checkpoints/model"
ACTION_CKPT="$ad_dir/$ckpt_name"
if [[ ! -f "$ACTION_CKPT" ]]; then
  echo "==> Downloading action decoder $ACTION_EXP/$ckpt_name"
  mkdir -p "$ad_dir"
  tmp="$(mktemp -d)"
  hf download "$AD_REPO" "action_decoder/chain/$ACTION_EXP/checkpoints/model/$ckpt_name" --local-dir "$tmp"
  mv -f "$tmp/action_decoder/chain/$ACTION_EXP/checkpoints/model/$ckpt_name" "$ACTION_CKPT"
  rm -rf "$tmp"
fi

stats_dir="$HOME/.cache/mimic_lerobot_stats"
STATS_PATH="$stats_dir/$STATS_HASH"
if [[ ! -f "$STATS_PATH" ]]; then
  echo "==> Downloading dataset statistics $STATS_HASH"
  mkdir -p "$stats_dir"
  tmp="$(mktemp -d)"
  hf download "$AD_REPO" "action_decoder/stats_cache/$STATS_HASH" --local-dir "$tmp"
  mv -f "$tmp/action_decoder/stats_cache/$STATS_HASH" "$STATS_PATH"
  rm -rf "$tmp"
fi

VIDEO_CKPT_PATH="$MODEL_DIR/checkpoints/video_backbone/${VIDEO_CKPT_NAME}.pt"

# 5. Deploy. Dry-run by default; LIVE=1 drives the arm.
DRY_RUN=1
[[ "${LIVE:-0}" == "1" ]] && DRY_RUN=0

echo ""
echo "==> Action decoder : $ACTION_CKPT"
echo "==> Video backbone : $VIDEO_CKPT_PATH"
echo "==> Statistics     : $STATS_PATH"
echo "==> Experiment     : $EXPERIMENT_NAME"
echo "==> Prompt         : $TASK_PROMPT"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "==> DRY-RUN (no movement). Re-run with LIVE=1 to drive the arm."
else
  echo "==> LIVE: the arm WILL move. Ensure it is calibrated and clear."
fi
echo ""

export TASK_PROMPT ACTION_CKPT VIDEO_CKPT_PATH EXPERIMENT_NAME STATS_PATH
DRY_RUN="$DRY_RUN" bash "$REPO_ROOT/scripts/deploy_so101.sh"
