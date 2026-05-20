#!/usr/bin/env bash
# Shared driver for run_eval_task1.sh / run_eval_task2.sh.
#
# The caller exports the per-task variables, then sources this file:
#   TASK_PROMPT        prompt text (must match a key in the task's T5 cache)
#   ACTION_EXP         action-decoder experiment dir on push-that-thing/model
#   ACTION_ITER        action-decoder iteration to deploy (e.g. 2500)
#   STATS_HASH         sha256 of the dataset-statistics cache
#   EXPERIMENT_NAME    mimic-video hydra experiment the decoder was trained under
#   PROMPT_EMB_SUBDIR  T5 cache subdir warmed by setup.sh (task1 / task2)
#   VIDEO_CKPT_NAME    basename of the shared video backbone (no .pt)
#
# Flags (env):  LIVE=1 drives the arm (default: dry-run / prints the command).
#               ROBOT_PORT, ROBOT_ID, CAMERA_IDX forwarded to deploy_so101.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PDT_DIR="$REPO_ROOT"
MODEL_DIR="$REPO_ROOT/mimic-video/model"
AD_REPO="push-that-thing/model"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN first." >&2
  exit 1
fi

# 1. Ensure a COMPLETE environment. Re-run setup if the venv is missing or
#    incomplete (the policy plugin can't import) — self-healing, so a half-built
#    venv never silently breaks the run.
if ! "$MODEL_DIR/.venv/bin/python" -c "import lerobot_policy_mimic_video" >/dev/null 2>&1; then
  echo "==> Environment missing or incomplete — running setup.sh"
  bash "$REPO_ROOT/setup.sh"
fi

# 2. Activate venv.
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/mimic_env.sh"
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"

# 3. Download this task's action decoder + statistics cache.
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
PROMPT_EMB_DIR="$HOME/.cache/mimic_lerobot_t5/${PROMPT_EMB_SUBDIR}/.t5_embeddings"

# 4. Deploy (dry-run unless LIVE=1).
DRY_RUN=1
[[ "${LIVE:-0}" == "1" ]] && DRY_RUN=0

echo ""
echo "==> Action decoder : $ACTION_CKPT"
echo "==> Video backbone : $VIDEO_CKPT_PATH"
echo "==> Statistics     : $STATS_PATH"
echo "==> T5 cache       : $PROMPT_EMB_DIR"
echo "==> Experiment     : $EXPERIMENT_NAME"
echo "==> Prompt         : $TASK_PROMPT"
[[ "$DRY_RUN" == "1" ]] && echo "==> DRY-RUN (no movement). Re-run with LIVE=1 to drive the arm." \
                        || echo "==> LIVE: the arm WILL move. Ensure it is calibrated and clear."
echo ""

export TASK_PROMPT ACTION_CKPT VIDEO_CKPT_PATH EXPERIMENT_NAME STATS_PATH PROMPT_EMB_DIR
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "==> SMOKE TEST (no arm): load pipeline + one inference"
  python "$REPO_ROOT/scripts/smoke_load.py"
else
  DRY_RUN="$DRY_RUN" bash "$REPO_ROOT/scripts/deploy_so101.sh"
fi
