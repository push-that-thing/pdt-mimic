#!/usr/bin/env bash
# Deploy task2 of the MIMIC video SO-101 policy (combined-LoRA variant).
# Same shape as deploy_so101.sh (task1) / deploy_so101_task2_slow.sh:
#   - action checkpoint: push-that-thing/model
#       :: action_decoder/chain/w2a_so101_task2_combined_lora32_iter2500_75m_2gpu_bsz2_accum8_val5/checkpoints/model/iter_*.pt
#   - stats cache:       push-that-thing/model
#       :: action_decoder/stats_cache/eeaabacd...  (pinned explicitly via
#          --policy.dataset_statistics_path so newest-mtime doesn't shadow it)
#   - T5 embedding:      push-that-thing/task_2 :: .t5_embeddings/*.npz
#   - prompt:            auto-detected from the downloaded T5 embedding (.npz key)
#
# Usage:
#   bash scripts/deploy_so101_task2.sh                    # baseline rollout
#   EPISODE_TIME_S=180 bash scripts/deploy_so101_task2.sh # 3-minute episode
#   NUM_EPISODES=5 EPISODE_TIME_S=20 bash scripts/deploy_so101_task2.sh
#   ROBOT_PORT=/dev/ttyACM1 bash scripts/deploy_so101_task2.sh
#   DRY_RUN=1 bash scripts/deploy_so101_task2.sh
#
# Episode / dataset flags:
#   EPISODE_TIME_S=15          seconds per episode
#   NUM_EPISODES=1             how many episodes to record
#   FPS=30                     control + camera rate
#
# Ablation flags (defaults match task1):
#   NUM_SAMPLING_STEP=35       total diffusion schedule length (model default 35)
#   STOP_AFTER_STEP=15         early-stop after N steps (model default 15)
#   N_ACTION_STEPS=10          targets executed per chunk (lower = less motion per chunk)
#   MAX_RELATIVE_TARGET=       per-tick motor delta cap (try 10-20)
#   ACTION_REPEAT=1            hold each policy action for N ticks
#   TORQUE_OFF_ON_EXIT=1       0 = leave torque on (default ON releases torque on Ctrl+C/crash)
#
# Override task2 specifics:
#   TASK2_ITER=000003100       which task2 checkpoint iter to use (1900, 3100, etc.)
#   TASK_PROMPT="..."          force a prompt (default = auto-detect from .npz)
#   TASK2_STATS_HASH=...       override stats file hash (default eeaabacd...)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mimic_env.sh
source "$SCRIPT_DIR/mimic_env.sh"

# --- task2 asset coordinates ---
TASK2_MODEL_REPO="push-that-thing/model"
TASK2_DATASET_REPO="push-that-thing/task_2"
TASK2_ITER="${TASK2_ITER:-000003100}"
TASK2_RUN="w2a_so101_task2_combined_lora32_iter2500_75m_2gpu_bsz2_accum8_val5"
TASK2_CKPT_REL="action_decoder/chain/${TASK2_RUN}/checkpoints/model/iter_${TASK2_ITER}.pt"
TASK2_STATS_HASH="${TASK2_STATS_HASH:-eeaabacd800b60e08ef93a5398f5c3e2e193c6fef2f7af04d47bca467bc69f85}"
TASK2_STATS_REL="action_decoder/stats_cache/${TASK2_STATS_HASH}"

# --- local target paths (mirror task1 layout) ---
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK2_CKPT_LOCAL="$REPO_ROOT/mimic-video/model/checkpoints/vam/so101/${TASK2_RUN}/checkpoints/model/iter_${TASK2_ITER}.pt"
STATS_CACHE_DIR="${MIMIC_STATS_CACHE_DIR:-$HOME/.cache/mimic_lerobot_stats}"
T5_CACHE_DIR="${MIMIC_T5_CACHE_DIR:-$HOME/.cache/mimic_lerobot_t5/.t5_embeddings}"

mkdir -p "$(dirname "$TASK2_CKPT_LOCAL")" "$STATS_CACHE_DIR" "$T5_CACHE_DIR"

hf_download() {
  # hf_download <repo_id> <relative path> <repo_type> -> stdout = local path
  # `hf download` prefixes its output with "path=", strip it.
  local repo="$1" rel="$2" rt="$3"
  hf download "$repo" "$rel" --repo-type "$rt" 2>/dev/null | sed -e 's/^path=//'
}

# 1) Action checkpoint
if [[ ! -f "$TASK2_CKPT_LOCAL" ]]; then
  echo "[task2] downloading action checkpoint iter_${TASK2_ITER}.pt …"
  src=$(hf_download "$TASK2_MODEL_REPO" "$TASK2_CKPT_REL" model)
  cp -f "$src" "$TASK2_CKPT_LOCAL"
fi

# 2) Stats cache file — pinned explicitly, not by mtime.
TASK2_STATS_LOCAL="$STATS_CACHE_DIR/$TASK2_STATS_HASH"
if [[ ! -f "$TASK2_STATS_LOCAL" ]]; then
  echo "[task2] downloading stats cache $TASK2_STATS_HASH …"
  src=$(hf_download "$TASK2_MODEL_REPO" "$TASK2_STATS_REL" model)
  cp -f "$src" "$TASK2_STATS_LOCAL"
fi

# 3) T5 embeddings — download the .t5_embeddings directory from the task_2
#    dataset and copy any new .npz files into the policy's cache dir.
TASK2_T5_HASH="5c53708f764ce3095860f6b20d62b352473c79d86d6e78cbd556789b4230c0f9"
if [[ ! -f "$T5_CACHE_DIR/${TASK2_T5_HASH}.npz" ]]; then
  echo "[task2] downloading T5 embeddings from $TASK2_DATASET_REPO/.t5_embeddings …"
  snapshot=$(hf download "$TASK2_DATASET_REPO" --repo-type dataset --include ".t5_embeddings/*" 2>/dev/null | sed -e 's/^path=//')
  if [[ -d "$snapshot/.t5_embeddings" ]]; then
    cp -fn "$snapshot/.t5_embeddings/"*.npz "$T5_CACHE_DIR/" 2>/dev/null || true
  fi
fi

# 4) Resolve task2 prompt. The .npz uses the prompt string as its key, so we
#    read it back to keep the script self-contained (override via TASK_PROMPT).
if [[ -z "${TASK_PROMPT:-}" ]]; then
  TASK_PROMPT=$(MIMIC_T5_NPZ="$T5_CACHE_DIR/${TASK2_T5_HASH}.npz" python - <<'PYEOF'
import os, numpy as np
with np.load(os.environ["MIMIC_T5_NPZ"], allow_pickle=False) as d:
    print(d.files[0])
PYEOF
  )
  if [[ -z "$TASK_PROMPT" ]]; then
    echo "ERROR: could not read task2 prompt from $T5_CACHE_DIR/${TASK2_T5_HASH}.npz. Set TASK_PROMPT=\"...\" explicitly." >&2
    exit 1
  fi
  echo "[task2] prompt: $TASK_PROMPT"
fi

# --- baseline robot/dataset config (matches deploy_so101.sh) ---
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-my_awesome_follower_arm}"

if [[ "${TORQUE_OFF_ON_EXIT:-1}" == "1" ]]; then
  release_torque_on_abnormal_exit() {
    local rc=$?
    if [[ "$rc" -ne 0 ]]; then
      echo >&2
      echo "deploy_so101_task2: releasing torque on $ROBOT_PORT (exit=$rc)..." >&2
      python - "$ROBOT_PORT" "$ROBOT_ID" <<'PYEOF' >&2 || true
import sys
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
port, robot_id = sys.argv[1], sys.argv[2]
r = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
r.connect(calibrate=False)
r.disconnect()
print("torque released.")
PYEOF
    fi
  }
  trap release_torque_on_abnormal_exit EXIT
fi

CAMERA_IDX="${CAMERA_IDX:-0}"
DATASET_REPO="${DATASET_REPO:-local/eval_mimic_so101_task2}"
DATASET_ROOT="${DATASET_ROOT:-$HF_HOME/eval_rollouts_task2/$(date +%Y%m%d-%H%M%S)}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-15}"
FPS="${FPS:-30}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
NUM_SAMPLING_STEP="${NUM_SAMPLING_STEP:-35}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:-15}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-}"
ACTION_REPEAT="${ACTION_REPEAT:-1}"
export MIMIC_ACTION_REPEAT="$ACTION_REPEAT"

# Make checkpoint path relative to repo root (config resolves both ways).
TASK2_CKPT_REPO_REL="mimic-video/model/checkpoints/vam/so101/${TASK2_RUN}/checkpoints/model/iter_${TASK2_ITER}.pt"

CMD=(
  python "$SCRIPT_DIR/lerobot_record_mimic.py"
  --robot.type=so101_follower
  --robot.port="$ROBOT_PORT"
  --robot.id="$ROBOT_ID"
)
if [[ -n "$MAX_RELATIVE_TARGET" ]]; then
  CMD+=(--robot.max_relative_target="$MAX_RELATIVE_TARGET")
fi
CMD+=(
  --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_IDX, width: 640, height: 480, fps: $FPS}}"
  --display_data="$DISPLAY_DATA"
  --policy.type=mimic_video
  --policy.device=cuda
  --policy.use_text_encoder=false
  --policy.num_sampling_step="$NUM_SAMPLING_STEP"
  --policy.stop_after_step="$STOP_AFTER_STEP"
  --policy.n_action_steps="$N_ACTION_STEPS"
  --policy.action_checkpoint_path="$TASK2_CKPT_REPO_REL"
  --policy.dataset_statistics_path="$TASK2_STATS_LOCAL"
  --policy.task="$TASK_PROMPT"
  --dataset.repo_id="$DATASET_REPO"
  --dataset.root="$DATASET_ROOT"
  --dataset.push_to_hub=false
  --dataset.num_episodes="$NUM_EPISODES"
  --dataset.episode_time_s="$EPISODE_TIME_S"
  --dataset.single_task="$TASK_PROMPT"
  --dataset.fps="$FPS"
)

LOG_DIR="${LOG_DIR:-/tmp/pdt-mimic-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/deploy_so101_task2_$(date +%Y%m%d-%H%M%S).log}"

echo "+ ${CMD[*]}"
echo "log: $LOG_FILE"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

{
  echo "=== deploy_so101_task2 $(date -Iseconds) ==="
  echo "host: $(hostname)  user: $(whoami)  cwd: $(pwd)"
  echo "branch: $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null)"
  echo "python: $(command -v python) ($(python --version 2>&1))"
  echo "task2 ckpt:  $TASK2_CKPT_LOCAL"
  echo "task2 stats: $TASK2_STATS_LOCAL"
  echo "task2 prompt: $TASK_PROMPT"
  echo "CMD: ${CMD[*]}"
  echo "==="
} >>"$LOG_FILE"

set +e
echo "" | "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
ec=${PIPESTATUS[1]}
set -e

echo
if [[ "$ec" -ne 0 ]]; then
  echo "deploy_so101_task2 exited $ec — full log: $LOG_FILE" >&2
  echo "last 40 lines:" >&2
  tail -40 "$LOG_FILE" >&2
fi
exit "$ec"
