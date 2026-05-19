#!/usr/bin/env bash
# Deploy the MIMIC video SO-101 policy on the actual robot.
#
# Defaults match the original baseline (so101_follower on /dev/ttyACM0, front
# camera at index 0, 640x480@30, no teleop, no behavior modifications).
#
# Usage:
#   bash scripts/deploy_so101.sh                        # baseline rollout
#   EPISODE_TIME_S=180 bash scripts/deploy_so101.sh     # 3-minute episode
#   NUM_EPISODES=5 EPISODE_TIME_S=20 bash scripts/deploy_so101.sh
#   ROBOT_PORT=/dev/ttyACM1 bash scripts/deploy_so101.sh
#   DRY_RUN=1 bash scripts/deploy_so101.sh              # print the command only
#
# Episode / dataset flags:
#   EPISODE_TIME_S=15          seconds per episode
#   NUM_EPISODES=1             how many episodes to record
#   FPS=30                     control + camera rate
#
# Ablation flags — each defaults to baseline behavior:
#   STOP_AFTER_STEP=15         diffusion steps (model default 15; lower = faster, coarser)
#   N_ACTION_STEPS=10          how many of the 90 planned targets to execute per chunk (lower = less motion per chunk)
#   MAX_RELATIVE_TARGET=       per-tick motor delta cap (empty = no cap; try 10–20)
#   ACTION_REPEAT=1            hold each policy action for N ticks (1 = no repeat)
#   TORQUE_OFF_ON_EXIT=1       0 = leave torque on after abnormal exit (default ON releases torque on Ctrl+C/crash)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mimic_env.sh
source "$SCRIPT_DIR/mimic_env.sh"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-my_awesome_follower_arm}"

# Ablation: release torque on abnormal exit (Ctrl+C during CUDA call usually
# skips lerobot's disconnect path). Off by default — original behavior was to
# leave torque on until power-cycled.
if [[ "${TORQUE_OFF_ON_EXIT:-1}" == "1" ]]; then
  release_torque_on_abnormal_exit() {
    local rc=$?
    if [[ "$rc" -ne 0 ]]; then
      echo >&2
      echo "deploy_so101: releasing torque on $ROBOT_PORT (exit=$rc)..." >&2
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
TASK_PROMPT="${TASK_PROMPT:-Push the white object to the right into the goal white circle. Stay within the two white lines.}"

# Output dataset (lerobot-record requires a dataset.repo_id even for inference rollouts).
DATASET_REPO="${DATASET_REPO:-local/eval_mimic_so101}"
DATASET_ROOT="${DATASET_ROOT:-$HF_HOME/eval_rollouts/$(date +%Y%m%d-%H%M%S)}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-15}"
FPS="${FPS:-30}"
# Rerun viewer needs a display. Default off so SSH/headless runs don't crash.
DISPLAY_DATA="${DISPLAY_DATA:-false}"
# Diffusion early-stop. Model default 15. Lower = faster compute, coarser.
STOP_AFTER_STEP="${STOP_AFTER_STEP:-15}"
# Ablation: how many of the 90 planned targets to enqueue per chunk. Lower =
# less total motion per chunk (idle gap unchanged). Default 10 matches config.
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
# Ablation: per-tick position delta cap per motor (normalized motor units,
# ~[-100, 100] full range). Empty = no cap (baseline). Try 10–20 to soften.
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-}"
# Ablation: hold each policy action for N control ticks → 1/N speed playback.
# 1 = original behavior. Read by lerobot_policy_mimic_video via env.
ACTION_REPEAT="${ACTION_REPEAT:-1}"
export MIMIC_ACTION_REPEAT="$ACTION_REPEAT"

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
  --policy.stop_after_step="$STOP_AFTER_STEP"
  --policy.n_action_steps="$N_ACTION_STEPS"
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
LOG_FILE="${LOG_FILE:-$LOG_DIR/deploy_so101_$(date +%Y%m%d-%H%M%S).log}"

echo "+ ${CMD[*]}"
echo "log: $LOG_FILE"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

{
  echo "=== deploy_so101 $(date -Iseconds) ==="
  echo "host: $(hostname)  user: $(whoami)  cwd: $(pwd)"
  echo "branch: $(git -C "$SCRIPT_DIR/.." rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD 2>/dev/null)"
  echo "python: $(command -v python) ($(python --version 2>&1))"
  echo "CMD: ${CMD[*]}"
  echo "==="
} >>"$LOG_FILE"

# 2>&1 + tee captures both streams. PIPESTATUS preserves the python exit code.
# stdin from /dev/null + a piped newline auto-accepts the SO-101 calibrate-confirm
# prompt ("Press ENTER to use provided calibration file, or 'c' to recalibrate")
# that fires whenever motor flash and the saved JSON don't match exactly.
set +e
echo "" | "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
ec=${PIPESTATUS[1]}
set -e

echo
if [[ "$ec" -ne 0 ]]; then
  echo "deploy_so101 exited $ec — full log: $LOG_FILE" >&2
  echo "last 40 lines:" >&2
  tail -40 "$LOG_FILE" >&2
fi
exit "$ec"
