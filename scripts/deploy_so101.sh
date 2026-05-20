#!/usr/bin/env bash
# Deploy the MIMIC video SO-101 policy on the actual robot.
#
# Defaults match comands.txt (so101_follower on /dev/ttyACM0, front camera at
# index 0, 640x480@30, no teleop). Every value is env-overridable.
#
# Usage:
#   bash scripts/deploy_so101.sh                        # 1 short rollout, no recording
#   ROBOT_PORT=/dev/ttyACM1 bash scripts/deploy_so101.sh
#   NUM_EPISODES=5 EPISODE_TIME_S=20 bash scripts/deploy_so101.sh
#   DRY_RUN=1 bash scripts/deploy_so101.sh              # print the command only

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mimic_env.sh
source "$SCRIPT_DIR/mimic_env.sh"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-my_awesome_follower_arm}"
CAMERA_IDX="${CAMERA_IDX:-0}"
TASK_PROMPT="${TASK_PROMPT:-Push the white object to the right into the goal white circle. Stay within the two white lines.}"

# Output dataset (lerobot-record requires a dataset.repo_id even for inference rollouts).
# Name must START with "eval_" when a policy is attached (lerobot sanity check).
DATASET_REPO="${DATASET_REPO:-local/eval_mimic_so101}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/eval_rollouts}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-15}"
FPS="${FPS:-30}"

CMD=(
  python "$SCRIPT_DIR/lerobot_record_mimic.py"
  --robot.type=so101_follower
  --robot.port="$ROBOT_PORT"
  --robot.id="$ROBOT_ID"
  --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_IDX, width: 640, height: 480, fps: $FPS}}"
  --display_data=true
  --policy.type=mimic_video
  --policy.device=cuda
  --policy.use_text_encoder="${USE_TEXT_ENCODER:-false}"
  --policy.task="$TASK_PROMPT"
  --dataset.repo_id="$DATASET_REPO"
  --dataset.root="$DATASET_ROOT"
  --dataset.push_to_hub=false
  --dataset.num_episodes="$NUM_EPISODES"
  --dataset.episode_time_s="$EPISODE_TIME_S"
  --dataset.single_task="$TASK_PROMPT"
  --dataset.fps="$FPS"
)

# Optional per-task overrides (set by run_eval_taskN.sh). Appended only when present.
[[ -n "${ACTION_CKPT:-}" ]]     && CMD+=(--policy.action_checkpoint_path="$ACTION_CKPT")
[[ -n "${VIDEO_CKPT_PATH:-}" ]] && CMD+=(--policy.video_checkpoint_path="$VIDEO_CKPT_PATH")
[[ -n "${EXPERIMENT_NAME:-}" ]] && CMD+=(--policy.experiment_name="$EXPERIMENT_NAME")
[[ -n "${STATS_PATH:-}" ]]      && CMD+=(--policy.dataset_statistics_path="$STATS_PATH")
[[ -n "${PROMPT_EMB_DIR:-}" ]]  && CMD+=(--policy.prompt_embedding_cache_dir="$PROMPT_EMB_DIR")

# Inference knobs (optional). num_sampling_step/stop_after_step = video backbone;
# action_num_denoising_steps = action decoder; n_action_steps = actions per replan.
[[ -n "${NUM_SAMPLING_STEP:-}" ]]    && CMD+=(--policy.num_sampling_step="$NUM_SAMPLING_STEP")
[[ -n "${STOP_AFTER_STEP:-}" ]]      && CMD+=(--policy.stop_after_step="$STOP_AFTER_STEP")
[[ -n "${ACTION_DENOISE_STEPS:-}" ]] && CMD+=(--policy.action_num_denoising_steps="$ACTION_DENOISE_STEPS")
[[ -n "${N_ACTION_STEPS:-}" ]]       && CMD+=(--policy.n_action_steps="$N_ACTION_STEPS")
[[ -n "${SEED:-}" ]]                 && CMD+=(--policy.seed="$SEED")
[[ -n "${USE_CUDA_GRAPHS:-}" ]]      && CMD+=(--policy.use_cuda_graphs="$USE_CUDA_GRAPHS")

echo "+ ${CMD[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

# Run under the crash-capture logger so a Python traceback, CUDA segfault, or a
# dropped terminal still leaves the full error in $LOG_DIR (default ~/eval_logs).
export LOG_TAG="${LOG_TAG:-deploy_so101}"
exec "$SCRIPT_DIR/run_with_log.sh" "${CMD[@]}"
