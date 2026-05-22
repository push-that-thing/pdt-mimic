#!/usr/bin/env bash
# Deploy the SO-101 push-that-thing TASK 3.1 policy — 75M action decoder.
#
#   export HF_TOKEN=hf_...
#   bash run_eval_task3_1.sh                       # dry-run (prints the command)
#   LIVE=1 ROBOT_PORT=/dev/ttyACM0 bash run_eval_task3_1.sh   # drive the arm
#
# Uses the old 75M config (512-dim `so101` net, same EXPERIMENT_NAME as task1).
# TASK_PROMPT must match push-that-thing/task_3_1's instruction verbatim — it is
# hash-matched against the T5 cache when use_text_encoder=false.
export TASK_PROMPT="${TASK_PROMPT:-Push the cube into the goal circle}"
export ACTION_EXP="${ACTION_EXP:-w2a_so101_task3_1_103ep_lora32_iter2500_75m_2gpu}"
export ACTION_ITER="${ACTION_ITER:-2500}"
export STATS_HASH="${STATS_HASH:-6a47b95b1df0498744e31779df3e63d887a0136892c0f801b5346aeb498fa254}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1}"
export PROMPT_EMB_SUBDIR="${PROMPT_EMB_SUBDIR:-task3_1}"
export VIDEO_CKPT_NAME="${VIDEO_CKPT_NAME:-v2w_push_combined_lora_rank32_iter2500_fused}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_run_eval_common.sh"
