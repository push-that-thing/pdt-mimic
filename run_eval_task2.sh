#!/usr/bin/env bash
# Deploy the SO-101 push-that-thing TASK 2 policy.
#
#   export HF_TOKEN=hf_...
#   bash run_eval_task2.sh                       # dry-run (prints the command)
#   LIVE=1 ROBOT_PORT=/dev/ttyACM0 bash run_eval_task2.sh   # drive the arm
#
# TODO: set TASK_PROMPT to task-2's exact instruction (it is hash-matched against
# the T5 cache when use_text_encoder=false). With USE_TEXT_ENCODER=true the prompt
# is encoded live instead (needs a large GPU for T5).
export TASK_PROMPT="${TASK_PROMPT:-Push the white object to the right into the goal white circle. Don’t let the white object and the obstacle in the middle touch.}"
export ACTION_EXP="${ACTION_EXP:-w2a_so101_task2_slow_136ep_lora32_iter2500_75m_2gpu}"
export ACTION_ITER="${ACTION_ITER:-1700}"
export STATS_HASH="${STATS_HASH:-0f8456e70aac1020c51972f716c06e6b1f4e13c2ce71bf8348aaeb5d58c8157e}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1}"
export PROMPT_EMB_SUBDIR="${PROMPT_EMB_SUBDIR:-task2}"
export VIDEO_CKPT_NAME="${VIDEO_CKPT_NAME:-v2w_push_combined_lora_rank32_iter2500_fused}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_run_eval_common.sh"
