#!/usr/bin/env bash
# Deploy the SO-101 push-that-thing TASK 1 policy.
#
#   export HF_TOKEN=hf_...
#   bash run_eval_task1.sh                       # dry-run (prints the command)
#   LIVE=1 ROBOT_PORT=/dev/ttyACM0 bash run_eval_task1.sh   # drive the arm

export TASK_PROMPT="${TASK_PROMPT:-Push the white object to the right into the goal white circle. Stay within the two white lines.}"
export ACTION_EXP="${ACTION_EXP:-w2a_so101_combined_lora32_iter2500_75m_2gpu_bsz2_accum8_val5}"
export ACTION_ITER="${ACTION_ITER:-3400}"
export STATS_HASH="${STATS_HASH:-77c34c080268301503540af7a2f650baef0326ae0c3401fb4e786bfdb0f5ebc6}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1}"
export PROMPT_EMB_SUBDIR="${PROMPT_EMB_SUBDIR:-task1}"
export VIDEO_CKPT_NAME="${VIDEO_CKPT_NAME:-v2w_push_combined_lora_rank32_iter2500_fused}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_run_eval_common.sh"
