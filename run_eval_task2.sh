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
export TASK_PROMPT="${TASK_PROMPT:-Push the white object to the right into the goal white circle. Stay within the two white lines.}"
export ACTION_EXP="${ACTION_EXP:-w2a_so101_task2_combined_lora32_iter2500_75m_2gpu_bsz2_accum8_val5}"
export ACTION_ITER="${ACTION_ITER:-2500}"
export STATS_HASH="${STATS_HASH:-eeaabacd800b60e08ef93a5398f5c3e2e193c6fef2f7af04d47bca467bc69f85}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1}"
export PROMPT_EMB_SUBDIR="${PROMPT_EMB_SUBDIR:-task2}"
export VIDEO_CKPT_NAME="${VIDEO_CKPT_NAME:-v2w_push_combined_lora_rank32_iter2500_fused}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_run_eval_common.sh"
