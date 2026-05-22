#!/usr/bin/env bash
# Deploy the SO-101 push-that-thing TASK 2 policy — 75M "combined" decoder ("fast").
#
#   export HF_TOKEN=hf_...
#   bash run_eval_task2_fast.sh                       # dry-run (prints the command)
#   LIVE=1 ROBOT_PORT=/dev/ttyACM0 bash run_eval_task2_fast.sh   # drive the arm
#
# Another 75M LoRA decoder (512-dim `so101` net, same EXPERIMENT_NAME as the
# other 75M scripts), trained on push-that-thing/task_2 with the combined video
# backbone. STATS_HASH is task_2's horizon-90 stats (eeaabacd) — verified against
# the dataset's action distribution, NOT task1's 77c34c08.
export TASK_PROMPT="${TASK_PROMPT:-Push the white object to the right into the goal white circle. Don’t let the white object and the obstacle in the middle touch.}"
export ACTION_EXP="${ACTION_EXP:-w2a_so101_task2_combined_lora32_iter2500_75m_2gpu_bsz2_accum8_val5}"
export ACTION_ITER="${ACTION_ITER:-3100}"
export STATS_HASH="${STATS_HASH:-eeaabacd800b60e08ef93a5398f5c3e2e193c6fef2f7af04d47bca467bc69f85}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1}"
export PROMPT_EMB_SUBDIR="${PROMPT_EMB_SUBDIR:-task2}"
export VIDEO_CKPT_NAME="${VIDEO_CKPT_NAME:-v2w_push_combined_lora_rank32_iter2500_fused}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_run_eval_common.sh"
