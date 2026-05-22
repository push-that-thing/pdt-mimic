#!/usr/bin/env bash
# Deploy the SO-101 push-that-thing TASK 2 policy — 125M merged action decoder.
#
# This is the larger 125M decoder (w2a_so101_125m_*, model_channels=640,
# num_blocks=13, trained on 8xH100 over merged task-2 data), kept separate from
# the 75M decoder in run_eval_task2_75m.sh so the two can be evaluated independently.
# It uses the 640-dim `so101_125m` net via the w2a_so101_125m_* experiment.
#
#   export HF_TOKEN=hf_...
#   bash run_eval_task2_125m.sh                       # dry-run (prints the command)
#   LIVE=1 ROBOT_PORT=/dev/ttyACM0 bash run_eval_task2_125m.sh   # drive the arm
export TASK_PROMPT="${TASK_PROMPT:-Push the white object to the right into the goal white circle. Don’t let the white object and the obstacle in the middle touch.}"
export ACTION_EXP="${ACTION_EXP:-w2a_so101_125m_8xh100_task2merged_bsz64}"
export ACTION_ITER="${ACTION_ITER:-2000}"
# Stats for push-that-thing/task_2_merged, action horizon 90 (train split,
# val_ratio=0.05) — the split the trainer used to build the normalizer. Lives
# in the task2merged-specific stats subdir, not the shared stats_cache.
export STATS_HASH="${STATS_HASH:-ad93e69310a789d29820a51190858c3205be9a94e5d063949ee8fd071ce0bc78}"
export STATS_REMOTE_SUBDIR="${STATS_REMOTE_SUBDIR:-stats_cache_task2merged}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-w2a_so101_125m_v2w_push_that_thing_lr1.000e-04_layer20_bsz1}"
export PROMPT_EMB_SUBDIR="${PROMPT_EMB_SUBDIR:-task2}"
export VIDEO_CKPT_NAME="${VIDEO_CKPT_NAME:-v2w_push_combined_lora_rank32_iter2500_fused}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_run_eval_common.sh"
