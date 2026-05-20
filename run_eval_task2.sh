#!/usr/bin/env bash
# Set up the environment and run the SO-101 "push-that-thing" task-2 policy.
#
# Quickstart:
#   export HF_TOKEN=hf_...
#   bash run_eval_task2.sh                 # dry-run (prints the deploy command)
#   LIVE=1 bash run_eval_task2.sh          # actually drive the arm
#
# TODO: confirm TASK_PROMPT matches the exact instruction string stored in
# push-that-thing/task_2's .t5_embeddings cache. With USE_TEXT_ENCODER=false
# (the default) the prompt is looked up by content hash; a mismatch will fail
# at load. If unsure, run with USE_TEXT_ENCODER=true (needs a large GPU for T5).
export TASK_PROMPT="${TASK_PROMPT:-Push the white object to the right into the goal white circle. Stay within the two white lines.}"

# Action decoder trained on task-2 data (combined task1+task2 LoRA backbone).
export ACTION_EXP="${ACTION_EXP:-w2a_so101_task2_combined_lora32_iter2500_75m_2gpu_bsz2_accum8_val5}"
export ACTION_ITER="${ACTION_ITER:-2500}"
export STATS_HASH="${STATS_HASH:-eeaabacd800b60e08ef93a5398f5c3e2e193c6fef2f7af04d47bca467bc69f85}"

# mimic-video hydra experiment the decoder was trained under (defines the so101
# net + policy_io). Override if your training used a different registered name.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1}"

# Shared combined task1+task2 video backbone + the T5 cache to warm.
export HF_DATASET_REPO="${HF_DATASET_REPO:-push-that-thing/task_2}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-push-that-thing/cosmos-task1-task2}"
export VIDEO_CKPT_FILE="${VIDEO_CKPT_FILE:-posttraining/video2world/v2w_push_combined_lora_rank32_lr1.778e-04_bsz32/checkpoints/model/iter_000002500_fused.pt}"
export VIDEO_CKPT_NAME="${VIDEO_CKPT_NAME:-v2w_push_combined_lora_rank32_iter2500_fused}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_run_eval_common.sh"
