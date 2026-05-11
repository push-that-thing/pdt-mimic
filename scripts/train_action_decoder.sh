#!/usr/bin/env bash
# Launch action decoder training on the current machine.
#
# Tunable via env vars:
#   BSZ    global batch size (must be in [1, 128, 256] and divisible by NGPU)
#   NGPU   number of GPUs  (default: all available)
#   LR     learning rate   (default: 1.000e-04)
#   LAYER  cross-attention layer index (default: 20)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMIC="$REPO_ROOT/mimic-video"

: "${NGPU:=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
: "${LR:=1.000e-04}"
: "${LAYER:=20}"
: "${VIDEO_CKPT:=v2w_push_that_thing}"
: "${DATA_CFG:=lerobot}"

# Pick global batch size: 128 for multi-GPU, 1 for single-GPU
if [ "${BSZ:-}" = "" ]; then
    if [ "$NGPU" -ge 2 ]; then
        BSZ=128
    else
        BSZ=1
    fi
fi

EXP="w2a_${DATA_CFG}_${VIDEO_CKPT}_lr${LR}_layer${LAYER}_bsz${BSZ}"

echo "Experiment : $EXP"
echo "GPUs       : $NGPU"
echo "Global BSZ : $BSZ  ($(( BSZ / NGPU )) per GPU)"

cd "$MIMIC/model"
source .venv/bin/activate

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
NVTE_FUSED_ATTN=0 \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/configs/config.py \
    -- experiment="$EXP"
