#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMIC_DIR="${ROOT_DIR}/mimic-video"
MODEL_DIR="${MIMIC_DIR}/model"

VIDEO_CKPT="${VIDEO_CKPT:-v2w_push_that_thing}"
DATA_CONFIG="${DATA_CONFIG:-so101_lerobot}"
LR="${LR:-1.000e-04}"
LAYER="${LAYER:-20}"

if [[ -z "${NGPU:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
  else
    NGPU="1"
  fi
fi

if [[ -z "${BSZ:-}" ]]; then
  if [[ "${NGPU}" -gt 1 ]]; then
    BSZ="128"
  else
    BSZ="1"
  fi
fi

MASTER_PORT="${MASTER_PORT:-12341}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-vam}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
USE_WANDB="${USE_WANDB:-1}"
SAVE_ITER="${SAVE_ITER:-1000}"

EXPERIMENT="${EXPERIMENT:-w2a_${DATA_CONFIG}_${VIDEO_CKPT}_lr${LR}_layer${LAYER}_bsz${BSZ}}"

export PYTHONPATH="${MODEL_DIR}:${ROOT_DIR}/lerobot/src:${PYTHONPATH:-}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
CUDA_LIB_PATHS=()
for dir in "${MODEL_DIR}"/.venv/lib/python*/site-packages/nvidia/*/lib; do
  [[ -d "${dir}" ]] && CUDA_LIB_PATHS+=("${dir}")
done
if [[ "${#CUDA_LIB_PATHS[@]}" -gt 0 ]]; then
  CUDA_LIB_PATHS_JOINED="$(IFS=:; echo "${CUDA_LIB_PATHS[*]}")"
  export LD_LIBRARY_PATH="${CUDA_LIB_PATHS_JOINED}:${LD_LIBRARY_PATH:-}"
fi
export WANDB_MODE
export WANDB_PROJECT
if [[ -n "${WANDB_ENTITY}" ]]; then
  export WANDB_ENTITY
fi

args=(
  --nproc_per_node="${NGPU}"
  --master_port="${MASTER_PORT}"
  -m scripts.train
  --config=cosmos_predict2/configs/config.py
  --
  "experiment=${EXPERIMENT}"
  "checkpoint.save_iter=${SAVE_ITER}"
)

if [[ "${USE_WANDB}" == "1" ]]; then
  args+=("callbacks=basic_wandb")
fi

if [[ "${CONFIG_DRYRUN:-0}" == "1" ]]; then
  args=(--nproc_per_node="${NGPU}" --master_port="${MASTER_PORT}" -m scripts.train --config=cosmos_predict2/configs/config.py --dryrun -- "experiment=${EXPERIMENT}")
  args+=("checkpoint.save_iter=${SAVE_ITER}")
  if [[ "${USE_WANDB}" == "1" ]]; then
    args+=("callbacks=basic_wandb")
  fi
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  args+=(
    "trainer.max_iter=${SMOKE_MAX_ITER:-2}"
    "trainer.logging_iter=1"
    "trainer.run_validation=False"
    "checkpoint.save_iter=${SMOKE_SAVE_ITER:-99999999}"
  )
fi

echo "Launching action decoder training:"
echo "  experiment=${EXPERIMENT}"
echo "  ngpu=${NGPU}"
echo "  bsz=${BSZ}"
echo "  save_iter=${SAVE_ITER}"
echo "  wandb=${USE_WANDB} (${WANDB_MODE})"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf "Command:"
  printf " %q" torchrun "${args[@]}"
  printf "\n"
  exit 0
fi

cd "${MODEL_DIR}"
source .venv/bin/activate

torchrun "${args[@]}"
