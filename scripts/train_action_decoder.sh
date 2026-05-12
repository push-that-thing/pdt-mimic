#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMIC_DIR="${ROOT_DIR}/mimic-video"
MODEL_DIR="${MIMIC_DIR}/model"

WANDB_CONFIG="${WANDB_CONFIG:-${ROOT_DIR}/configs/wandb/action_decoder.env}"
if [[ -n "${WANDB_CONFIG}" ]]; then
  if [[ -f "${WANDB_CONFIG}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${WANDB_CONFIG}"
    set +a
  else
    echo "W&B config file not found: ${WANDB_CONFIG}" >&2
    exit 1
  fi
fi

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
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-1}"
MICRO_BSZ="${MICRO_BSZ:-${TRAIN_BSZ:-${BSZ}}}"
VAL_RATIO="${VAL_RATIO:-}"
VAL_EPISODES="${VAL_EPISODES:-}"
RUN_VALIDATION="${RUN_VALIDATION:-}"
VALIDATION_MODE="${VALIDATION_MODE:-}"
VALIDATION_ITER="${VALIDATION_ITER:-}"
MAX_VAL_ITER="${MAX_VAL_ITER:-}"
EARLY_STOPPING="${EARLY_STOPPING:-0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
EARLY_STOP_START_ITER="${EARLY_STOP_START_ITER:-0}"

EXPERIMENT="${EXPERIMENT:-w2a_${DATA_CONFIG}_${VIDEO_CKPT}_lr${LR}_layer${LAYER}_bsz${BSZ}}"
JOB_NAME="${JOB_NAME:-}"

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
  "trainer.grad_accum_iter=${GRAD_ACCUM_ITER}"
)

if [[ "${EARLY_STOPPING}" == "1" ]]; then
  if [[ "${USE_WANDB}" == "1" ]]; then
    args+=("callbacks=basic_wandb_earlystop")
  else
    args+=("callbacks=basic_earlystop")
  fi
elif [[ "${USE_WANDB}" == "1" ]]; then
  args+=("callbacks=basic_wandb")
fi
if [[ -n "${JOB_NAME}" ]]; then
  args+=("job.name=${JOB_NAME}")
fi
if [[ "${MICRO_BSZ}" != "${BSZ}" ]]; then
  args+=("dataloader_train.batch_size=${MICRO_BSZ}")
fi
if [[ -n "${VAL_RATIO}" ]]; then
  args+=("dataloader_train.dataset.val_ratio=${VAL_RATIO}")
  args+=("dataloader_val.dataset.val_ratio=${VAL_RATIO}")
fi
if [[ -n "${VAL_EPISODES}" ]]; then
  args+=("dataloader_train.dataset.num_val_episodes=${VAL_EPISODES}")
  args+=("dataloader_val.dataset.num_val_episodes=${VAL_EPISODES}")
fi
if [[ "${EARLY_STOPPING}" != "1" ]]; then
  if [[ -n "${RUN_VALIDATION}" ]]; then
    args+=("trainer.run_validation=${RUN_VALIDATION}")
  fi
  if [[ -n "${VALIDATION_MODE}" ]]; then
    args+=("model.config.validation_mode=${VALIDATION_MODE}")
  fi
  if [[ -n "${VALIDATION_ITER}" ]]; then
    args+=("trainer.validation_iter=${VALIDATION_ITER}")
  fi
  if [[ -n "${MAX_VAL_ITER}" ]]; then
    args+=("trainer.max_val_iter=${MAX_VAL_ITER}")
  fi
fi
if [[ "${EARLY_STOPPING}" == "1" ]]; then
  args+=("trainer.run_validation=True")
  args+=("model.config.validation_mode=${VALIDATION_MODE:-loss_only}")
  args+=("trainer.validation_iter=${VALIDATION_ITER:-500}")
  args+=("trainer.max_val_iter=${MAX_VAL_ITER:-64}")
  args+=("trainer.callbacks.early_stopping.patience=${EARLY_STOP_PATIENCE}")
  args+=("trainer.callbacks.early_stopping.min_delta=${EARLY_STOP_MIN_DELTA}")
  args+=("trainer.callbacks.early_stopping.start_iter=${EARLY_STOP_START_ITER}")
fi

if [[ "${CONFIG_DRYRUN:-0}" == "1" ]]; then
  args=(--nproc_per_node="${NGPU}" --master_port="${MASTER_PORT}" -m scripts.train --config=cosmos_predict2/configs/config.py --dryrun -- "experiment=${EXPERIMENT}")
  args+=("checkpoint.save_iter=${SAVE_ITER}")
  args+=("trainer.grad_accum_iter=${GRAD_ACCUM_ITER}")
  if [[ "${EARLY_STOPPING}" == "1" ]]; then
    if [[ "${USE_WANDB}" == "1" ]]; then
      args+=("callbacks=basic_wandb_earlystop")
    else
      args+=("callbacks=basic_earlystop")
    fi
  elif [[ "${USE_WANDB}" == "1" ]]; then
    args+=("callbacks=basic_wandb")
  fi
  if [[ -n "${JOB_NAME}" ]]; then
    args+=("job.name=${JOB_NAME}")
  fi
  if [[ "${MICRO_BSZ}" != "${BSZ}" ]]; then
    args+=("dataloader_train.batch_size=${MICRO_BSZ}")
  fi
  if [[ -n "${VAL_RATIO}" ]]; then
    args+=("dataloader_train.dataset.val_ratio=${VAL_RATIO}")
    args+=("dataloader_val.dataset.val_ratio=${VAL_RATIO}")
  fi
  if [[ -n "${VAL_EPISODES}" ]]; then
    args+=("dataloader_train.dataset.num_val_episodes=${VAL_EPISODES}")
    args+=("dataloader_val.dataset.num_val_episodes=${VAL_EPISODES}")
  fi
  if [[ "${EARLY_STOPPING}" != "1" ]]; then
    if [[ -n "${RUN_VALIDATION}" ]]; then
      args+=("trainer.run_validation=${RUN_VALIDATION}")
    fi
    if [[ -n "${VALIDATION_MODE}" ]]; then
      args+=("model.config.validation_mode=${VALIDATION_MODE}")
    fi
    if [[ -n "${VALIDATION_ITER}" ]]; then
      args+=("trainer.validation_iter=${VALIDATION_ITER}")
    fi
    if [[ -n "${MAX_VAL_ITER}" ]]; then
      args+=("trainer.max_val_iter=${MAX_VAL_ITER}")
    fi
  fi
  if [[ "${EARLY_STOPPING}" == "1" ]]; then
    args+=("trainer.run_validation=True")
    args+=("model.config.validation_mode=${VALIDATION_MODE:-loss_only}")
    args+=("trainer.validation_iter=${VALIDATION_ITER:-500}")
    args+=("trainer.max_val_iter=${MAX_VAL_ITER:-64}")
    args+=("trainer.callbacks.early_stopping.patience=${EARLY_STOP_PATIENCE}")
    args+=("trainer.callbacks.early_stopping.min_delta=${EARLY_STOP_MIN_DELTA}")
    args+=("trainer.callbacks.early_stopping.start_iter=${EARLY_STOP_START_ITER}")
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
args+=("$@")

echo "Launching action decoder training:"
echo "  experiment=${EXPERIMENT}"
echo "  ngpu=${NGPU}"
echo "  registered_bsz=${BSZ}"
echo "  micro_bsz=${MICRO_BSZ}"
echo "  grad_accum_iter=${GRAD_ACCUM_ITER}"
echo "  effective_bsz=$((MICRO_BSZ * NGPU * GRAD_ACCUM_ITER))"
echo "  save_iter=${SAVE_ITER}"
if [[ -n "${VAL_RATIO}" ]]; then
  echo "  val_ratio=${VAL_RATIO}"
fi
if [[ -n "${VAL_EPISODES}" ]]; then
  echo "  val_episodes=${VAL_EPISODES}"
fi
if [[ "${EARLY_STOPPING}" == "1" ]]; then
  echo "  validation_mode=${VALIDATION_MODE:-loss_only}"
  echo "  validation_iter=${VALIDATION_ITER:-500}"
  echo "  max_val_iter=${MAX_VAL_ITER:-64}"
  echo "  early_stopping=1 (patience=${EARLY_STOP_PATIENCE}, min_delta=${EARLY_STOP_MIN_DELTA}, start_iter=${EARLY_STOP_START_ITER})"
fi
echo "  wandb_config=${WANDB_CONFIG}"
if [[ -n "${JOB_NAME}" ]]; then
  echo "  job_name=${JOB_NAME}"
fi
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
