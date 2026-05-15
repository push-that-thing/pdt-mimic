#!/usr/bin/env bash
# Shared environment for mimic-video deployment on this box.
# Source from other scripts: `source "$(dirname "$0")/mimic_env.sh"`

set -eo pipefail

PDT_DIR="${PDT_DIR:-/home/rl_10/rl_project_world/deploy_mimic/pdt-mimic}"
MIMIC_DIR="$PDT_DIR/mimic-video"
MODEL_DIR="$MIMIC_DIR/model"
VENV="$MODEL_DIR/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: venv missing at $VENV. Re-run uv sync --extra cu126 in $MODEL_DIR." >&2
  exit 1
fi

# Bundled CUDA libs (avoids needing sudo ldconfig — required for transformer_engine + cosmos).
export CUDA_HOME="$VENV/lib/python3.10/site-packages/nvidia"
NVIDIA_LIBS="$(find "$CUDA_HOME" -maxdepth 2 -name lib -type d | tr '\n' ':' | sed 's/:$//')"
export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Cosmos / imaginaire are loaded from source, not installed.
export PYTHONPATH="$MODEL_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Keep calibration + dataset lookup pointed at the project root (matches comands.txt).
export HF_HOME="${HF_HOME:-/home/rl_10/rl_project_world}"

# HF auth — token file from the project root.
if [[ -z "${HF_TOKEN:-}" && -r "/home/rl_10/rl_project_world/token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < /home/rl_10/rl_project_world/token)"
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Activate the venv so `python`/`lerobot-record` resolve to the right binaries.
# shellcheck disable=SC1091
source "$VENV/bin/activate"
