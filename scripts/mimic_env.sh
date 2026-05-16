#!/usr/bin/env bash
# Shared environment for mimic-video deployment on this box.
# Source from other scripts: `source "$(dirname "$0")/mimic_env.sh"`

set -eo pipefail

: "${PDT_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MIMIC_DIR="$PDT_DIR/mimic-video"
MODEL_DIR="$MIMIC_DIR/model"
VENV="$MODEL_DIR/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: venv missing at $VENV." >&2
  echo "  x86_64 build:  uv sync --extra cu126 in $MODEL_DIR" >&2
  echo "  Jetson Thor:   see scripts/setup_thor.sh" >&2
  exit 1
fi

# Bundled CUDA libs. Layout differs by platform: x86_64 cu126 wheels drop libs
# under nvidia/<pkg>/lib; the Jetson Thor cu13 wheels use nvidia/cu13/lib.
PY_LIB="$VENV/lib/$(basename "$(find "$VENV/lib" -maxdepth 1 -name 'python*' -type d | head -n1)")/site-packages"
THOR_CUDA_LIB="$PY_LIB/nvidia/cu13/lib"
if [[ -d "$THOR_CUDA_LIB" ]]; then
  NVIDIA_LIBS="$THOR_CUDA_LIB"
else
  export CUDA_HOME="$PY_LIB/nvidia"
  NVIDIA_LIBS="$(find "$CUDA_HOME" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':' | sed 's/:$//')"
fi
export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Cosmos / imaginaire are loaded from source, not installed.
export PYTHONPATH="$MODEL_DIR${PYTHONPATH:+:$PYTHONPATH}"

# HF cache. Honor a user-set HF_HOME; otherwise use the standard ~/.cache location.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# HF auth — token file from the standard HF cache location.
if [[ -z "${HF_TOKEN:-}" && -r "$HF_HOME/token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HF_HOME/token")"
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Activate the venv so `python`/`lerobot-record` resolve to the right binaries.
# shellcheck disable=SC1091
source "$VENV/bin/activate"
