#!/usr/bin/env bash
# Shared environment for mimic-video SO-101 deployment.
# Source it: `source scripts/mimic_env.sh`
#
# Paths are derived from this file's location, so it works from any checkout.

set -eo pipefail

PDT_DIR="${PDT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="$PDT_DIR/mimic-video/model"
VENV="$MODEL_DIR/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: venv missing at $VENV. Run 'bash setup.sh' first." >&2
  exit 1
fi

# Bundled CUDA libs (avoids sudo ldconfig — needed by transformer_engine + cosmos).
for _nv in "$VENV"/lib/python*/site-packages/nvidia; do
  [[ -d "$_nv" ]] && export CUDA_HOME="$_nv" && break
done
if [[ -n "${CUDA_HOME:-}" ]]; then
  # find|tr|sed is SIGPIPE-safe (consumers read all input); no `head` under pipefail.
  NVIDIA_LIBS="$(find "$CUDA_HOME" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':' | sed 's/:$//')"
  export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Cosmos / imaginaire are imported from source.
export PYTHONPATH="$MODEL_DIR${PYTHONPATH:+:$PYTHONPATH}"

# HF auth from the environment (export HF_TOKEN before running).
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
