#!/usr/bin/env bash
# Set up the mimic-video inference venv on a Jetson Thor (aarch64, CUDA 13).
#
# This is the aarch64 counterpart to .brev/setup.sh (which targets x86_64 +
# cu126). It uses the jetson-ai-lab.io sbsa/cu130 wheel index for the
# GPU-specific packages (torch, flash-attn, cuDSS) and skips the four
# x86-only deps from cosmos_predict2 (apex, transformer-engine, natten,
# triton-from-cosmos-index) — the cosmos sources are patched on the
# `thor_inference` branch so they import cleanly without TE/apex/decord.
#
# Usage:
#   bash scripts/setup_thor.sh                            # full install
#   DRY_RUN=1 bash scripts/setup_thor.sh                  # print steps
#   SKIP_CHECKPOINTS=1 bash scripts/setup_thor.sh         # venv only

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PDT_DIR/mimic-video/model"
VENV="$MODEL_DIR/.venv"

JETSON_INDEX="${JETSON_INDEX:-https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple/}"
PY_VERSION="${PY_VERSION:-3.12}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
[dry-run] Would do, in order:
  1. uv venv --python $PY_VERSION $VENV
  2. uv pip install (from $JETSON_INDEX) torch nvidia-cudss-cu13 flash-attn
  3. uv pip install cosmos runtime deps + lerobot
  4. uv pip install -e $PDT_DIR/lerobot -e $PDT_DIR/lerobot_policy_mimic_video --no-deps
  5. bash scripts/fetch_deploy_checkpoints.sh
  6. snapshot_download jonpai/mimic-video tokenizer/text_encoder structure
  7. snapshot_download push-that-thing/task_1 .t5_embeddings/
  8. snapshot_download push-that-thing/model action_decoder/stats_cache/
EOF
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "$HOME/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
  else
    echo "ERROR: HF_TOKEN unset and no token at ~/.cache/huggingface/token." >&2
    echo "       Run: hf auth login" >&2
    exit 1
  fi
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "==> 1. Creating venv at $VENV (Python $PY_VERSION)"
cd "$MODEL_DIR"
uv venv --python "$PY_VERSION" "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "==> 2. Installing torch + cuDSS + flash-attn from $JETSON_INDEX"
uv pip install --extra-index-url "$JETSON_INDEX" \
  "torch>=2.9,<2.11" nvidia-cudss-cu13 flash-attn

echo "==> 3. Installing cosmos + lerobot runtime deps"
uv pip install --extra-index-url "$JETSON_INDEX" \
  attrs einops hydra-core omegaconf peft safetensors transformers \
  sentencepiece ftfy numpy opencv-python mediapy webdataset tqdm \
  diffusers iopath fvcore boto3 "imageio[ffmpeg]" pillow loguru pyyaml \
  threadpoolctl "protobuf<5" termcolor better-profanity nltk easydict \
  pynvml scikit-image qwen-vl-utils retinaface-py pycocotools attrdict \
  pandas wandb absl-py megatron-core modelscope tyro av ipdb h11 \
  draccus jsonlines torchcodec deepdiff gymnasium "datasets>=4" rerun-sdk \
  pyserial

echo "==> 4. Installing lerobot + mimic policy in editable mode (no deps)"
uv pip install --no-deps -e "$PDT_DIR/lerobot" -e "$PDT_DIR/lerobot_policy_mimic_video"

if [[ "${SKIP_CHECKPOINTS:-0}" == "1" ]]; then
  echo "SKIP_CHECKPOINTS=1, exiting before checkpoint fetch."
  exit 0
fi

echo "==> 5. Fetching action checkpoint + fusing video backbone"
bash "$PDT_DIR/scripts/fetch_deploy_checkpoints.sh"

echo "==> 6. Pulling cosmos tokenizer + T5 dir structure from jonpai/mimic-video"
cd "$MODEL_DIR"
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
ckpt = Path("checkpoints")
ckpt.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="jonpai/mimic-video",
    local_dir=str(ckpt),
    allow_patterns=["text_encoder/*", "video_backbone/tokenizer/*"],
)
PY

echo "==> 7. Pulling precomputed T5 prompt embeddings"
python - <<'PY'
from huggingface_hub import snapshot_download
import pathlib
snapshot_download(
    repo_id="push-that-thing/task_1",
    repo_type="dataset",
    local_dir=str(pathlib.Path.home() / ".cache" / "mimic_lerobot_t5"),
    allow_patterns=[".t5_embeddings/*"],
)
PY

echo "==> 8. Pulling normalizer stats cache"
python - <<'PY'
from huggingface_hub import snapshot_download
import pathlib, shutil
tmp = pathlib.Path("/tmp/stats_dl_setup")
target = pathlib.Path.home() / ".cache" / "mimic_lerobot_stats"
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="push-that-thing/model",
    repo_type="model",
    local_dir=str(tmp),
    allow_patterns=["action_decoder/stats_cache/*"],
)
src = tmp / "action_decoder" / "stats_cache"
for f in src.iterdir():
    shutil.copy2(f, target / f.name)
print("stats cache:", target)
PY

echo ""
echo "Setup complete. Smoke-test with:"
echo "  source scripts/mimic_env.sh"
echo "  python -c 'from lerobot_policy_mimic_video.modeling_mimic_video import MimicVideoPolicy; print(\"ok\")'"
echo ""
echo "Deploy on the SO-101 arm with:"
echo "  bash scripts/deploy_so101.sh"
