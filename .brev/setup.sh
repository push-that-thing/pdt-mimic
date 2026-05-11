#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMIC_DIR="${ROOT_DIR}/mimic-video"
MODEL_DIR="${MIMIC_DIR}/model"
LEROBOT_DIR="${ROOT_DIR}/lerobot"
CHECKPOINT_DIR="${MODEL_DIR}/checkpoints"

HF_TOKEN="${HF_TOKEN:-}"
HF_DATASET_REPO="${HF_DATASET_REPO:-push-that-thing/task_1}"
HF_MODEL_REPO="${HF_MODEL_REPO:-push-that-thing/model}"
VIDEO_CKPT_NAME="${VIDEO_CKPT_NAME:-v2w_push_that_thing}"
VIDEO_CKPT_FILE="${VIDEO_CKPT_FILE:-video_backbone/${VIDEO_CKPT_NAME}.pt}"
LEROBOT_REF="${LEROBOT_REF:-v0.4.0}"

if [[ -z "${HF_TOKEN}" ]]; then
  echo "HF_TOKEN must be set so private HuggingFace assets can be downloaded." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -d "${LEROBOT_DIR}/.git" ]]; then
  git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}"
fi
git -C "${LEROBOT_DIR}" fetch --tags
git -C "${LEROBOT_DIR}" checkout "${LEROBOT_REF}"

cd "${MODEL_DIR}"
uv sync --extra cu126
source .venv/bin/activate

NVRTC_LIB_DIR="$(python - <<'PY'
import importlib.util
import pathlib

spec = importlib.util.find_spec("nvidia")
if spec is None or spec.submodule_search_locations is None:
    raise SystemExit

nvidia_dir = pathlib.Path(next(iter(spec.submodule_search_locations)))
for path in [nvidia_dir / "cuda_nvrtc" / "lib", *nvidia_dir.glob("cuda_nvrtc*/lib")]:
    if (path / "libnvrtc.so").exists() or list(path.glob("libnvrtc.so*")):
        print(path)
        break
PY
)"
if [[ -n "${NVRTC_LIB_DIR}" ]] && command -v sudo >/dev/null 2>&1; then
  echo "${NVRTC_LIB_DIR}" | sudo tee /etc/ld.so.conf.d/brev-python-nvrtc.conf >/dev/null
  sudo ldconfig
fi

uv pip install \
  "numpy==1.26.4" \
  "datasets>=4.0.0,<4.2.0" \
  "huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0" \
  "av>=15.0.0,<16.0.0" \
  "jsonlines>=4.0.0,<5.0.0" \
  "draccus==0.10.0" \
  "gymnasium>=1.1.1,<2.0.0" \
  "deepdiff>=7.0.1,<9.0.0" \
  "torchcodec>=0.2.1,<0.6.0" \
  "wandb==0.19.11"
uv pip install --no-deps -e "${LEROBOT_DIR}"

python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="jonpai/mimic-video",
    local_dir=str(checkpoint_dir),
    allow_patterns=[
        "text_encoder/*",
        "video_backbone/tokenizer/*",
    ],
)
PY

export CHECKPOINT_DIR HF_MODEL_REPO VIDEO_CKPT_NAME VIDEO_CKPT_FILE
python - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
target = checkpoint_dir / "video_backbone" / f"{os.environ['VIDEO_CKPT_NAME']}.pt"
target.parent.mkdir(parents=True, exist_ok=True)

repo_id = os.environ["HF_MODEL_REPO"]
requested = os.environ["VIDEO_CKPT_FILE"]
candidates = [
    requested,
    target.name,
    f"video_backbone/{target.name}",
]
files = set(HfApi().list_repo_files(repo_id=repo_id))
filename = next((candidate for candidate in candidates if candidate in files), requested)

path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=str(checkpoint_dir),
)
downloaded = Path(path)
if downloaded.resolve() != target.resolve():
    shutil.copyfile(downloaded, target)
print(f"Video checkpoint ready at {target}")
PY

export HF_DATASET_REPO
python - <<'PY'
import os
import pathlib
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_DATASET_REPO"],
    repo_type="dataset",
    local_dir=str(pathlib.Path.home() / ".cache" / "mimic_lerobot_t5"),
    allow_patterns=[".t5_embeddings/*"],
)
print("T5 embedding cache warmed.")
PY

echo "Brev setup complete. Launch with: bash scripts/train_action_decoder.sh"
