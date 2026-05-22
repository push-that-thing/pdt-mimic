#!/usr/bin/env bash
# One-time machine setup for SO-101 MIMIC video-action deployment.
#
# Does, in order: install uv -> init submodules -> uv sync (cu126 cosmos deps)
# -> install lerobot runtime deps -> repin torch to cu128 (Blackwell / RTX 5090)
# -> install lerobot + policy + hardware deps -> download video backbone +
# tokenizer + T5 cache. Re-running rebuilds a broken venv cleanly.
#
# After this, deploy with:  bash run_eval_task1.sh   (or task2)
#
# Usage:
#   bash setup.sh            # HF_TOKEN is optional; set it only for private HF repos
#
# Tunables (env):
#   TORCH_VERSION (2.7.0)  TORCHVISION_VERSION (0.22.0)  CUDA_TAG (cu128)
#   HF_MODEL_REPO  VIDEO_CKPT_FILE  VIDEO_CKPT_NAME  (the shared video backbone)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$ROOT_DIR/mimic-video/model"
CKPT_DIR="$MODEL_DIR/checkpoints"

TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
CUDA_TAG="${CUDA_TAG:-cu128}"

HF_MODEL_REPO="${HF_MODEL_REPO:-push-that-thing/cosmos-task1-task2}"
VIDEO_CKPT_FILE="${VIDEO_CKPT_FILE:-posttraining/video2world/v2w_push_combined_lora_rank32_lr1.778e-04_bsz32/checkpoints/model/iter_000002500_fused.pt}"
VIDEO_CKPT_NAME="${VIDEO_CKPT_NAME:-v2w_push_combined_lora_rank32_iter2500_fused}"

# HF_TOKEN is only needed for *private* HF repos. The push-that-thing/* and
# jonpai/mimic-video assets are public, so downloads work anonymously; set the
# token only if you point this at a private repo.
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
else
  echo "==> HF_TOKEN not set — downloading public HF assets anonymously"
fi

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2. Submodules: model code (mimic-video) + lerobot for hardware I/O.
#    A vendored zip already ships these populated, so fetch only when missing —
#    and only if this is a real git checkout with submodule metadata.
if [[ -f "$MODEL_DIR/pyproject.toml" && -f "$ROOT_DIR/lerobot/pyproject.toml" ]]; then
  echo "==> Submodules already present (vendored) — skipping git submodule init"
elif [[ -f "$ROOT_DIR/.gitmodules" ]] && git -C "$ROOT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  echo "==> Initializing submodules"
  git -C "$ROOT_DIR" submodule update --init mimic-video lerobot
else
  echo "ERROR: submodules not vendored and no git metadata to fetch them." >&2
  echo "       Populate mimic-video/ and lerobot/ — see README 'Git / submodule setup'." >&2
  exit 1
fi

# 3. venv + cosmos deps
echo "==> uv sync (cu126 cosmos deps) in $MODEL_DIR"
cd "$MODEL_DIR"
uv sync --extra cu126
source .venv/bin/activate

# 4. lerobot runtime deps (lerobot itself is installed --no-deps in step 6).
#    Installed BEFORE the cu128 repin so torch-dependent wheels don't pull a
#    cu126 torch over it. rerun-sdk is held <0.24 on purpose: 0.24+ requires
#    numpy>=2, which conflicts with cosmos's pinned numpy==1.26.4.
echo "==> Installing lerobot runtime deps"
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
  "rerun-sdk>=0.23.0,<0.24.0" \
  "wandb==0.19.11"

# 5. Blackwell / RTX 5090 (sm_120) needs a cu128 torch build. Do this LAST so
#    nothing re-pulls a cu126 torch over it.
echo "==> Repinning torch $TORCH_VERSION + torchvision $TORCHVISION_VERSION ($CUDA_TAG)"
uv pip install --reinstall \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# 6. lerobot + policy plugin + hardware deps (--no-deps: don't disturb the stack).
echo "==> Installing lerobot, mimic_video policy, hardware deps"
uv pip install --no-deps -e "$ROOT_DIR/lerobot"
uv pip install --no-deps -e "$ROOT_DIR/lerobot_policy_mimic_video"
uv pip install -r "$ROOT_DIR/requirements-deploy.txt"

# 7. Shared model assets -> BYOP default paths.
echo "==> Downloading tokenizer"
mkdir -p "$CKPT_DIR/video_backbone/tokenizer"
hf download jonpai/mimic-video "video_backbone/tokenizer/tokenizer.pth" --local-dir "$CKPT_DIR"

echo "==> Downloading video backbone ($VIDEO_CKPT_NAME)"
export CKPT_DIR HF_MODEL_REPO VIDEO_CKPT_FILE VIDEO_CKPT_NAME MODEL_DIR
python - <<'PY'
import os, shutil, subprocess, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ckpt_dir = Path(os.environ["CKPT_DIR"])
target = ckpt_dir / "video_backbone" / f"{os.environ['VIDEO_CKPT_NAME']}.pt"
target.parent.mkdir(parents=True, exist_ok=True)
repo, requested = os.environ["HF_MODEL_REPO"], os.environ["VIDEO_CKPT_FILE"]
files = set(HfApi().list_repo_files(repo_id=repo))

needs_fuse = False
filename = requested if requested in files else None
if filename is None and requested.endswith("_fused.pt"):
    unfused = requested[:-len("_fused.pt")] + ".pt"
    filename = unfused if unfused in files else None
    needs_fuse = filename is not None
if filename is None:
    raise SystemExit(f"Could not find '{requested}' (or unfused source) in {repo}.")

path = Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=str(ckpt_dir)))
if needs_fuse:
    fuse = Path(os.environ["MODEL_DIR"]) / "scripts" / "fuse_lora_ckpt.py"
    subprocess.run([sys.executable, str(fuse), str(path)], check=True)
    path = path.with_name(path.stem + "_fused.pt")
if path.resolve() != target.resolve():
    shutil.copyfile(path, target)
print(f"Video backbone ready at {target}")
PY

echo "==> Warming T5 prompt-embedding caches (task_1, task_2, task_3_1)"
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

for repo, sub in [("push-that-thing/task_1", "task1"), ("push-that-thing/task_2", "task2"),
                  ("push-that-thing/task_3_1", "task3_1")]:
    dest = Path.home() / ".cache" / "mimic_lerobot_t5" / sub
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest),
                      allow_patterns=[".t5_embeddings/*"])
    print(f"warmed {repo} -> {dest}/.t5_embeddings")
PY

echo ""
echo "Setup complete. Deploy with:  bash run_eval_task1.sh   (dry-run; LIVE=1 to move the arm)"
