#!/usr/bin/env bash
# Runs once on Brev instance creation.
# Set HF_TOKEN as a Brev secret before launching the instance.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMIC="$REPO_ROOT/mimic-video"
CKPT_DIR="$MIMIC/model/checkpoints"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Add it as a Brev secret." >&2
    exit 1
fi

# ── 1. uv ─────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env"
fi

# ── 2. Python 3.10 training environment ───────────────────────────────────────
cd "$MIMIC/model"
uv sync --extra cu126
source .venv/bin/activate

# ── 3. lerobot (needed by LeRobotDataset at training time) ────────────────────
if [ ! -d "$REPO_ROOT/lerobot" ]; then
    git clone --depth 1 https://github.com/huggingface/lerobot.git "$REPO_ROOT/lerobot"
fi

# ── 4. Text encoder + tokenizer (jonpai/mimic-video) ──────────────────────────
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="jonpai/mimic-video",
    local_dir="$CKPT_DIR",
    allow_patterns=["text_encoder/*", "video_backbone/tokenizer/*"],
)
EOF

# ── 5. Fine-tuned video backbone (push-that-thing/model → v2w_push_that_thing.pt)
if [ ! -f "$CKPT_DIR/video_backbone/v2w_push_that_thing.pt" ]; then
    mkdir -p "$CKPT_DIR/video_backbone"
    python - <<EOF
from huggingface_hub import hf_hub_download
import shutil, pathlib
src = hf_hub_download(
    repo_id="push-that-thing/model",
    filename="checkpoints/model/iter_000000500_fused.pt",
)
dst = pathlib.Path("$CKPT_DIR/video_backbone/v2w_push_that_thing.pt")
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst)
print(f"Saved video backbone to {dst}")
EOF
fi

# ── 6. T5 embeddings (push-that-thing/task_1 → ~/.cache/mimic_lerobot_t5) ─────
python - <<EOF
import pathlib
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="push-that-thing/task_1",
    repo_type="dataset",
    local_dir=pathlib.Path.home() / ".cache" / "mimic_lerobot_t5",
    allow_patterns=[".t5_embeddings/*"],
)
EOF

echo ""
echo "Setup complete. Run scripts/train_action_decoder.sh to start training."
