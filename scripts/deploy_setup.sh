#!/usr/bin/env bash
# Layer SO-101 deployment-only dependencies on top of the mimic-video venv.
#
# Prerequisites:
#   - bash .brev/setup.sh has already run successfully (mimic-video/model/.venv
#     exists, Python 3.10 + cu126, lerobot 0.5 cloned at ./lerobot).
#
# What this script does:
#   - Activates the existing mimic-video venv (no new venv is created).
#   - Installs requirements-deploy.txt (feetech-servo-sdk + pyserial).
#   - Reinstalls lerobot in editable mode with the [feetech] extra, with
#     --no-deps so we never pull a conflicting torch/numpy version into the
#     existing pinned stack.
#   - Sanity-checks the imports the deploy script needs.
#
# Re-running is idempotent.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMIC_DIR="${ROOT_DIR}/mimic-video"
MODEL_DIR="${MIMIC_DIR}/model"
LEROBOT_DIR="${ROOT_DIR}/lerobot"

if [[ ! -d "${MODEL_DIR}/.venv" ]]; then
  echo "ERROR: ${MODEL_DIR}/.venv not found." >&2
  echo "       Run bash .brev/setup.sh first to create the mimic-video venv." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${MODEL_DIR}/.venv/bin/activate"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not on PATH. .brev/setup.sh normally installs it; check your shell." >&2
  exit 1
fi

echo "[deploy_setup] installing requirements-deploy.txt into ${MODEL_DIR}/.venv ..."
uv pip install -r "${ROOT_DIR}/requirements-deploy.txt"

if [[ -d "${LEROBOT_DIR}/.git" ]]; then
  echo "[deploy_setup] reinstalling lerobot (editable, --no-deps) with [feetech] extra ..."
  if ! uv pip install --no-deps -e "${LEROBOT_DIR}[feetech]" 2>/dev/null; then
    echo "[deploy_setup] [feetech] extra not available on this lerobot tag; falling back to plain editable install."
    uv pip install --no-deps -e "${LEROBOT_DIR}"
  fi
else
  echo "[deploy_setup] WARNING: ${LEROBOT_DIR} is not a git checkout. lerobot may not be installed." >&2
fi

echo ""
echo "[deploy_setup] verifying imports:"
python - <<'PY'
import importlib

CHECKS = [
    ("torch", None),
    ("numpy", None),
    ("cv2", None),
    ("pyarrow", None),
    ("lerobot", None),
    ("lerobot.robots.so_follower.so_follower", "SO101Follower"),
    ("lerobot.robots.so_follower.config_so_follower", "SO101FollowerConfig"),
    ("lerobot.cameras.opencv", "OpenCVCameraConfig"),
    ("feetech_servo_sdk", None),
]

ok = True
for mod, attr in CHECKS:
    try:
        m = importlib.import_module(mod)
        version = getattr(m, "__version__", "n/a")
        if attr is not None and not hasattr(m, attr):
            print(f"  {mod}.{attr:<25s} MISSING")
            ok = False
        else:
            label = f"{mod}.{attr}" if attr else mod
            print(f"  {label:<55s} OK ({version})")
    except Exception as exc:
        print(f"  {mod:<55s} FAILED: {exc}")
        ok = False

if not ok:
    raise SystemExit(1)
PY

echo ""
echo "[deploy_setup] complete."
echo ""
echo "Next steps:"
echo "  1) Compute deployment statistics once:"
echo "       python scripts/export_dataset_stats.py \\"
echo "         --dataset-dir dataset/task_1_v3 \\"
echo "         --output stats.json"
echo ""
echo "  2) Dry-run the deploy script (no hardware, no movement):"
echo "       python scripts/deploy_so101_vam.py \\"
echo "         --task \"push the cube\" \\"
echo "         --checkpoint <path-to-action-decoder.pt> \\"
echo "         --video-model-path mimic-video/model/checkpoints/video_backbone/v2w_push_that_thing.pt \\"
echo "         --dataset-statistics-path stats.json"
echo ""
echo "  3) When ready, add --enable-robot --port <serial> --robot-id <id> to actually move the arm."
