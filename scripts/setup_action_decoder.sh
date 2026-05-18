#!/usr/bin/env bash
# Set up action-decoder LoRA checkpoints and dataset-statistics caches for
# task1 and/or task2.
#
# Usage:
#   bash scripts/setup_action_decoder.sh --iter-task1 100 2500 --iter-task2 100 2500
#   bash scripts/setup_action_decoder.sh --iter-task1 100 --stats-hash-task1 <sha256>

HF_REPO="push-that-thing/model"
TASK1_PATH="action_decoder/chain/w2a_so101_combined_lora32_iter2500_75m_2gpu_bsz2_accum8_val5/checkpoints/model"
TASK2_PATH="action_decoder/chain/w2a_so101_task2_combined_lora32_iter2500_75m_2gpu_bsz2_accum8_val5/checkpoints/model"
STATS_HF_DIR="action_decoder/stats_cache"

# Defaults — override with --stats-hash-task{1,2} <sha256>.
STATS_HASH_TASK1="77c34c080268301503540af7a2f650baef0326ae0c3401fb4e786bfdb0f5ebc6"
STATS_HASH_TASK2="eeaabacd800b60e08ef93a5398f5c3e2e193c6fef2f7af04d47bca467bc69f85"

ITER_TASK1=()
ITER_TASK2=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iter-task1)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ITER_TASK1+=("$1"); shift
      done
      ;;
    --iter-task2)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ITER_TASK2+=("$1"); shift
      done
      ;;
    --stats-hash-task1)
      STATS_HASH_TASK1="$2"; shift 2 ;;
    --stats-hash-task2)
      STATS_HASH_TASK2="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ ${#ITER_TASK1[@]} -eq 0 && ${#ITER_TASK2[@]} -eq 0 ]]; then
  echo "Usage: bash scripts/setup_action_decoder.sh --iter-task1 <iter>... --iter-task2 <iter>..." >&2
  exit 1
fi

cat <<BANNER
================================================================================
  setup_action_decoder.sh
================================================================================
  This script prepares action-decoder LoRA checkpoints for inference. It will:

    1. Download the requested iter_*.pt files for task1 and/or task2 from
       $HF_REPO into <parent>/checkpoints/action-decoder/{task1,task2}/.
       Filenames are zero-padded to 9 digits (iter 100 -> iter_000000100.pt,
       iter 2500 -> iter_000002500.pt).
    2. Download the matching dataset-statistics cache for each engaged task
       from $HF_REPO/$STATS_HF_DIR/<sha256> into
       <parent>/checkpoints/action-decoder/<taskN>/stats_cache/<sha256>.
       Default hashes are hard-coded; override with --stats-hash-task{1,2}.

  Requested:
    task1 iters:       ${ITER_TASK1[*]:-(none)}
    task1 stats hash:  $STATS_HASH_TASK1
    task2 iters:       ${ITER_TASK2[*]:-(none)}
    task2 stats hash:  $STATS_HASH_TASK2

  Requirements: hf CLI on PATH. If it isn't, the script tries to activate
  mimic-video/model/.venv automatically; otherwise run
  'source scripts/setup_video_model.sh' first.

  Usage:
    bash scripts/setup_action_decoder.sh --iter-task1 100 2500 --iter-task2 100 2500
    bash scripts/setup_action_decoder.sh --iter-task1 100 --stats-hash-task1 <sha256>
================================================================================
BANNER

(
  set -euo pipefail

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "$script_dir/.." && pwd)"
  parent_dir="$(cd "$repo_root/.." && pwd)"
  out_root="$parent_dir/checkpoints/action-decoder"

  if ! command -v hf >/dev/null 2>&1; then
    venv="$repo_root/mimic-video/model/.venv/bin/activate"
    [[ -f "$venv" ]] && source "$venv"
  fi
  if ! command -v hf >/dev/null 2>&1; then
    echo "Error: hf CLI not found. Run 'source scripts/setup_video_model.sh' first." >&2
    exit 1
  fi

  download_iters() {
    local task_name="$1" repo_path="$2"
    shift 2
    local iters=("$@")
    local out_dir="$out_root/$task_name"
    mkdir -p "$out_dir"
    for iter in "${iters[@]}"; do
      [[ "$iter" =~ ^[0-9]+$ ]] || { echo "Invalid iter '$iter' (expected integer)" >&2; return 1; }
      local ckpt_name
      ckpt_name="$(printf 'iter_%09d.pt' "$iter")"
      echo "==> Downloading $task_name $ckpt_name"
      hf download "$HF_REPO" "$repo_path/$ckpt_name" --local-dir "$out_dir"
      mv -f "$out_dir/$repo_path/$ckpt_name" "$out_dir/$ckpt_name"
    done
    rm -rf "$out_dir/${repo_path%%/*}"
  }

  download_stats() {
    local task_name="$1" stats_hash="$2"
    [[ "$stats_hash" =~ ^[0-9a-f]{64}$ ]] || {
      echo "Invalid stats hash '$stats_hash' (expected 64-char sha256)" >&2
      return 1
    }
    local out_dir="$out_root/$task_name/stats_cache"
    mkdir -p "$out_dir"
    echo "==> Downloading $task_name stats cache $stats_hash"
    hf download "$HF_REPO" "$STATS_HF_DIR/$stats_hash" --local-dir "$out_dir"
    mv -f "$out_dir/$STATS_HF_DIR/$stats_hash" "$out_dir/$stats_hash"
    rm -rf "$out_dir/${STATS_HF_DIR%%/*}"
  }

  if [[ ${#ITER_TASK1[@]} -gt 0 ]]; then
    download_iters task1 "$TASK1_PATH" "${ITER_TASK1[@]}"
    download_stats task1 "$STATS_HASH_TASK1"
  fi
  if [[ ${#ITER_TASK2[@]} -gt 0 ]]; then
    download_iters task2 "$TASK2_PATH" "${ITER_TASK2[@]}"
    download_stats task2 "$STATS_HASH_TASK2"
  fi
)
