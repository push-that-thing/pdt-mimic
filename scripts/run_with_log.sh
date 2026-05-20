#!/usr/bin/env bash
# Run any command with full crash-capture logging.
#
# Mirrors stdout+stderr to the terminal AND to a timestamped log file, so a
# crash leaves the error on disk even when the terminal dies. Records the
# command, GPU/torch context, and the final exit status (incl. fatal signals).
#
# Usage:
#   scripts/run_with_log.sh python -m lerobot.calibrate --robot.type=so101_follower ...
#   LOG_DIR=/tmp/logs scripts/run_with_log.sh bash run_eval_task1.sh
#
# Env:
#   LOG_DIR   where logs go            (default: $HOME/eval_logs)
#   LOG_TAG   filename tag             (default: derived from the command)

set -uo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

LOG_DIR="${LOG_DIR:-$HOME/eval_logs}"
mkdir -p "$LOG_DIR"
tag="${LOG_TAG:-$(basename "${2:-$1}" | tr -c 'A-Za-z0-9._-' '_')}"
ts="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${tag}_${ts}.log"
ln -sf "$LOG_FILE" "$LOG_DIR/${tag}_latest.log"

# Flush Python immediately, and dump a traceback on fatal signals (SIGSEGV /
# SIGABRT) — crucial for CUDA/Blackwell crashes that otherwise leave no trace.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

{
  echo "===== run_with_log ====="
  echo "time    : $(date -Is)"
  echo "host    : $(hostname)"
  echo "pwd     : $PWD"
  echo "command : $*"
  echo "git     : $(git -C "$PWD" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "--- nvidia-smi ---"
  nvidia-smi 2>&1 || echo "(no nvidia-smi)"
  echo "--- torch ---"
  timeout 30 python -c 'import torch; print("torch", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available(), "| arch", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])' 2>&1 || echo "(torch import failed/timed out)"
  echo "========================"
} | tee "$LOG_FILE"

# Run it. Straight pipe (not process substitution) so tee is synchronized with
# the pipeline and the trailing lines are never truncated.
"$@" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}

{
  echo "========================"
  if [[ $status -gt 128 ]]; then
    sig=$((status - 128))
    echo "CRASH: killed by signal $sig (exit $status) — $(kill -l "$sig" 2>/dev/null || true)"
    echo "--- last kernel messages (OOM / Xid / segfault clues) ---"
    dmesg 2>/dev/null | tail -n 25 || echo "(dmesg needs root; rerun: sudo dmesg | tail -n 25)"
  elif [[ $status -ne 0 ]]; then
    echo "FAILED: exit code $status (see traceback above)"
  else
    echo "OK: exit 0"
  fi
  echo "log saved: $LOG_FILE"
} | tee -a "$LOG_FILE"

exit "$status"
