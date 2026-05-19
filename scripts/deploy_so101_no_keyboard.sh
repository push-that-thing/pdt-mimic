#!/usr/bin/env bash
# Hands-free deploy launcher for Thor when you only have 2 USB-A ports and
# can't keep a keyboard plugged in alongside the SO-101 arm and the camera.
#
# Workflow:
#   1. Plug in keyboard. Source mimic_env.sh. Run this script. Then walk away.
#   2. The script prints big banners, beeps, and times each phase so you can
#      swap USB devices without typing anything.
#   3. Phase A: wait for SO-101 controller to appear at /dev/ttyACM* or /dev/ttyUSB*.
#   4. Phase B: wait for camera to appear at /dev/video*.
#   5. Auto-launch scripts/deploy_so101.sh with the detected port + camera index.
#
# Defaults can be overridden:
#   ARM_TIMEOUT_S=180 CAM_TIMEOUT_S=180 PRE_PHASE_GRACE_S=10 \
#     bash scripts/deploy_so101_no_keyboard.sh
#
# Anything you can normally pass to deploy_so101.sh works too:
#   NUM_EPISODES=3 EPISODE_TIME_S=60 bash scripts/deploy_so101_no_keyboard.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/pdt-mimic-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/no_keyboard_deploy_$(date +%Y%m%d-%H%M%S).log}"

PRE_PHASE_GRACE_S="${PRE_PHASE_GRACE_S:-10}"
ARM_TIMEOUT_S="${ARM_TIMEOUT_S:-180}"
CAM_TIMEOUT_S="${CAM_TIMEOUT_S:-180}"
POST_DETECT_GRACE_S="${POST_DETECT_GRACE_S:-5}"
LAUNCH_GRACE_S="${LAUNCH_GRACE_S:-10}"

# Snapshots of which devices were present at the start so we ignore them when
# polling — anything that appears AFTER the script starts is what we want.
SNAPSHOT_ARM="$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | sort | tr '\n' ' ')"
SNAPSHOT_CAM="$(ls /dev/video* 2>/dev/null | sort | tr '\n' ' ')"

log() { echo -e "$*" | tee -a "$LOG_FILE"; }

# Audible bell + banner to grab attention from across the room.
banner() {
  local msg="$1"
  printf '\a'
  log ""
  log "============================================================"
  log "$msg"
  log "============================================================"
  log ""
}

countdown() {
  local seconds="$1"
  local label="$2"
  for ((i = seconds; i > 0; i--)); do
    printf "\r%s %ds  " "$label" "$i" | tee -a "$LOG_FILE" >/dev/null
    printf "\r%s %ds  " "$label" "$i" >&2
    sleep 1
  done
  printf "\r%s done.    \n" "$label" | tee -a "$LOG_FILE"
}

# Poll for a new device that wasn't there at startup.
wait_for_new_device() {
  local pattern="$1"      # e.g. "/dev/ttyACM* /dev/ttyUSB*"
  local snapshot="$2"     # space-separated list of devices present at start
  local timeout="$3"
  local label="$4"

  local deadline=$(( $(date +%s) + timeout ))
  while (( $(date +%s) < deadline )); do
    # shellcheck disable=SC2086
    for d in $pattern; do
      [[ -e "$d" ]] || continue
      if [[ " $snapshot " != *" $d "* ]]; then
        echo "$d"
        return 0
      fi
    done
    sleep 1
  done
  return 1
}

{
  echo "=== no-keyboard deploy started $(date -Iseconds) ==="
  echo "host: $(hostname)  user: $(whoami)"
  echo "ARM_TIMEOUT_S=$ARM_TIMEOUT_S  CAM_TIMEOUT_S=$CAM_TIMEOUT_S"
  echo "snapshot arm: ${SNAPSHOT_ARM:-(none)}"
  echo "snapshot cam: ${SNAPSHOT_CAM:-(none)}"
} >>"$LOG_FILE"

banner "STEP 1/3 — UNPLUG KEYBOARD, PLUG IN THE SO-101 ARM
After ${PRE_PHASE_GRACE_S}s grace, the script will poll for up to ${ARM_TIMEOUT_S}s.
Log: $LOG_FILE"
countdown "$PRE_PHASE_GRACE_S" "grace before arm detection"

log "Polling for new /dev/ttyACM* or /dev/ttyUSB* ..."
if ARM_PORT="$(wait_for_new_device "/dev/ttyACM* /dev/ttyUSB*" "$SNAPSHOT_ARM" "$ARM_TIMEOUT_S" "arm")"; then
  printf '\a\a'
  banner "ARM DETECTED at $ARM_PORT  (kept; do not unplug)"
else
  banner "ERROR — no new serial port after ${ARM_TIMEOUT_S}s. Exiting."
  exit 1
fi

countdown "$POST_DETECT_GRACE_S" "settling arm"

banner "STEP 2/3 — PLUG IN THE CAMERA INTO THE SECOND USB PORT
Keep the arm plugged in. ${PRE_PHASE_GRACE_S}s grace, then poll for ${CAM_TIMEOUT_S}s."
countdown "$PRE_PHASE_GRACE_S" "grace before camera detection"

log "Polling for new /dev/video* ..."
if CAM_DEV="$(wait_for_new_device "/dev/video*" "$SNAPSHOT_CAM" "$CAM_TIMEOUT_S" "camera")"; then
  CAM_IDX="${CAM_DEV#/dev/video}"
  printf '\a\a'
  banner "CAMERA DETECTED at $CAM_DEV (index $CAM_IDX)"
else
  banner "ERROR — no new /dev/video* after ${CAM_TIMEOUT_S}s. Exiting."
  exit 1
fi

countdown "$POST_DETECT_GRACE_S" "settling camera"

banner "STEP 3/3 — STARTING DEPLOY in ${LAUNCH_GRACE_S}s
  ROBOT_PORT=$ARM_PORT
  CAMERA_IDX=$CAM_IDX
  Episodes: ${NUM_EPISODES:-1}, episode_time_s: ${EPISODE_TIME_S:-15}
Press Ctrl-C now to abort. After this you can leave the keyboard out.
Log: $LOG_FILE"
countdown "$LAUNCH_GRACE_S" "launching deploy"

ROBOT_PORT="$ARM_PORT" CAMERA_IDX="$CAM_IDX" LOG_FILE="$LOG_FILE" \
  exec bash "$SCRIPT_DIR/deploy_so101.sh"
