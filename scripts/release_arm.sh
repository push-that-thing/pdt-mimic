#!/usr/bin/env bash
# Limp the SO-101 follower so the arm can be moved by hand.
# Connects to the controller and immediately disconnects, which honors
# disable_torque_on_disconnect=True and turns off the motor torque.
#
# Usage:
#   bash scripts/release_arm.sh                       # /dev/ttyACM0
#   ROBOT_PORT=/dev/ttyACM1 bash scripts/release_arm.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mimic_env.sh
source "$SCRIPT_DIR/mimic_env.sh"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-my_awesome_follower_arm}"

python - "$ROBOT_PORT" "$ROBOT_ID" <<'PYEOF'
import sys
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
port, robot_id = sys.argv[1], sys.argv[2]
r = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
r.connect(calibrate=False)
r.disconnect()
print(f"torque released on {port}; arm is movable by hand.")
PYEOF
