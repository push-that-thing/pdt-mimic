#!/usr/bin/env python
"""Release SO-101 follower torque so the arm can be repositioned by hand.

Called by scripts/deploy_so101.sh on exit / Ctrl-C. lerobot only calls
robot.disconnect() (which disables torque via disable_torque_on_disconnect) on a
*clean* finish, so a Ctrl-C mid-rollout leaves the arm stiff. This reconnects
only the motor bus — no cameras, no calibration, no goal-position writes, so the
arm never moves — then disconnects to drop torque.

The just-killed deploy process may still hold the serial port for a moment, so
opening is retried briefly.
"""
import argparse
import time

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", default=None)
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))

    last_err: Exception | None = None
    for _ in range(args.retries):
        try:
            # Open the serial port only: no handshake, no configure(), no
            # calibration, no movement — nothing that could energise the arm.
            robot.bus.connect(handshake=False)
        except Exception as e:  # port may still be held by the just-killed deploy
            last_err = e
            time.sleep(args.delay)
            continue
        # disconnect() -> bus.disconnect(disable_torque_on_disconnect=True): torque off.
        robot.disconnect()
        print(f"Torque released on {args.port} — the arm is free to reposition.")
        return 0

    print(f"Could not release torque on {args.port}: {last_err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
