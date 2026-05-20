"""Slowly drive the SO-101 follower arm to the training-data start pose.

Target = median of state[0] across all 10 episodes of push-that-thing/task_1.
The arm interpolates linearly from its current pose to the target over
DURATION seconds at FPS Hz; per-tick motion is hard-capped to MAX_STEP_DEG
regardless of how big the gap is, so a long path just takes longer rather
than commanding a snap.

Run via:
    source scripts/mimic_env.sh
    python scripts/goto_start_pose.py             # 5 s ramp, 1 deg/tick cap
    python scripts/goto_start_pose.py --duration 10
    python scripts/goto_start_pose.py --dry-run   # print plan, don't move
"""
from __future__ import annotations

import argparse
import signal
import time

import torch

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

# Median of state[0] across all 10 training episodes of push-that-thing/task_1.
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TRAINING_START_POSE = {
    "shoulder_pan":   -0.57,
    "shoulder_lift": -101.19,
    "elbow_flex":     97.05,
    "wrist_flex":     50.68,
    "wrist_roll":      2.02,
    "gripper":         1.08,
}


def build_robot(port: str, robot_id: str, max_step_deg: float) -> SO101Follower:
    # No camera needed for goto-pose, but lerobot requires the cameras dict.
    cfg = SO101FollowerConfig(
        port=port,
        id=robot_id,
        cameras={"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)},
        max_relative_target=max_step_deg,
    )
    return SO101Follower(cfg)


def connect_no_prompt(robot: SO101Follower) -> None:
    if not robot.calibration:
        raise RuntimeError(
            f"No calibration loaded. Expected at {robot.calibration_fpath}. "
            "Check ~/.cache/huggingface/lerobot/calibration/ or $HF_HOME/lerobot/calibration/."
        )
    robot.bus.connect()
    if not robot.bus.is_calibrated:
        print("       motor flash differs from JSON — writing calibration to motors...")
        robot.bus.write_calibration(robot.calibration)
    for cam in robot.cameras.values():
        cam.connect()
    robot.configure()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--robot-id", default="my_awesome_follower_arm")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="Approximate seconds to reach the target pose (if max-step-deg allows).")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-step-deg", type=float, default=1.5,
                    help="Hard cap on per-tick joint motion (degrees). Safer = lower.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan; do not connect bus or move.")
    args = ap.parse_args()

    print("Training start pose target (deg):")
    for j in JOINT_ORDER:
        print(f"  {j:>14}: {TRAINING_START_POSE[j]:+7.2f}")

    print(f"\nConnecting to SO-101 on {args.port}...")
    robot = build_robot(args.port, args.robot_id, args.max_step_deg)
    if args.dry_run:
        print("DRY RUN — would connect, no motion sent.")
        return 0

    connect_no_prompt(robot)

    interrupted = {"stop": False}
    def _sigint(_signum, _frame):
        interrupted["stop"] = True
        print("\n!! SIGINT — stopping at current position.")
    signal.signal(signal.SIGINT, _sigint)

    try:
        raw = robot.get_observation()
        start = {j: float(raw[f"{j}.pos"]) for j in JOINT_ORDER}
        print("\nCurrent pose (deg):")
        for j in JOINT_ORDER:
            print(f"  {j:>14}: {start[j]:+7.2f}   target: {TRAINING_START_POSE[j]:+7.2f}   delta: {TRAINING_START_POSE[j] - start[j]:+7.2f}")

        # How many ticks does the largest delta need at max_step_deg per tick?
        biggest_delta = max(abs(TRAINING_START_POSE[j] - start[j]) for j in JOINT_ORDER)
        ticks_for_step_cap = int(biggest_delta / args.max_step_deg) + 1
        ticks_for_duration = int(args.duration * args.fps)
        n_ticks = max(ticks_for_step_cap, ticks_for_duration)
        print(f"\nRamping over {n_ticks} ticks ({n_ticks / args.fps:.1f} s).")

        dt = 1.0 / args.fps
        next_t = time.perf_counter()
        for i in range(n_ticks):
            if interrupted["stop"]:
                break
            alpha = min(1.0, (i + 1) / n_ticks)
            cmd = {f"{j}.pos": start[j] + alpha * (TRAINING_START_POSE[j] - start[j]) for j in JOINT_ORDER}
            robot.send_action(cmd)
            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        final_raw = robot.get_observation()
        print("\nFinal pose reached (deg):")
        for j in JOINT_ORDER:
            v = float(final_raw[f"{j}.pos"])
            err = TRAINING_START_POSE[j] - v
            print(f"  {j:>14}: {v:+7.2f}   target: {TRAINING_START_POSE[j]:+7.2f}   err: {err:+5.2f}")
    finally:
        print("\nDisconnecting robot...")
        try:
            robot.disconnect()
        except Exception as e:
            print(f"  (disconnect warning: {e!r})")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
