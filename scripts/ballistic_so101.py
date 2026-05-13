"""Open-loop ballistic test: generate ONE action chunk and execute it on the SO-101.

Flow:
  1. Connect SO-101 follower (no teleop).
  2. Read one observation (joint state + front camera frame).
  3. Block on policy.predict_action_chunk(...) (~3.5 min on the 3060).
  4. Stream the 90 actions to the robot at 30 FPS (3 s of motion).
  5. Disconnect.

Skips the lerobot-record loop entirely so the slow chunk compute does not race
any FPS watchdog. Use to confirm decoder + calibration on hardware before
chasing real-time inference.

Run via:
    source scripts/mimic_env.sh
    python scripts/ballistic_so101.py
"""
from __future__ import annotations

import argparse
import datetime
import signal
import sys
import time

import numpy as np
import torch

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot_policy_mimic_video import MimicVideoConfig, MimicVideoPolicy

# Joint order used during training (matches the calibration JSON key order).
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def build_robot(port: str, robot_id: str, camera_idx: int, fps: int, max_step_deg: float) -> SO101Follower:
    cam_cfg = OpenCVCameraConfig(index_or_path=camera_idx, width=640, height=480, fps=fps)
    cfg = SO101FollowerConfig(
        port=port,
        id=robot_id,
        cameras={"front": cam_cfg},
        # Hard safety belt: cap the per-tick relative motion regardless of what the policy outputs.
        max_relative_target=max_step_deg,
    )
    return SO101Follower(cfg)


def build_policy(use_fp16_state: bool) -> MimicVideoPolicy:
    cfg = MimicVideoConfig(
        use_text_encoder=False,
        load_on_init=False,
        n_action_steps=90,  # ballistic: consume the whole chunk in one go
    )
    cfg.input_features = cfg.default_input_features
    cfg.output_features = cfg.default_output_features
    policy = MimicVideoPolicy(cfg)
    policy.load_mimic_pipeline()
    if use_fp16_state:
        policy._dummy = policy._dummy.to(torch.float16)
    return policy


def raw_obs_to_policy_batch(raw_obs: dict, camera_key: str, state_key: str) -> dict:
    state_vals = [float(raw_obs[f"{j}.pos"]) for j in JOINT_ORDER]
    state = torch.tensor(state_vals, dtype=torch.float32)
    image = raw_obs["front"]
    return {camera_key: image, state_key: state}


def action_vec_to_robot_dict(action: torch.Tensor) -> dict:
    vals = action.detach().cpu().float().tolist()
    return {f"{joint}.pos": v for joint, v in zip(JOINT_ORDER, vals)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--robot-id", default="my_awesome_follower_arm")
    ap.add_argument("--camera-idx", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-step-deg", type=float, default=5.0,
                    help="Cap per-tick relative joint motion (degrees). Safety belt against policy outliers.")
    ap.add_argument("--num-actions", type=int, default=90,
                    help="How many of the 90 chunk steps to actually execute. Start small (e.g. 30) on first run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the chunk's actions but do NOT send them to the robot.")
    ap.add_argument("--save-tag", default=None,
                    help="If set, save the chunk + start state to /tmp/chunk_<tag>.npz for comparison.")
    args = ap.parse_args()

    print("[1/5] Loading policy (cosmos backbone + decoder)...")
    policy = build_policy(use_fp16_state=False)
    print("       policy loaded.")

    print(f"[2/5] Connecting to SO-101 on {args.port}...")
    robot = build_robot(args.port, args.robot_id, args.camera_idx, args.fps, args.max_step_deg)
    # Skip lerobot's interactive calibrate() flow — robot.calibration is already
    # populated from the JSON, just push it to the motors and proceed.
    if not robot.calibration:
        raise RuntimeError(
            f"No calibration loaded for {args.robot_id}. "
            f"Expected at {robot.calibration_fpath} — check ~/.cache/huggingface/lerobot/calibration/."
        )
    robot.bus.connect()
    if not robot.bus.is_calibrated:
        print("       motor flash differs from JSON — writing calibration to motors...")
        robot.bus.write_calibration(robot.calibration)
    for cam in robot.cameras.values():
        cam.connect()
    robot.configure()
    print("       robot connected. Calibration:", list(robot.calibration.keys()))

    interrupted = {"stop": False}
    def _sigint(_signum, _frame):
        interrupted["stop"] = True
        print("\n!! SIGINT received — will disconnect after current send_action.")
    signal.signal(signal.SIGINT, _sigint)

    try:
        print("[3/5] Reading one observation (state + camera)...")
        raw = robot.get_observation()
        start_state = [float(raw[f"{j}.pos"]) for j in JOINT_ORDER]
        print("       initial state (deg):", [f"{v:+.2f}" for v in start_state])

        batch = raw_obs_to_policy_batch(raw, policy.config.camera_key, policy.config.state_key)
        batch["task"] = policy.config.task

        print(f"[4/5] Generating action chunk on GPU (this takes ~3.5 min on the 3060)...")
        t0 = time.time()
        chunk = policy.predict_action_chunk(batch)  # shape (1, 90, 6)
        chunk = chunk[0].detach().cpu().float()
        print(f"       chunk ready in {time.time() - t0:.1f}s   shape={tuple(chunk.shape)}")

        if args.save_tag:
            ts = datetime.datetime.now().strftime("%H%M%S")
            out = f"/tmp/chunk_{args.save_tag}_{ts}.npz"
            np.savez(out,
                     chunk=chunk.numpy(),
                     start_state=np.array(start_state, dtype=np.float32),
                     joint_order=np.array(JOINT_ORDER),
                     task=np.array(policy.config.task))
            print(f"       saved chunk + start_state to {out}")

        n_exec = min(args.num_actions, chunk.shape[0])
        print(f"       first action: {chunk[0].tolist()}")
        print(f"       last  action ({n_exec - 1}): {chunk[n_exec - 1].tolist()}")
        print(f"       delta vs start (last-start, deg): {[f'{chunk[n_exec-1][i].item() - start_state[i]:+.2f}' for i in range(6)]}")

        if args.dry_run:
            print("[5/5] DRY RUN — not sending any commands. Done.")
            return 0

        print(f"[5/5] Streaming {n_exec} actions at {args.fps} FPS ({n_exec / args.fps:.1f} s of motion)...")
        dt = 1.0 / args.fps
        next_t = time.perf_counter()
        for i in range(n_exec):
            if interrupted["stop"]:
                break
            action_dict = action_vec_to_robot_dict(chunk[i])
            robot.send_action(action_dict)
            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
        print(f"       sent {i + 1}/{n_exec} actions.")
    finally:
        print("Disconnecting robot...")
        try:
            robot.disconnect()
        except Exception as e:
            print(f"  (disconnect warning: {e!r})")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
