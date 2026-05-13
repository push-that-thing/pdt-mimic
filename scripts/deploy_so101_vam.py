#!/usr/bin/env python3
"""Standalone deployment of a MimicVideo / VAM SO-101 checkpoint on a real arm.

This script connects to a LeRobot SO-101 follower (lerobot >= 0.5), reads
joint state + a camera frame, runs MimicVideo inference via the native
``Video2World2ActionPipeline``, and converts the predicted 90-step action
chunk back into LeRobot SO-101 action dicts. By default it never moves the
robot (dry-run). Movement requires ``--enable-robot`` explicitly.

Two execution modes:

* **Manual burst mode** (default, safest): wait for Enter, plan one chunk
  of actions, send the first few of them, hold, settle, repeat.
* **Continuous mode** (opt-in, ``--continuous``): continuously read
  observations and replan when the action buffer is exhausted.

Safety:
  * Default is dry-run. ``robot.send_action(...)`` is only called when
    ``--enable-robot`` is passed.
  * Every commanded action is clamped to ``current_state +/- max_delta``
    per joint.
  * Hard preflight checks: joint keys, image key, action dimension.
  * try/finally guarantees ``robot.disconnect()``.

Units are exactly what lerobot 0.5 SO101Follower (``use_degrees=True``,
its default) produces and consumes: degrees for joints 0..4, percent
[0, 100] for the gripper. No unit conversion is done in this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
DEFAULT_MAX_DELTA: tuple[float, ...] = (0.04, 0.04, 0.04, 0.04, 0.06, 0.10)
DEFAULT_EXPERIMENT = "w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1"

ACTION_DIM = 6
STATE_DIM = 6
IMG_H, IMG_W = 480, 640
OBS_HORIZON_IMG_DEFAULT = 5
OBS_HORIZON_LOWDIM_DEFAULT = 1
ACTION_HORIZON_DEFAULT = 90  # only used as a sanity check


# --------------------------------------------------------------------------- #
# Path bootstrap so mimic-video and lerobot can be imported
# --------------------------------------------------------------------------- #

def _bootstrap_pythonpath() -> None:
    """Insert mimic-video/model and lerobot/src into sys.path.

    This mirrors the PYTHONPATH that ``scripts/train_action_decoder.sh`` sets up.
    Both paths are optional — if the modules are already importable (e.g.
    because they're installed in the active venv), we don't add anything.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    candidates = [
        repo_root / "mimic-video" / "model",
        repo_root / "lerobot" / "src",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_bootstrap_pythonpath()


# --------------------------------------------------------------------------- #
# Lazy heavy imports
# --------------------------------------------------------------------------- #

def _import_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:
        raise ImportError(
            "Could not import torch. Make sure the mimic-video venv "
            "(mimic-video/model/.venv) is activated."
        ) from exc


def _import_cv2():
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError as exc:
        raise ImportError(
            "Could not import cv2 (opencv-python). It is part of the "
            "mimic-video env; activate mimic-video/model/.venv."
        ) from exc


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #

def _stats_lists_to_arrays(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """Convert a JSON-loaded stats dict back to numpy arrays.

    The mimic-video normalizer expects ``np.ndarray`` values for the per-field
    statistics; ``json.load`` produces nested lists.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for group, fields in stats.items():
        out[group] = {key: np.asarray(value, dtype=np.float32) for key, value in fields.items()}
    return out


# --------------------------------------------------------------------------- #
# Pipeline loading
# --------------------------------------------------------------------------- #

def load_vam_pipeline(args: argparse.Namespace):
    """Build the ``Video2World2ActionPipeline`` for the SO-101 checkpoint.

    This is a thin specialization of ``load_video2world2action_pipeline`` from
    ``mimic-video/eval/libero/run.py``. We swap the in-process dataset access
    for a pre-computed ``stats.json`` so that deployment has no dependency on
    HuggingFace network access or the lerobot dataset adapter.
    """
    torch = _import_torch()

    from cosmos_predict2.configs.config import make_config
    from cosmos_predict2.data.action.utils import extract_normalization_types
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
    from cosmos_predict2.pipelines.world2action import World2ActionPipeline
    from imaginaire.lazy_config import instantiate
    from imaginaire.utils.config_helper import override

    config = make_config()
    config = override(config, ["--", f"experiment={args.experiment}"])
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    dtype = torch.bfloat16

    print(f"[VAM] Loading video backbone from {args.video_model_path} ...")
    v2w = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=args.video_model_path,
        device=args.device,
        torch_dtype=dtype,
        load_ema_to_reg=False,
    )

    print(f"[VAM] Loading action decoder from {args.action_model_path} ...")
    w2a = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=args.action_model_path,
        device=args.device,
        dtype=dtype,
    )

    data_config = instantiate(config.data_config)

    print(f"[VAM] Loading dataset statistics from {args.dataset_statistics_path} ...")
    with open(args.dataset_statistics_path) as f:
        stats_raw = json.load(f)
    stats = _stats_lists_to_arrays(stats_raw)

    expected_groups = {"obs/lowdim_concat", "action/lowdim_concat"}
    missing = expected_groups - stats.keys()
    if missing:
        raise ValueError(
            f"{args.dataset_statistics_path} is missing required groups: {sorted(missing)}. "
            "Did you generate it with scripts/export_dataset_stats.py?"
        )

    state_dim = stats["obs/lowdim_concat"]["mean"].shape[-1]
    action_dim = stats["action/lowdim_concat"]["mean"].shape[-1]
    if state_dim != STATE_DIM or action_dim != ACTION_DIM:
        raise ValueError(
            f"stats.json reports state_dim={state_dim} and action_dim={action_dim}, "
            f"but the SO-101 model expects state_dim={STATE_DIM} and action_dim={ACTION_DIM}."
        )

    w2a.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device=args.device,
        dtype=dtype,
    )

    pipeline = Video2World2ActionPipeline(v2w, w2a)
    if args.device == "cuda":
        pipeline = pipeline.cuda()
    return pipeline


# --------------------------------------------------------------------------- #
# Observation adapter
# --------------------------------------------------------------------------- #

@dataclass
class SO101Observation:
    """One sampled observation from the SO-101 follower."""

    state: np.ndarray  # shape (6,) float32 — degrees x5 + percent
    image: np.ndarray  # shape (H, W, 3) uint8 RGB
    timestamp: float


class SO101ObservationAdapter:
    """Turn LeRobot SO-101 observations into model-ready tensors."""

    def __init__(
        self,
        joint_order: list[str],
        image_key: str,
        img_horizon: int,
        lowdim_horizon: int,
        image_size: tuple[int, int] = (IMG_H, IMG_W),
    ):
        self.joint_order = list(joint_order)
        self.image_key = image_key
        self.img_horizon = img_horizon
        self.lowdim_horizon = lowdim_horizon
        self.image_size = image_size
        self._img_history: deque = deque(maxlen=img_horizon)
        self._state_history: deque = deque(maxlen=lowdim_horizon)

    def reset(self) -> None:
        self._img_history.clear()
        self._state_history.clear()

    def read(self, robot: Any) -> SO101Observation:
        """Read one observation from the robot and validate its shape."""
        obs_dict = robot.get_observation()

        missing_joints = [f"{j}.pos" for j in self.joint_order if f"{j}.pos" not in obs_dict]
        if missing_joints:
            raise KeyError(
                f"robot.get_observation() is missing joint keys: {missing_joints}. "
                f"Keys present: {sorted(obs_dict.keys())}"
            )
        if self.image_key not in obs_dict:
            raise KeyError(
                f"robot.get_observation() is missing image key '{self.image_key}'. "
                f"Keys present: {sorted(obs_dict.keys())}"
            )

        state = np.array(
            [float(obs_dict[f"{j}.pos"]) for j in self.joint_order],
            dtype=np.float32,
        )

        image = obs_dict[self.image_key]
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        if image.dtype != np.uint8:
            raise TypeError(
                f"Image dtype is {image.dtype}; expected uint8. "
                "Did your camera config use the right pixel format?"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Image shape is {image.shape}; expected (H, W, 3)."
            )

        return SO101Observation(state=state, image=image, timestamp=time.time())

    def push(self, obs: SO101Observation) -> None:
        """Add an observation to history. Pads short histories with copies."""
        self._img_history.append(obs.image)
        self._state_history.append(obs.state)
        while len(self._img_history) < self.img_horizon:
            self._img_history.append(obs.image.copy())
        while len(self._state_history) < self.lowdim_horizon:
            self._state_history.append(obs.state.copy())

    def _process_image(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        """Resize, normalize to [-1, 1], reshape to (C, 1, H, W)."""
        cv2 = _import_cv2()
        target_h, target_w = self.image_size
        if image_rgb_uint8.shape[:2] != (target_h, target_w):
            image_rgb_uint8 = cv2.resize(
                image_rgb_uint8,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
        x = image_rgb_uint8.astype(np.float32) / 255.0
        x = 2.0 * x - 1.0
        x = x.transpose(2, 0, 1)
        return x[:, None, :, :]

    def build_model_input(self, device: str, dtype: Any) -> tuple[Any, Any]:
        """Return ``(input_vid, state_B_HO_O)`` tensors ready for the pipeline.

        Shapes:
            input_vid     : (1, 3, T_img, H, W)   bfloat16 cuda
            state_B_HO_O  : (1, T_state, 6)       bfloat16 cuda
        """
        torch = _import_torch()
        if len(self._img_history) == 0 or len(self._state_history) == 0:
            raise RuntimeError(
                "Observation history is empty. Call push() at least once before build_model_input()."
            )

        processed = [self._process_image(img) for img in self._img_history]
        imgs = np.concatenate(processed, axis=1)  # (3, T, H, W)
        input_vid = torch.from_numpy(imgs[None]).to(device=device, dtype=dtype)

        state = np.stack(list(self._state_history), axis=0)  # (T_state, 6)
        state_B_HO_O = torch.from_numpy(state[None]).to(device=device, dtype=dtype)
        return input_vid, state_B_HO_O


# --------------------------------------------------------------------------- #
# Inference wrapper
# --------------------------------------------------------------------------- #

class VAMInference:
    """Thin wrapper around ``Video2World2ActionPipeline`` for one-shot inference."""

    def __init__(
        self,
        pipeline: Any,
        num_sampling_step: int,
        stop_after_step: int,
        use_cuda_graphs: bool,
        seed: int = 0,
    ):
        self.pipeline = pipeline
        self.num_sampling_step = num_sampling_step
        self.stop_after_step = stop_after_step
        self.use_cuda_graphs = use_cuda_graphs
        self.seed = seed

    def infer(self, input_vid: Any, state_B_HO_O: Any, task: str) -> np.ndarray:
        """Run one denoising and return the predicted action chunk.

        Returns a ``(H, 6)`` numpy array, where H is the model's action horizon
        (90 for the SO-101 model).
        """
        torch = _import_torch()
        with torch.no_grad():
            pred = self.pipeline(
                input_vid=input_vid,
                state_B_HO_O=state_B_HO_O,
                prompt=task,
                num_sampling_step=self.num_sampling_step,
                stop_after_step=self.stop_after_step,
                seed=self.seed,
                use_cuda_graphs=self.use_cuda_graphs,
            )
        actions = pred[0].float().cpu().numpy()
        if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"Pipeline returned actions of shape {actions.shape}; "
                f"expected (H, {ACTION_DIM})."
            )
        return actions


# --------------------------------------------------------------------------- #
# Action adapter
# --------------------------------------------------------------------------- #

@dataclass
class SO101ActionAdapter:
    """Convert a predicted 6-vector to a LeRobot SO-101 action dict + clamp."""

    joint_order: list[str]
    max_delta: np.ndarray
    mode: str = "absolute"
    action_mean: np.ndarray | None = None
    action_std: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.max_delta = np.asarray(self.max_delta, dtype=np.float32)
        if self.max_delta.shape != (ACTION_DIM,):
            raise ValueError(
                f"max_delta has shape {self.max_delta.shape}; expected ({ACTION_DIM},)."
            )
        if self.mode not in {"absolute", "delta"}:
            raise ValueError(f"action_mode must be 'absolute' or 'delta', got {self.mode!r}.")
        if (self.action_mean is None) != (self.action_std is None):
            raise ValueError(
                "Provide both --action-mean-path and --action-std-path, or neither."
            )

    def maybe_renormalize(self, pred: np.ndarray) -> np.ndarray:
        if self.action_mean is None or self.action_std is None:
            return pred
        return (pred - self.action_mean) / np.where(self.action_std == 0.0, 1.0, self.action_std)

    def to_action_dict(
        self,
        pred: np.ndarray,
        current_state: np.ndarray,
    ) -> tuple[dict[str, float], bool]:
        """Return ``({joint.pos: float}, clamped)`` for a single predicted action."""
        if pred.shape != (ACTION_DIM,):
            raise ValueError(f"Predicted action shape {pred.shape} != ({ACTION_DIM},)")
        if current_state.shape != (STATE_DIM,):
            raise ValueError(f"Current state shape {current_state.shape} != ({STATE_DIM},)")

        pred = self.maybe_renormalize(pred.astype(np.float32))
        current_state = current_state.astype(np.float32)

        target = current_state + pred if self.mode == "delta" else pred
        delta = target - current_state
        clipped_delta = np.clip(delta, -self.max_delta, +self.max_delta)
        safe_target = current_state + clipped_delta
        clamped = bool(np.any(np.not_equal(clipped_delta, delta)))

        action_dict = {
            f"{name}.pos": float(safe_target[i])
            for i, name in enumerate(self.joint_order)
        }
        return action_dict, clamped


# --------------------------------------------------------------------------- #
# Robot construction (real and dummy)
# --------------------------------------------------------------------------- #

@dataclass
class DummyRobot:
    """Stand-in robot for dry-runs that never touches hardware."""

    joint_order: list[str]
    image_key: str = "front"
    image_size: tuple[int, int] = (IMG_H, IMG_W)
    is_connected: bool = field(default=False, init=False)
    _step: int = field(default=0, init=False)

    def connect(self) -> None:
        self.is_connected = True
        print("[DummyRobot] connected (no hardware I/O)")

    def disconnect(self) -> None:
        self.is_connected = False
        print("[DummyRobot] disconnected")

    def get_observation(self) -> dict[str, Any]:
        self._step += 1
        obs: dict[str, Any] = {f"{j}.pos": 0.0 for j in self.joint_order}
        obs[self.image_key] = np.zeros(
            (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
        )
        return obs

    def send_action(self, action_dict: dict[str, float]) -> dict[str, float]:
        print(f"[DummyRobot] send_action: {action_dict}")
        return action_dict


def make_robot(args: argparse.Namespace) -> Any:
    """Build a real ``SO101Follower`` or a ``DummyRobot`` depending on flags."""
    if not args.enable_robot:
        print("[deploy] --enable-robot NOT set: using DummyRobot (no hardware will be touched).")
        return DummyRobot(joint_order=args.joint_order, image_key=args.image_key)

    try:
        from lerobot.robots.so_follower.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.cameras.opencv import OpenCVCameraConfig
    except ImportError as exc:
        raise ImportError(
            "Could not import lerobot 0.5 SO-101 hardware classes. "
            "Run bash scripts/deploy_setup.sh to install the required dependencies."
        ) from exc

    print(
        f"[deploy] connecting to SO-101 on port={args.port}, id={args.robot_id}, "
        f"camera={args.image_key}@/dev/video{args.camera_index}, "
        f"640x480@{args.camera_fps}fps."
    )
    cam_cfg = OpenCVCameraConfig(
        index_or_path=args.camera_index,
        fps=args.camera_fps,
        width=640,
        height=480,
    )
    cfg = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        use_degrees=True,
        max_relative_target=None,
        cameras={args.image_key: cam_cfg},
    )
    return SO101Follower(cfg)


# --------------------------------------------------------------------------- #
# Execution loops
# --------------------------------------------------------------------------- #

def run_burst(
    args: argparse.Namespace,
    robot: Any,
    obs_adapter: SO101ObservationAdapter,
    vam: VAMInference,
    act_adapter: SO101ActionAdapter,
) -> None:
    print(
        f"\n[burst mode] fps={args.fps}, actions_per_burst={args.actions_per_burst}, "
        f"settle={args.settle_time:.2f}s, dry_run={args.dry_run}, enable_robot={args.enable_robot}"
    )
    print(f"             joints={obs_adapter.joint_order}")
    print(f"             max_delta={list(act_adapter.max_delta)}")
    if args.manual_trigger:
        print("             Press Enter to plan a burst. Type 'q' + Enter to quit.\n")
    else:
        print("             Auto-stepping (no manual trigger).\n")

    torch = _import_torch()
    period = 1.0 / float(args.fps)
    burst_id = 0

    while True:
        if args.manual_trigger:
            try:
                user_input = input("[burst] >> ")
            except (EOFError, KeyboardInterrupt):
                print("\n[burst] interrupted; exiting.")
                return
            if user_input.strip().lower() in {"q", "quit", "exit"}:
                print("[burst] exiting on user request.")
                return

        burst_id += 1
        obs_adapter.reset()
        current = obs_adapter.read(robot)
        obs_adapter.push(current)

        if burst_id == 1:
            print(f"[burst] First observation keys: {sorted(robot.get_observation().keys())}")
            print(f"[burst] Initial joint state  : {current.state}")

        input_vid, state_BHO = obs_adapter.build_model_input(
            device=args.device,
            dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        )

        t0 = time.time()
        chunk = vam.infer(input_vid, state_BHO, args.task)
        infer_ms = (time.time() - t0) * 1000.0
        print(
            f"[burst {burst_id}] planned {chunk.shape[0]} actions in {infer_ms:.0f} ms. "
            f"first-action target = {chunk[0]}"
        )

        for step in range(args.actions_per_burst):
            current_state = obs_adapter.read(robot).state
            action_dict, clamped = act_adapter.to_action_dict(chunk[step], current_state)
            tag = " (CLAMPED)" if clamped else ""
            if args.enable_robot and not args.dry_run:
                print(f"[burst {burst_id} step {step}]{tag} sending {action_dict}")
                robot.send_action(action_dict)
            else:
                print(f"[burst {burst_id} step {step}]{tag} [DRY-RUN] would send {action_dict}")
            time.sleep(period)

        if args.settle_time > 0:
            print(f"[burst {burst_id}] settling for {args.settle_time:.2f}s ...\n")
            time.sleep(args.settle_time)


def run_continuous(
    args: argparse.Namespace,
    robot: Any,
    obs_adapter: SO101ObservationAdapter,
    vam: VAMInference,
    act_adapter: SO101ActionAdapter,
) -> None:
    print(
        f"\n[continuous] fps={args.fps}, num_execute_actions={args.num_execute_actions}, "
        f"dry_run={args.dry_run}, enable_robot={args.enable_robot}"
    )
    print("             Ctrl-C to stop.\n")

    torch = _import_torch()
    period = 1.0 / float(args.fps)
    action_buffer: np.ndarray | None = None
    buffer_idx = 0
    tick = 0

    try:
        while True:
            tick += 1
            tick_start = time.time()

            current = obs_adapter.read(robot)
            obs_adapter.push(current)

            if action_buffer is None or buffer_idx >= args.num_execute_actions:
                input_vid, state_BHO = obs_adapter.build_model_input(
                    device=args.device,
                    dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
                )
                t0 = time.time()
                action_buffer = vam.infer(input_vid, state_BHO, args.task)
                infer_ms = (time.time() - t0) * 1000.0
                buffer_idx = 0
                print(f"[continuous tick={tick}] replanned in {infer_ms:.0f} ms")

            action_dict, clamped = act_adapter.to_action_dict(
                action_buffer[buffer_idx], current.state
            )
            tag = " (CLAMPED)" if clamped else ""
            if args.enable_robot and not args.dry_run:
                robot.send_action(action_dict)
                if clamped or tick % max(1, args.fps) == 0:
                    print(f"[continuous tick={tick}]{tag} {action_dict}")
            else:
                print(f"[continuous tick={tick}]{tag} [DRY-RUN] {action_dict}")
            buffer_idx += 1

            elapsed = time.time() - tick_start
            if elapsed < period:
                time.sleep(period - elapsed)
            elif elapsed > 2.0 * period:
                print(f"[continuous] slipped tick by {(elapsed - period) * 1000:.0f} ms")
    except KeyboardInterrupt:
        print("\n[continuous] stopping on Ctrl-C.")


# --------------------------------------------------------------------------- #
# CLI parsing
# --------------------------------------------------------------------------- #

def _parse_csv_floats(value: str, expected_len: int, name: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    try:
        floats = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--{name} contains non-float entry: {value!r}") from exc
    if len(floats) != expected_len:
        raise argparse.ArgumentTypeError(
            f"--{name} must have {expected_len} entries; got {len(floats)} ({value!r})"
        )
    return floats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths -------------------------------------------------------------
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Action decoder .pt (alias for --action-model-path).")
    p.add_argument("--action-model-path", type=str, default=None,
                   help="Action decoder .pt checkpoint.")
    p.add_argument("--video-model-path", type=str, required=True,
                   help="Video backbone .pt (e.g. v2w_push_that_thing.pt).")
    p.add_argument("--dataset-statistics-path", type=str, required=True,
                   help="stats.json produced by scripts/export_dataset_stats.py.")
    p.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT,
                   help="Hydra experiment name for the VAM config.")

    # Robot connection --------------------------------------------------------
    p.add_argument("--port", type=str, default="/dev/ttyACM0",
                   help="Serial port for SO-101 follower.")
    p.add_argument("--robot-id", type=str, default="so101_deploy",
                   help="LeRobot robot id; selects the calibration file.")
    p.add_argument("--camera-index", type=int, default=0,
                   help="OpenCV camera index (e.g. 0 for /dev/video0).")
    p.add_argument("--camera-fps", type=int, default=30,
                   help="Camera capture frame rate.")
    p.add_argument("--image-key", type=str, default="front",
                   help="Key for the camera image in robot.get_observation().")
    p.add_argument("--joint-order", type=str, default=",".join(DEFAULT_JOINT_ORDER),
                   help="Comma-separated joint names (must match training order).")

    # Task --------------------------------------------------------------------
    p.add_argument("--task", type=str, required=True,
                   help="Natural-language task description (must match training).")

    # Inference ---------------------------------------------------------------
    p.add_argument("--device", type=str, default="cuda",
                   help="Torch device. CPU is unsupported by mimic-video flash-attn.")
    p.add_argument("--fps", type=int, default=5,
                   help="Control rate at which the script issues actions.")
    p.add_argument("--action-mode", type=str, default="absolute",
                   choices=["absolute", "delta"],
                   help="Interpret predictions as absolute targets (SO-101 default) or deltas.")
    p.add_argument("--img-horizon", type=int, default=OBS_HORIZON_IMG_DEFAULT)
    p.add_argument("--lowdim-horizon", type=int, default=OBS_HORIZON_LOWDIM_DEFAULT)
    p.add_argument("--stop-video-denoising-step", type=int, default=25,
                   help="Number of video denoising steps before extracting xattn features.")
    p.add_argument("--num-sampling-step", type=int, default=35,
                   help="Total flow-matching sampling steps.")
    p.add_argument("--use-cuda-graphs", dest="use_cuda_graphs",
                   action="store_true", default=True)
    p.add_argument("--no-cuda-graphs", dest="use_cuda_graphs", action="store_false")
    p.add_argument("--seed", type=int, default=0)

    # Execution ---------------------------------------------------------------
    p.add_argument("--num-execute-actions", type=int, default=8,
                   help="Actions per replan in continuous mode.")
    p.add_argument("--actions-per-burst", type=int, default=2,
                   help="Actions per burst in burst mode.")
    p.add_argument("--settle-time", type=float, default=1.0,
                   help="Seconds of hold between bursts.")
    p.add_argument("--manual-trigger", dest="manual_trigger",
                   action="store_true", default=True)
    p.add_argument("--no-manual-trigger", dest="manual_trigger", action="store_false")
    p.add_argument("--continuous", action="store_true", default=False,
                   help="Use continuous mode instead of burst.")

    # Safety ------------------------------------------------------------------
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                   help="(Only takes effect together with --enable-robot.)")
    p.add_argument("--enable-robot", action="store_true", default=False,
                   help="Required for robot.send_action() to actually move the arm.")
    p.add_argument("--max-delta", type=str,
                   default=",".join(str(x) for x in DEFAULT_MAX_DELTA),
                   help="Per-joint max delta per tick (degrees x5 + percent for gripper).")

    # Optional renormalization ------------------------------------------------
    p.add_argument("--action-mean-path", type=str, default=None,
                   help="Optional .npy with shape (6,) — subtract from predictions before clamping.")
    p.add_argument("--action-std-path", type=str, default=None,
                   help="Optional .npy with shape (6,) — divide predictions by this after subtracting mean.")

    args = p.parse_args()

    # --- Reconcile aliases / validations -------------------------------------
    if args.action_model_path is None and args.checkpoint is None:
        p.error("--action-model-path or --checkpoint is required.")
    if args.action_model_path is None:
        args.action_model_path = args.checkpoint

    args.joint_order = [j.strip() for j in args.joint_order.split(",") if j.strip()]
    if len(args.joint_order) != STATE_DIM:
        p.error(f"--joint-order must have {STATE_DIM} entries; got {len(args.joint_order)}.")
    args.max_delta = _parse_csv_floats(args.max_delta, ACTION_DIM, "max-delta")

    if args.action_mode == "delta":
        print(
            "WARNING: --action-mode=delta is unusual for the SO-101 model "
            "(trained with target_repr=ABSOLUTE). Use only if you know what you're doing."
        )

    if args.action_mean_path is not None:
        args.action_mean = np.load(args.action_mean_path).astype(np.float32).reshape(-1)
        if args.action_mean.shape != (ACTION_DIM,):
            p.error(
                f"--action-mean-path must hold a ({ACTION_DIM},) array; got {args.action_mean.shape}."
            )
    else:
        args.action_mean = None
    if args.action_std_path is not None:
        args.action_std = np.load(args.action_std_path).astype(np.float32).reshape(-1)
        if args.action_std.shape != (ACTION_DIM,):
            p.error(
                f"--action-std-path must hold a ({ACTION_DIM},) array; got {args.action_std.shape}."
            )
    else:
        args.action_std = None
    if (args.action_mean is None) != (args.action_std is None):
        p.error("Provide both --action-mean-path and --action-std-path, or neither.")

    # Safety: never move the robot when --enable-robot is missing.
    if not args.enable_robot:
        args.dry_run = True

    if args.fps <= 0:
        p.error("--fps must be > 0.")
    if args.actions_per_burst < 1:
        p.error("--actions-per-burst must be >= 1.")
    if args.num_execute_actions < 1:
        p.error("--num-execute-actions must be >= 1.")
    if args.img_horizon < 1 or args.lowdim_horizon < 1:
        p.error("--img-horizon and --lowdim-horizon must be >= 1.")
    if args.device != "cuda":
        print(
            f"WARNING: --device={args.device!r}. mimic-video inference requires CUDA "
            "in practice (flash-attn, transformer-engine). Expect failures on CPU."
        )

    return args


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    print(
        f"[deploy_so101_vam] dry_run={args.dry_run}, enable_robot={args.enable_robot}, "
        f"continuous={args.continuous}, task={args.task!r}"
    )

    pipeline = load_vam_pipeline(args)
    vam = VAMInference(
        pipeline=pipeline,
        num_sampling_step=args.num_sampling_step,
        stop_after_step=args.stop_video_denoising_step,
        use_cuda_graphs=args.use_cuda_graphs,
        seed=args.seed,
    )

    obs_adapter = SO101ObservationAdapter(
        joint_order=args.joint_order,
        image_key=args.image_key,
        img_horizon=args.img_horizon,
        lowdim_horizon=args.lowdim_horizon,
    )
    act_adapter = SO101ActionAdapter(
        joint_order=args.joint_order,
        max_delta=np.asarray(args.max_delta, dtype=np.float32),
        mode=args.action_mode,
        action_mean=args.action_mean,
        action_std=args.action_std,
    )

    robot = make_robot(args)
    robot.connect()
    try:
        try:
            sample_obs = robot.get_observation()
            print(f"[deploy] obs keys after connect: {sorted(sample_obs.keys())}")
        except Exception as exc:
            print(f"[deploy] WARNING: could not read initial observation: {exc}")

        if args.continuous:
            run_continuous(args, robot, obs_adapter, vam, act_adapter)
        else:
            run_burst(args, robot, obs_adapter, vam, act_adapter)
    finally:
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"[deploy] WARNING during disconnect: {exc}")


if __name__ == "__main__":
    main()
