# SO-101 VAM Deployment Runbook

How to take a trained MimicVideo / VAM SO-101 checkpoint and run it on a real
LeRobot SO-101 follower arm using `scripts/deploy_so101_vam.py`.

The script is intentionally a thin standalone wrapper:
- It reuses MimicVideo's native `Video2World2ActionPipeline` for inference.
- It uses LeRobot 0.5 (`SO101Follower`) **only** for SO-101 hardware I/O.
- It contains zero LIBERO/Bridge code paths.

It does not move the robot unless `--enable-robot` is explicitly passed.

## 0. Branch state

All deployment work lives on the `so101_deployment` branch (PDT). The
`mimic-video` submodule is untouched. `main` and `jessie` are not modified.

```bash
git status --short --branch
git -C mimic-video status --short --branch
```

## 1. Prerequisites

Run the existing training-environment bootstrap once:

```bash
export HF_TOKEN=hf_...
bash .brev/setup.sh
```

This creates `mimic-video/model/.venv` (Python 3.10 + cu126), clones
`lerobot` into `./lerobot`, downloads the text encoder / video backbone, and
warms the T5 cache.

The runbook assumes the deployment machine **also has the SO-101 calibration
file** that was on the arm during dataset recording:

```
~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<robot_id>.json
```

If this file is missing or was made on a different physical arm, joint zero
offsets and the gripper closed reference will drift and the model will be
wrong despite identical unit space.

## 2. Layer in deployment dependencies

```bash
bash scripts/deploy_setup.sh
```

This activates `mimic-video/model/.venv`, installs
`requirements-deploy.txt` (`feetech-servo-sdk`, `pyserial`), reinstalls
lerobot with the `[feetech]` extra in `--no-deps` editable mode, and runs
an import sanity check.

## 3. Compute deployment statistics (once)

The action decoder normalizes its input state and un-normalizes its
predicted actions using statistics derived from the training dataset. These
are **not** baked into the checkpoint .pt file and must be exported once:

```bash
source mimic-video/model/.venv/bin/activate
python scripts/export_dataset_stats.py \
    --dataset-dir dataset/task_1_v3 \
    --output stats.json
```

The script reads `<dataset-dir>/data/chunk-*/file-*.parquet` directly via
pyarrow (no lerobot dataset adapter is required), calls the same
`array_to_stats` helper the trainer used, and writes a ~10 KB JSON with
both `obs/lowdim_concat` and `action/lowdim_concat` statistics.

Tip: keep this `stats.json` alongside the checkpoint .pt — they are a
single deployable unit.

## 4. Dry-run (no hardware, no movement)

By default the script uses a `DummyRobot`: nothing is opened, nothing
moves. This is the right place to confirm the model loads, the pipeline
runs, and the inferred actions look sane.

```bash
source mimic-video/model/.venv/bin/activate
python scripts/deploy_so101_vam.py \
    --task "push the cube" \
    --checkpoint model/action_decoder/chain/w2a_so101_h100_chain_bsz2_accum8_full204/checkpoints/model/iter_000001000.pt \
    --video-model-path mimic-video/model/checkpoints/video_backbone/v2w_push_that_thing.pt \
    --dataset-statistics-path stats.json
```

Expected console output (truncated):

```
[VAM] Loading video backbone from ...
[VAM] Loading action decoder from ...
[VAM] Loading dataset statistics from stats.json ...
[deploy] --enable-robot NOT set: using DummyRobot (no hardware will be touched).
[deploy] obs keys after connect: ['elbow_flex.pos', 'front', ...]

[burst mode] fps=5, actions_per_burst=2, settle=1.00s, dry_run=True, enable_robot=False
[burst] Press Enter to plan a burst. Type 'q' + Enter to quit.

[burst] >>
[burst] First observation keys: [...]
[burst 1] planned 90 actions in ~250 ms. first-action target = [...]
[burst 1 step 0] [DRY-RUN] would send {'shoulder_pan.pos': 0.0, ..., 'gripper.pos': 0.0}
[burst 1 step 1] [DRY-RUN] would send {'shoulder_pan.pos': 0.0, ..., 'gripper.pos': 0.0}
[burst 1] settling for 1.00s ...
```

If you ever see `(CLAMPED)`, the safety clamp triggered for that step.

## 5. Drive the real arm

Only when step 4 looks clean. Start with the most conservative settings:

```bash
python scripts/deploy_so101_vam.py \
    --task "push the cube" \
    --checkpoint .../iter_000001000.pt \
    --video-model-path .../v2w_push_that_thing.pt \
    --dataset-statistics-path stats.json \
    --enable-robot \
    --port /dev/ttyACM0 \
    --robot-id my_so101 \
    --camera-index 0 \
    --actions-per-burst 1
```

Press Enter to plan + send one action. Inspect what the arm does. Re-press
Enter for the next burst. Bump `--actions-per-burst` to 2, then 3, only as
each lower setting feels safe.

The default per-joint per-tick clamp is intentionally tiny:

```text
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]  ±0.04°  ±0.04°  ±0.04°  ±0.04°  ±0.06°
gripper                                                            ±0.10%
```

Tighten with `--max-delta` once you trust the model.

## 6. Continuous mode (later)

After burst mode is reliable for many bursts in a row:

```bash
python scripts/deploy_so101_vam.py \
    ... \
    --enable-robot \
    --continuous \
    --num-execute-actions 8
```

Inference takes ~250 ms per replan on H100; the loop will print
`slipped tick by ...` when it can't keep up with `--fps`. Replanning every
8 actions at 5 Hz (= 1.6 s control window) is a reasonable starting point.

## 7. Safety summary

| Knob | Default | Effect |
|---|---|---|
| `--enable-robot` | OFF | The ONLY flag that allows `send_action()`. Without it, all actions are logged in `[DRY-RUN]` form. |
| `--dry-run` | ON | Forced ON when `--enable-robot` is missing. With `--enable-robot --no-dry-run`, the script sends actions. |
| `--manual-trigger` | ON | Wait for Enter between bursts. |
| `--max-delta` | `0.04,0.04,0.04,0.04,0.06,0.10` | Per-joint per-tick clamp. Units = degrees x5 + percent for gripper. |
| `--fps` | 5 | Control rate. |
| `--actions-per-burst` | 2 | How many actions to send before settling. |
| `--settle-time` | 1.0 s | Hold time after each burst. |
| `--stop-video-denoising-step` | 25 | Matches the runbook's "deploy-realistic range". |
| `--num-sampling-step` | 35 | Same as offline eval. |

The script wraps the entire control loop in `try/finally` so the robot is
always disconnected on Ctrl-C or exception.

## 8. Verify calibration before trusting a session

Quick standalone check (no model, no inference): connect, read one
observation, compare to the dataset range from `meta/stats.json`.

```bash
source mimic-video/model/.venv/bin/activate
PYTHONPATH=mimic-video/model:lerobot/src python - <<'PY'
import json, numpy as np
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="my_so101",
    use_degrees=True,
    cameras={"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)},
)
robot = SO101Follower(cfg)
robot.connect()
try:
    obs = robot.get_observation()
    state = np.array([obs[f"{j}.pos"] for j in
        ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")])
    print("live state:", state)
    with open("dataset/task_1_v3/meta/stats.json") as f:
        s = json.load(f)["observation.state"]
    in_range = np.all((state >= np.array(s["min"]) - 5) & (state <= np.array(s["max"]) + 5))
    print("dataset state range:", s["min"], "to", s["max"])
    print("live observation within (±5) of dataset range:", bool(in_range))
finally:
    robot.disconnect()
PY
```

If `in_range` is False, do not trust the model output — re-calibrate first.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'shoulder_pan.pos'` | Wrong lerobot version | Confirm lerobot 0.5 with `from lerobot.robots.so_follower.so_follower import SO101Follower`. Rerun `bash scripts/deploy_setup.sh`. |
| `ImportError: feetech_servo_sdk` | Deployment deps not installed | `bash scripts/deploy_setup.sh` |
| `Image dtype is float32; expected uint8` | Camera config wrong / wrong backend | Check `OpenCVCameraConfig(width=640, height=480, fps=30)`. |
| `Predicted action dim is 7, expected 6` | Wrong experiment | Verify `--experiment` ends with `so101_lerobot`. |
| `(CLAMPED)` on every step | `max_delta` too tight, or first-action delta too large | Bump `--max-delta`, or check that the live state is within the dataset's range. |
| `live observation outside dataset range` | Calibration drift | Re-run lerobot calibration with the same `robot_id`. |
| Inference > 1 s | First call CUDA-graph compile | Should be ~250 ms on subsequent calls. `--no-cuda-graphs` disables. |
| Cannot find experiment config | mimic-video import path | Make sure you are in the repo root; the script auto-bootstraps `mimic-video/model` and `lerobot/src` onto `sys.path`. |

## 10. File map

```
pdt-mimic/
├── requirements-deploy.txt          # feetech-servo-sdk + pyserial
├── scripts/
│   ├── deploy_setup.sh              # layer deploy deps over mimic-video venv
│   ├── export_dataset_stats.py      # parquet -> stats.json (one-shot)
│   └── deploy_so101_vam.py          # the deploy script
└── docs/
    └── so101_deployment.md          # this file
```

No files in the `mimic-video/` submodule were modified.
