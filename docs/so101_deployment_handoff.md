# SO-101 VAM Deployment — Handoff

This document is the complete state-of-the-world for the SO-101 deployment
work done on the `so101_deployment` branch of both `pdt-mimic` and its
`mimic-video` submodule. Pick this up if you want to test, deploy, modify,
or hand off the work to someone else.

The implementation goal: take a trained MimicVideo / VAM checkpoint for
the SO-101 push-that-thing task and run it on a real LeRobot SO-101 follower
arm, without retraining, without a LeRobot policy plugin, and without
`lerobot-rollout`. A standalone deploy script only.

---

## 1. TL;DR — Current state

| Item | State |
|---|---|
| Branch in `pdt-mimic` | `so101_deployment`, **0 commits ahead of `jessie`** |
| Branch in `mimic-video` submodule | `so101_deployment`, **0 commits ahead of `jessie`**, **no edits** |
| `main` and `jessie` in either repo | Untouched |
| Files written | 5 new files in `pdt-mimic`, all in working tree (untracked) |
| Implementation | Done — passes parse, lint, `--help`, `bash -n` |
| Tested end-to-end on real hardware | **NOT YET** — requires CUDA machine + SO-101 arm |
| Pushed to origin | No |
| Committed | No |

The work is ready to be pushed to a CUDA machine and tested. Nothing has
been validated on real hardware yet.

---

## 2. Files written (all under `pdt-mimic/`)

| Path | Lines | Purpose |
|---|---|---|
| `requirements-deploy.txt` | 17 | `feetech-servo-sdk` + `pyserial`. Layered on top of the existing mimic-video Python 3.10 + cu126 venv. |
| `scripts/deploy_setup.sh` | ~90 | Activates the mimic-video venv, installs the deploy deps, reinstalls lerobot with the `[feetech]` extra in `--no-deps` mode, runs an import sanity check covering everything the deploy script needs. |
| `scripts/export_dataset_stats.py` | 177 | Reads `observation.state` and `action` from `<dataset>/data/chunk-*/file-*.parquet` via pyarrow, runs them through the trainer's `array_to_stats`, writes a ~10 KB `stats.json` with the exact dict layout `World2ActionPipeline.normalizer.build_from_stats(...)` expects. |
| `scripts/deploy_so101_vam.py` | 861 | The deployment script. Loads the VAM pipeline, optionally connects to a LeRobot 0.5 `SO101Follower` (or a `DummyRobot`), runs `Video2World2ActionPipeline`, clamps every action to `current_state ± max_delta`, and only sends to hardware when `--enable-robot` is set. Manual burst (default) and continuous modes. |
| `docs/so101_deployment.md` | 244 | Step-by-step runbook (you can also read this one as the operator guide). |
| `docs/so101_deployment_handoff.md` | this file | Project handoff. |

`mimic-video/` submodule: 0 edits.

---

## 3. Architecture

```
+----------------------------------+         +-------------------------+
|  scripts/deploy_so101_vam.py     |         |  mimic-video (UNCHANGED)|
|  (pdt-mimic, ~861 lines)         |         |  ─────────────────────  |
|                                  |         |  Video2World2Action     |
|  - CLI parsing + safety gates    |         |  Pipeline.__call__      |
|  - Robot lifecycle (try/finally) |         |  → tensor (1, 90, 6)    |
|  - Burst / continuous loop       |         +-------------------------+
|                                  |                    ^
|  +-------------------------------+                    |
|  | SO101ObservationAdapter       |                    |
|  |   robot.get_observation()     |─── image (1,3,5,480,640) bfloat16
|  |   resize, [-1,1], history     |     state (1,1,6) bfloat16
|  +-------------------------------+                    |
|                                  |                    |
|  +-------------------------------+                    |
|  | VAMInferenceWrapper           |────────────────────/
|  |   pipeline(...)               |
|  +-------------------------------+
|                                  |
|  +-------------------------------+
|  | SO101ActionAdapter            |◄─── pred (90,6) absolute joint targets
|  |   target = pred (absolute)    |
|  |   clamp to current ± max_delta|
|  |   {<joint>.pos: float, ...}   |
|  +-------------------------------+
|                                  |
|  +-------------------------------+
|  | Safety gate / DummyRobot      |
|  |  - --enable-robot OFF: dummy  |
|  |  - --dry-run forced if !robot |
|  +-------------------------------+
+----------------------------------+
       │
       │  (only when --enable-robot)
       ▼
   robot.send_action({joint.pos: degrees, ..., gripper.pos: percent})
```

Three pieces:
1. **Observation adapter** — flat LeRobot dict → model tensors.
2. **VAM wrapper** — thin call into the unchanged MimicVideo pipeline.
3. **Action adapter** — model output → `send_action()` dict + safety clamp.

---

## 4. Confirmed facts (from the user's repo + dataset)

These were verified by reading actual code/data, not assumed:

| Fact | Source |
|---|---|
| SO-101 model has `in_channels=out_channels=6`, `max_horizon=91`, smaller arch than libero/bridge (`model_channels=256`, `num_blocks=7`). | `mimic-video/model/cosmos_predict2/configs/defaults/world2action_pipe.py` "so101" net |
| Default experiment for the SO-101 chain run is `w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1`. | `mimic-video/model/scripts/evaluate_action_decoder_checkpoint.py` DEFAULT_EXPERIMENT |
| `policy_io/lerobot.yaml` sets `target_repr: ABSOLUTE` for both state and action; both with VARIANCE normalization. | `mimic-video/model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml` |
| Action chunk: 90 actions at 30 Hz, `action_shift_s=0.2` (first action 0.2 s after observation). | Same yaml |
| Observation image: 5 frames at 5 Hz, resized to 480×640, normalized `2x − 1` → `[-1, 1]`. | `mimic-video/model/cosmos_predict2/data/action/dataset_lerobot.py` `_process_images` |
| `World2ActionPipeline.__call__` returns un-normalized actions (denormalization happens inside the pipeline). | `mimic-video/model/cosmos_predict2/pipelines/world2action.py` line 240 |
| Joint order in the dataset: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. | `dataset/task_1_v3/meta/info.json` `features.observation.state.names` |
| Dataset is 204 episodes, 88193 frames, 30 fps, single 480×640 AV1 camera at `observation.images.front`. | `dataset/task_1_v3/meta/info.json` |
| Dataset recorded with `use_degrees=True` (lerobot 0.5 default). | `dataset/task_1_v3/meta/stats.json`: shoulder_lift min = −102.5°, wrist_roll max = +163.0° — impossible in `RANGE_M100_100`. |
| Gripper has effective range ~[1.1, 7.4] in `RANGE_0_100` mode (barely opens). | Same `meta/stats.json`: gripper min=1.08, max=7.41, std=0.28. |
| LeRobot 0.5 SO-101 import: `from lerobot.robots.so_follower.so_follower import SO101Follower` (aliased; class is `SOFollower`). Config from `lerobot.robots.so_follower.config_so_follower.SO101FollowerConfig`. Camera config from `lerobot.cameras.opencv.OpenCVCameraConfig`. | lerobot 0.5 source on GitHub |
| `use_degrees=True` is the **default** in lerobot 0.5 `SOFollowerConfig`. | Same source |
| Gripper is **always** `MotorNormMode.RANGE_0_100` regardless of `use_degrees`. | Same source |

---

## 5. Assumptions (things assumed to be true that you should verify before live deploy)

| Assumption | Why I assumed it | Verification step |
|---|---|---|
| The deploy machine has a CUDA GPU compatible with cu126 + PyTorch 2.6.0. | Whole stack pinned to cu126 in `mimic-video/model/pyproject.toml`. | `nvidia-smi`, `python -c "import torch; print(torch.cuda.is_available())"` |
| The CUDA machine can install `feetech-servo-sdk` and access the SO-101 over a USB serial port. | Stock lerobot 0.5 SO-101 requirement. | `bash scripts/deploy_setup.sh` import sanity check |
| `.brev/setup.sh` (which the README points to) is the canonical way to create the venv on this hardware. It works on any Linux + CUDA box, not just on Brev. | It uses `uv sync --extra cu126`, no Brev-specific calls. | Run it; check `mimic-video/model/.venv/bin/python --version` returns 3.10. |
| The SO-101 calibration file used during recording exists on the deploy machine (or has been copied across). | The model output is meaningless without identical calibration. | After `robot.connect()`, run the one-liner in §11 below; it should report live state within ±5 of the dataset range. |
| The default joint order from `meta/info.json` is the same order the live arm reports keys in. | Both come from lerobot's stock SO-101 config. | After connect, print `obs.keys()` (the deploy script does this automatically once). |
| The image key on the live arm is `front` (the deploy script's default `--image-key`). | Matches your dataset feature `observation.images.front` and lerobot's convention. | First print of `obs.keys()` includes `front`. |
| Native camera resolution is 640×480 at 30 fps. | User-confirmed; also matches `meta/info.json` AV1 stream specs. | If `OpenCVCameraConfig` fails to set these, lerobot raises at `connect()`. |
| The training video backbone is `v2w_push_that_thing.pt`. | `docs/action_decoder_training.md` and `.brev/setup.sh`. | File exists at `mimic-video/model/checkpoints/video_backbone/v2w_push_that_thing.pt` after setup. |
| The chosen action decoder checkpoint is a `.pt` file matching the SO-101 architecture (in=out=6, max_horizon=91). | Trained with experiment `w2a_so101_lerobot_*`. | `World2ActionPipeline.from_config(...)` raises shape errors on mismatch. |
| The task prompt at deploy matches the recorded language string. | Robustness of natural-language conditioning is limited; closer = better. | Read `dataset/task_1_v3/meta/tasks.parquet` and pass that exact string as `--task`. |

---

## 6. Critical things to remember at deploy time

1. **`--enable-robot` is the kill switch.** Without it the script ALWAYS uses a `DummyRobot` and never opens hardware. The robot will not move under any circumstances unless this flag is present.
2. **Default per-tick `max_delta` is `[0.04°, 0.04°, 0.04°, 0.04°, 0.06°, 0.10%]`.** Extremely conservative. Increase only when bursts look correct.
3. **The first inference is slow** (CUDA-graph compile, ~5–15 s). Subsequent calls ~250 ms. Pass `--no-cuda-graphs` if you want a slow-but-stable behavior.
4. **Action mode = absolute.** Do not pass `--action-mode=delta` for the SO-101 model.
5. **The script wraps the loop in `try/finally`.** Ctrl-C still triggers `robot.disconnect()`.
6. **`.brev/setup.sh` pins `LEROBOT_REF=v0.4.0` by default.** You want 0.5: `export LEROBOT_REF=v0.5.0` before running it.
7. **Image color order.** Live camera is RGB by default (lerobot 0.5 `OpenCVCameraConfig.color_mode=ColorMode.RGB`); training dataset frames are also RGB (pyav default). No swap needed.
8. **Units.** Joints 0–4 are degrees; gripper is percent (0–100). Both directions: from `get_observation()` AND to `send_action()`. The script does NO unit conversion.

---

## 7. What was deferred / NOT done

Be honest about scope:

- **Real-hardware validation.** Nothing has actually moved an SO-101 arm.
- **A pytest suite.** No automated tests. Manual smoke tests in §9 cover the same surface area.
- **`--read-only` flag** (open hardware but never send). The calibration-check one-liner in §11 covers this use case.
- **Bus-level safety net.** I left `max_relative_target=None` on `SO101FollowerConfig` because the deploy script does its own per-tick clamping. If you'd like belt-and-suspenders, set it too.
- **Multi-camera support.** Single `front` camera only. Adding more is mechanical but not done.
- **Auto-discovery of the serial port.** `--port` is a required-with-default. No `udev` magic.
- **Continuous-mode tuning per-GPU.** Defaults are reasonable for H100/A100; weaker GPUs will see "slipped tick" messages.
- **Pre-flight calibration drift check inside the script.** Has to be done with the one-liner in §11.
- **A LeRobot policy plugin / `lerobot-rollout` integration.** Out of scope per user requirements.
- **Stats hash matching with training.** `mimic-video`'s training stats cache uses a sha256 key over training parameters. Our `export_dataset_stats.py` doesn't try to match that hash — it just produces the same numerical values. If the trainer's stats hash is ever embedded in the checkpoint, we'd need to update.

---

## 8. Next steps (in order)

1. **Push the branch to your remote** (or copy the working tree across).
   ```bash
   git push -u origin so101_deployment
   git -C mimic-video push -u origin so101_deployment   # if you also want this branch on remote
   ```
2. **On the CUDA machine: clone, switch to branch, run setup.** Details in §9 below.
3. **Export `stats.json` once** from your dataset (the same script can run on the laptop only if torch is installed, otherwise run it on the deploy machine after step 2).
4. **Dry-run the deploy script** to verify model loads + inference works end-to-end.
5. **Verify SO-101 calibration matches recording** (one-liner in §11).
6. **First live test with `--actions-per-burst 1`**, in a safe pose, ready to e-stop.
7. **Iterate** — increase burst size, adjust `max_delta`, finally try continuous mode.

When you are happy and want to capture the work in git:

```bash
git add scripts/deploy_so101_vam.py scripts/export_dataset_stats.py scripts/deploy_setup.sh requirements-deploy.txt docs/so101_deployment.md docs/so101_deployment_handoff.md
git commit -m "Add standalone SO-101 VAM deployment script"
# Optionally also commit the mimic-video branch state — it has no changes, just a new branch.
```

(I will not commit unless you tell me to.)

---

## 9. Test plan (tiered, on the CUDA machine)

### Tier 0 — already passed locally
- Python parsing OK on both `.py` files.
- `bash -n scripts/deploy_setup.sh` OK.
- `--help` renders cleanly for both Python scripts.
- No real linter errors.

### Tier 1 — environment bootstrap
```bash
git clone --recurse-submodules -b so101_deployment <your-pdt-mimic-remote>.git
cd pdt-mimic
git -C mimic-video checkout so101_deployment

export HF_TOKEN=hf_...                # required for private push-that-thing/*
export LEROBOT_REF=v0.5.0             # we want 0.5, not the default 0.4.0
bash .brev/setup.sh
bash scripts/deploy_setup.sh
```
**Pass criteria:** the import sanity check in `deploy_setup.sh` ends with `OK` for `torch`, `numpy`, `cv2`, `pyarrow`, `lerobot`, `SO101Follower`, `SO101FollowerConfig`, `OpenCVCameraConfig`, `feetech_servo_sdk`.

### Tier 2 — stats export
```bash
source mimic-video/model/.venv/bin/activate
python scripts/export_dataset_stats.py \
    --dataset-dir dataset/task_1_v3 \
    --output stats.json
```
**Pass criteria:** the script prints
```
Per-dim mean (state) : [-0.59, -17.97, +26.90, +62.66, +6.85, +1.12]   (approximately)
Per-dim mean (action): [-0.52, -19.24, +25.71, +62.53, +6.92, +1.04]   (approximately)
```
These should match the means in `dataset/task_1_v3/meta/stats.json`. If a dim is in [-1, 1] or way off, something is wrong with the parquet column read.

### Tier 3 — dry-run inference (no hardware)
```bash
python scripts/deploy_so101_vam.py \
    --task "<your task string from tasks.parquet>" \
    --checkpoint <path-to-action-decoder.pt> \
    --video-model-path mimic-video/model/checkpoints/video_backbone/v2w_push_that_thing.pt \
    --dataset-statistics-path stats.json
```
**Pass criteria:**
- Both `[VAM] Loading ...` messages succeed.
- `obs keys after connect: [...]` includes `front` and all six `<joint>.pos` (DummyRobot keys).
- Pressing Enter prints `[burst 1] planned 90 actions in <NNN> ms.` (first call slow due to CUDA graph capture; subsequent fast).
- `[DRY-RUN] would send {'shoulder_pan.pos': 0.0, ..., 'gripper.pos': 0.0}` (zeros because DummyRobot returns zero state; clamping pulls every action back to zero — this is expected).

### Tier 4 — verify live calibration (no model, no motion)
Run the one-liner in §11 below. It opens the SO-101, reads one observation, prints joint values, and checks them against `dataset/task_1_v3/meta/stats.json`.

**Pass criteria:** `live observation within (±5) of dataset range: True`.

### Tier 5 — single-action live test
Arm in safe pose. Be ready with the e-stop / kill-switch.
```bash
python scripts/deploy_so101_vam.py \
    --task "<your task>" \
    --checkpoint <...> \
    --video-model-path <...> \
    --dataset-statistics-path stats.json \
    --enable-robot \
    --port /dev/ttyACM0 \
    --robot-id <your-id> \
    --camera-index 0 \
    --actions-per-burst 1
```
Press Enter exactly once.
**Pass criteria:** arm twitches by at most `max_delta` per joint, prints the dict it sent, then holds. No exceptions during `send_action()`.

### Tier 6 — multi-action bursts
Same command, bump `--actions-per-burst` to 2, then 3, then 5. Watch trajectory smoothness. If the arm starts oscillating or jerking → reduce `max_delta` and/or `actions-per-burst`.

### Tier 7 — continuous mode (only after Tier 6 looks great)
```bash
python scripts/deploy_so101_vam.py ... \
    --enable-robot \
    --continuous \
    --num-execute-actions 8
```
Tolerate occasional `slipped tick by ...` messages. If they're constant, lower `--fps` or accept that the GPU is too slow.

---

## 10. CLI reference (deploy script)

```text
Required:
  --task <str>                   Natural-language task description (must match training).
  --video-model-path <path>      Video backbone .pt (v2w_push_that_thing.pt).
  --dataset-statistics-path <path>  stats.json from export_dataset_stats.py.
  --action-model-path <path>     OR --checkpoint <path>   Action decoder .pt.

Robot (only used with --enable-robot):
  --port <str>                   /dev/ttyACM0 (default)
  --robot-id <str>               Calibration namespace, e.g. so101_deploy
  --camera-index <int>           OpenCV index, default 0
  --camera-fps <int>             Default 30
  --image-key <str>              Default "front"
  --joint-order <csv>            Default shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper

Inference:
  --experiment <str>             Default w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1
  --device <str>                 Default cuda
  --fps <int>                    Default 5 (control rate)
  --action-mode {absolute,delta} Default absolute (SO-101)
  --img-horizon <int>            Default 5
  --lowdim-horizon <int>         Default 1
  --stop-video-denoising-step <int>  Default 25
  --num-sampling-step <int>      Default 35
  --use-cuda-graphs / --no-cuda-graphs   Default on
  --seed <int>                   Default 0

Execution:
  --actions-per-burst <int>      Default 2
  --num-execute-actions <int>    Default 8 (continuous mode)
  --settle-time <float>          Default 1.0
  --manual-trigger / --no-manual-trigger   Default on
  --continuous                   Default off (burst mode)

Safety:
  --dry-run / --no-dry-run       Default --dry-run; forced ON if --enable-robot not given
  --enable-robot                 Default off; required for any motion
  --max-delta <csv>              Default 0.04,0.04,0.04,0.04,0.06,0.10

Optional:
  --action-mean-path <path>      .npy (6,) — subtract before clamp (advanced)
  --action-std-path <path>       .npy (6,) — divide after subtract (advanced)
```

---

## 11. Useful one-liners

### Check SO-101 calibration against dataset range
```bash
source mimic-video/model/.venv/bin/activate
PYTHONPATH=mimic-video/model:lerobot/src python - <<'PY'
import json, numpy as np
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="<your-robot-id>",
    use_degrees=True,
    cameras={"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)},
)
robot = SO101Follower(cfg)
robot.connect()
try:
    obs = robot.get_observation()
    joints = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
    state = np.array([obs[f"{j}.pos"] for j in joints])
    print("live state:", state)
    print("image 'front':", obs["front"].shape, obs["front"].dtype)
    with open("dataset/task_1_v3/meta/stats.json") as f:
        s = json.load(f)["observation.state"]
    in_range = np.all((state >= np.array(s["min"]) - 5) & (state <= np.array(s["max"]) + 5))
    print("dataset range min:", s["min"])
    print("dataset range max:", s["max"])
    print("live within (+/-5) of dataset range:", bool(in_range))
finally:
    robot.disconnect()
PY
```

### Get the recorded task prompt
```bash
python - <<'PY'
import pyarrow.parquet as pq
print(pq.read_table("dataset/task_1_v3/meta/tasks.parquet").to_pylist())
PY
```

### See current branch state in both repos
```bash
git status --short --branch
git -C mimic-video status --short --branch
```

---

## 12. Open questions (require user input or live-machine verification)

| # | Question | Why it matters | Default I picked (and where) |
|---|---|---|---|
| 1 | What is the SO-101 serial port on your deploy machine? | Required by `SO101FollowerConfig.port`. | `--port /dev/ttyACM0` |
| 2 | What `robot_id` did you use when you ran `robot.calibrate()` originally? Is the calibration file present on the deploy machine? | The model is calibration-dependent. Without it, OOD. | `--robot-id so101_deploy` (almost certainly NOT yours) |
| 3 | Which USB camera index is the SO-101's front camera? | Wrong index → wrong image. | `--camera-index 0` |
| 4 | Which action-decoder checkpoint do you want to use? | The chain run produced many. Runbook suggests `iter_000001000.pt`. | None — required CLI arg |
| 5 | What exact task string was used during recording? | Better match → better natural-language conditioning. Read from `meta/tasks.parquet`. | None — required CLI arg |
| 6 | What value of `stop_video_denoising_step` works best for your trained checkpoint? Docs say 25–30 deploy-realistic. | Affects action quality. | `--stop-video-denoising-step 25` |
| 7 | Should `stats.json` be committed alongside the checkpoint on HF? | Travels with the model; deploy-day friendly. | Not committed |
| 8 | Do you want `max_relative_target` set on `SO101FollowerConfig` for belt-and-suspenders bus-level clamping? | Extra safety net. | Not set; software clamp only |
| 9 | When stats are recomputed (e.g. dataset grows), should the deploy script auto-detect a mismatch with the trained checkpoint? | Currently it trusts the user. | Not implemented |
| 10 | Do you ever want `--read-only` (open hardware but never send)? | The calibration one-liner covers this for now. | Not implemented |

---

## 13. Quick reference: full deploy command

```bash
source mimic-video/model/.venv/bin/activate

python scripts/deploy_so101_vam.py \
    --task "push the cube from A to B" \
    --checkpoint model/action_decoder/chain/w2a_so101_h100_chain_bsz2_accum8_full204/checkpoints/model/iter_000001000.pt \
    --video-model-path mimic-video/model/checkpoints/video_backbone/v2w_push_that_thing.pt \
    --dataset-statistics-path stats.json \
    --enable-robot \
    --port /dev/ttyACM0 \
    --robot-id <YOUR_ROBOT_ID> \
    --camera-index 0 \
    --actions-per-burst 1 \
    --settle-time 2.0
```

Press Enter for each burst. Type `q` then Enter to quit. Ctrl-C is also handled cleanly.

---

## 14. Where to look if something breaks

| Symptom | Where to look |
|---|---|
| Cannot import lerobot SO-101 classes | `scripts/deploy_setup.sh` end-of-run sanity check; ensure `LEROBOT_REF=v0.5.0` was set before `.brev/setup.sh` |
| `Image dtype is float32, expected uint8` | OpenCV camera config; lerobot 0.5 `OpenCVCameraConfig` should return uint8 RGB by default |
| `Predicted action dim is 7, expected 6` | `--experiment` is wrong; default is correct for SO-101 |
| `(CLAMPED)` on every joint, every burst | Model wants to move further than `--max-delta` allows; increase incrementally |
| `live observation outside dataset range` | Wrong calibration file; re-copy from recording machine or re-calibrate |
| Inference > 1 s on every burst | CUDA graphs disabled or shape changing each call; check `--use-cuda-graphs` |
| KeyError on `<joint>.pos` | Wrong lerobot version or `--joint-order` mismatch |
| OOM at model load | Pin GPU memory (no other processes); check `nvidia-smi` |

Full troubleshooting table in `docs/so101_deployment.md` section 9.

---

## 15. Files NOT to touch on this branch

- Anything under `mimic-video/` — kept pristine on purpose for clean upstream rebase.
- `.brev/setup.sh` — fine to set `LEROBOT_REF` via env, but don't edit the file itself.
- `dataset/`, `model/` — external HF clones, not part of the deploy work.

---

Last updated by the implementation agent on the day this branch was created.
No commits made; everything is in the working tree.
