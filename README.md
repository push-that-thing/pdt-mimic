# robot_learning_wm — SO-101 deployment

Run a trained MIMIC video-action model on a real LeRobot **SO-101** arm for the
*push-that-thing* tasks.

## Run

```bash
export HF_TOKEN=hf_...                 # OPTIONAL — only for private HF repos; public ones download without it

bash setup.sh                          # once per machine (~5 min)
SMOKE=1 bash run_eval_task1.sh         # verify without the arm (load + one inference)
LIVE=1 ROBOT_PORT=/dev/ttyACM0 ROBOT_ID=my_so101 CAMERA_IDX=0 bash run_eval_task1.sh
```

The `push-that-thing/*` and `jonpai/mimic-video` HF assets are **public**, so no
`HF_TOKEN` is needed — `setup.sh` downloads them anonymously. Set the token only
if you repoint `HF_MODEL_REPO` (or the T5 caches) at a private repo.

- `run_eval_task2.sh` is identical for task 2.
- Without `LIVE=1` it's a **dry-run** — prints the deploy command, moves nothing.
- `setup.sh` runs automatically on the first `run_eval` if the env isn't built.

## Git / submodule setup

This project depends on two git submodules — `mimic-video` (the model code) and
`lerobot` (hardware I/O, pinned to `v0.4.0`). How you populate them depends on
how you got the project:

**Received as the submission zip** — the submodules are already **vendored**
inside `mimic-video/` and `lerobot/` as plain files. Nothing git-related is
required: `setup.sh` detects the vendored sources and skips submodule fetching.
Just run `bash setup.sh`.

**Cloned from git** — populate the submodules once:

```bash
git clone --recurse-submodules <repo-url>
# …or, in an existing clone:
git submodule update --init mimic-video lerobot
```

**Building the submission zip** (maintainer) — populate the submodules first so
their source travels as plain files, then archive without the heavy rebuilt venv
and git history:

```bash
git submodule update --init mimic-video lerobot   # fills mimic-video/ and lerobot/
zip -r robot_learning_wm.zip . -x '*/.venv/*' -x '.git/*'
```

## Configure inference

Prefix any run with these env vars (defaults in parens):

| Var | Effect |
|---|---|
| `NUM_SAMPLING_STEP` | video backbone denoise steps (35) |
| `STOP_AFTER_STEP` | stop video denoise to extract features (30) |
| `ACTION_DENOISE_STEPS` | action-decoder diffusion steps (10) |
| `N_ACTION_STEPS` | actions executed per replan, `LIVE` only (10) |
| `SEED` | RNG seed (0) |

```bash
NUM_SAMPLING_STEP=20 ACTION_DENOISE_STEPS=20 SMOKE=1 bash run_eval_task1.sh
```

## Prerequisites

- Linux/x86_64 CUDA box; for a 5090/Blackwell an R570+ driver.
- For `LIVE=1`: an SO-101 follower + front camera, plus the arm's calibration JSON at
  `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<robot_id>.json`.

## Notes

- `setup.sh` repins torch to cu128 for Blackwell (sm_120); override `TORCH_VERSION` /
  `CUDA_TAG` if a cosmos extension fails to import.
- `EXPERIMENT_NAME` (top of `run_eval_task*.sh`) must match the experiment the action
  decoder was trained under.
- The task-2 prompt must match `push-that-thing/task_2`'s T5 cache entry, or run with
  `USE_TEXT_ENCODER=true`.
