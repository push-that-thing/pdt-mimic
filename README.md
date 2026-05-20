# robot_learning_wm — SO-101 deployment

Run a trained MIMIC video-action model on a real LeRobot **SO-101** arm for the
*push-that-thing* tasks.

## Run

```bash
export HF_TOKEN=hf_...                 # access to the private push-that-thing/* repos

bash setup.sh                          # once per machine (~5 min)
SMOKE=1 bash run_eval_task1.sh         # verify without the arm (load + one inference)
LIVE=1 ROBOT_PORT=/dev/ttyACM0 ROBOT_ID=my_so101 CAMERA_IDX=0 bash run_eval_task1.sh
```

- `run_eval_task2.sh` is identical for task 2.
- Without `LIVE=1` it's a **dry-run** — prints the deploy command, moves nothing.
- `setup.sh` runs automatically on the first `run_eval` if the env isn't built.

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
