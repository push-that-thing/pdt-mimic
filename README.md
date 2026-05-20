# robot_learning_wm — SO-101 deployment

Run a trained MIMIC video-action model on a real LeRobot **SO-101** arm for the
*push-that-thing* tasks. Two steps: one machine setup, then one command per task.

```bash
export HF_TOKEN=hf_...

bash setup.sh            # once per machine (~5 min): env, cu128 torch, deps, shared models
bash run_eval_task1.sh   # dry-run: downloads the task decoder, prints the deploy command
LIVE=1 ROBOT_PORT=/dev/ttyACM0 ROBOT_ID=my_so101 CAMERA_IDX=0 bash run_eval_task1.sh
```

Task 2 is the same via `run_eval_task2.sh`.

## What `setup.sh` does

1. Installs `uv`, initializes the `mimic-video` + `lerobot` submodules.
2. Creates the venv (`uv sync --extra cu126`) for the cosmos deps.
3. **Repins torch to cu128** (`torch 2.7.0 + torchvision 0.22.0`) — required for
   Blackwell GPUs (RTX 5090, sm_120). Override with `TORCH_VERSION` /
   `TORCHVISION_VERSION` / `CUDA_TAG`.
4. Installs lerobot, the `mimic_video` policy plugin, and SO-101 hardware deps
   (`requirements-deploy.txt`).
5. Downloads the shared video backbone + tokenizer and warms the task-1/task-2
   T5 prompt-embedding caches.

`run_eval_task*.sh` then downloads that task's action decoder + statistics cache
and runs the policy. (If you skip `setup.sh`, the first `run_eval` runs it for you.)

## Files

```
setup.sh                       # one-time machine setup
run_eval_task1.sh / _task2.sh  # per-task deploy entry points
requirements-deploy.txt        # SO-101 hardware deps
lerobot_policy_mimic_video/    # the mimic_video LeRobot policy
scripts/
  deploy_so101.sh              # runs lerobot-record with the policy
  mimic_env.sh                 # activates the venv (paths auto-derived)
  lerobot_record_mimic.py      # registers the plugin, calls lerobot-record
  _run_eval_common.sh          # shared run_eval driver
lerobot/  mimic-video/         # submodules
```

## Prerequisites

- Linux/x86_64 CUDA box. For a 5090/Blackwell you need an R570+ driver.
- A LeRobot SO-101 follower + front camera (for `LIVE=1` runs).
- The arm's **calibration JSON** from when the data was recorded, at
  `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<robot_id>.json`.
- `HF_TOKEN` with access to the private `push-that-thing/*` repos.

## Test without the arm

```bash
SMOKE=1 bash run_eval_task1.sh
```
Loads the pipeline and runs one inference on a fake frame (no robot/camera).
Prints torch/CUDA + GPU arch info first, so a Blackwell/cu128 problem is obvious.

## Safety

`run_eval_task*.sh` is **dry-run by default** — it prints the exact
`lerobot-record` command without moving anything. Add `LIVE=1` to drive the arm.

## Notes

- **Submodule must be reachable.** The `mimic-video` pin is a merge commit
  (`feat/deploy-with-precompute` on `push-that-thing/mimic-video`); it must be
  pushed before cloning elsewhere or `git submodule update` will fail.
- **`EXPERIMENT_NAME`** must match the hydra experiment the action decoder was
  trained under (selects the so101 net + `policy_io`). Override in the run_eval
  script if different.
- **Task-2 prompt** must match the instruction string in `push-that-thing/task_2`'s
  T5 cache (hash-matched when `use_text_encoder=false`). Set it, or run with
  `USE_TEXT_ENCODER=true` (needs a large GPU for T5).
- **cu128 vs cosmos extensions.** The cosmos deps are built for torch 2.6/cu126;
  the cu128 repin in `setup.sh` may need a different `TORCH_VERSION` if a
  compiled extension (e.g. transformer_engine) fails to import.
