# robot_learning_wm — SO-101 Video-Action Model deployment

Deploy a trained MIMIC video-action model (VAM) on a real LeRobot **SO-101**
follower arm for the *push-that-thing* tasks. The video backbone + action
decoder run through MIMIC's `Video2World2ActionPipeline`, wrapped as a LeRobot
bring-your-own-policy (`mimic_video`) and driven by `lerobot-record`.

```
robot_learning_wm/
├── run_eval_task1.sh            # one-command setup + deploy, task 1
├── run_eval_task2.sh            # one-command setup + deploy, task 2
├── requirements-deploy.txt      # SO-101 hardware deps (feetech, pyserial)
├── .brev/setup.sh               # env bootstrap (venv, lerobot, base downloads)
├── lerobot_policy_mimic_video/  # the BYOP policy plugin
├── scripts/
│   ├── _run_eval_common.sh      # shared driver for the run_eval_* scripts
│   ├── deploy_so101.sh          # runs lerobot-record with the mimic_video policy
│   ├── mimic_env.sh             # activates the venv + CUDA libs
│   ├── lerobot_record_mimic.py  # registers the plugin, calls lerobot-record
│   ├── check_camera.py          # live cam vs training-frame sanity check
│   ├── goto_start_pose.py       # drive the arm to the training start pose
│   └── smoke_load.py            # load the pipeline + one inference (no robot)
├── lerobot/                     # submodule, pinned v0.4.0 (hardware I/O)
└── mimic-video/                 # submodule (model code + configs)
```

## Prerequisites

- A CUDA GPU box (Linux/x86_64; the venv lockfile is linux-only).
- A LeRobot **SO-101 follower** arm + a front camera, for live runs.
- The **calibration JSON** for *that* arm, from when the training data was
  recorded, at:
  ```
  ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<robot_id>.json
  ```
  A wrong/missing file makes joint zeros drift and the policy mis-behaves.
- `HF_TOKEN` with access to the private `push-that-thing/*` repos.

## Quickstart

```bash
export HF_TOKEN=hf_...

# Dry-run: sets up the env, downloads checkpoints, prints the deploy command.
# Nothing moves.
bash run_eval_task1.sh

# Drive the arm (only after a clean dry-run + calibration check):
LIVE=1 ROBOT_PORT=/dev/ttyACM0 ROBOT_ID=my_so101 CAMERA_IDX=0 bash run_eval_task1.sh
```

Same for task 2 via `run_eval_task2.sh`.

The first invocation runs `.brev/setup.sh` (creates the cu126 venv, checks out
LeRobot v0.4.0, downloads the text encoder + tokenizer + video backbone, warms
the T5 prompt-embedding cache), installs the `mimic_video` plugin + hardware
deps, then downloads that task's action decoder and dataset-statistics cache.
Subsequent runs skip anything already present.

## What each run_eval downloads and where

| Artifact | Source | Local path |
|---|---|---|
| Video backbone (fused) | `push-that-thing/cosmos-task1-task2` | `mimic-video/model/checkpoints/video_backbone/<name>.pt` |
| Text encoder + tokenizer | `jonpai/mimic-video` | `mimic-video/model/checkpoints/{text_encoder,video_backbone/tokenizer}/` |
| T5 prompt-embedding cache | `push-that-thing/task_{1,2}` | `~/.cache/mimic_lerobot_t5/.t5_embeddings/` |
| Action decoder | `push-that-thing/model` | `mimic-video/model/checkpoints/vam/so101/<exp>/checkpoints/model/iter_*.pt` |
| Dataset statistics | `push-that-thing/model` | `~/.cache/mimic_lerobot_stats/<sha256>` |

Per-task knobs (prompt, action experiment, iter, stats hash, experiment name,
video backbone) are env-overridable at the top of each `run_eval_task*.sh`.

## Safety

- **Dry-run is the default.** `run_eval_task*.sh` only drives the arm with
  `LIVE=1`. The dry-run prints the exact `lerobot-record` command without
  executing it.
- Before a live run, verify framing with `python scripts/check_camera.py` and
  move to the start pose with `python scripts/goto_start_pose.py` (supports
  `--dry-run`).
- Confirm the live joint state is within the dataset range before trusting the
  policy (see the calibration note above).

## Notes / gotchas

- **Submodule must be pushed.** Deployment uses a merged `mimic-video` commit
  (precompute-latents + so101 deployment code). Push it to
  `push-that-thing/mimic-video` and update the pin before cloning on the robot
  box, or `git submodule update` will fail to fetch it.
- **`EXPERIMENT_NAME`** must match the hydra experiment the action decoder was
  trained under (it selects the so101 net + `policy_io`). Override it in the
  run_eval script if your training used a different registered name.
- **Task-2 prompt** must match the instruction string stored in
  `push-that-thing/task_2`'s T5 cache (looked up by content hash when
  `use_text_encoder=false`). Confirm it, or run with `USE_TEXT_ENCODER=true`
  (needs a large GPU to hold T5-XXL).
