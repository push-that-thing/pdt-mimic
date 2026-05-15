# SO-101 Action Decoder Training Runbook

This document captures the exact launch flow for training the SO-101 action
decoder from the fine-tuned Cosmos video backbone, plus the knobs to tune based
on what we observe.

## Goal

Train a deployable SO-101 action decoder that predicts 6D low-dimensional
actions from:

- SO-101 LeRobot dataset: `push-that-thing/task_1`
- Fine-tuned video backbone checkpoint from `push-that-thing/model`
- Precomputed T5 task embeddings from `push-that-thing/task_1`
- W&B logging under `elias-andreoli-eth-z-rich/vam`

The current default video checkpoint is:

```text
push-that-thing/model:checkpoints/model/iter_000000500_fused.pt
```

During setup it is copied to:

```text
mimic-video/model/checkpoints/video_backbone/v2w_push_that_thing.pt
```

### Alternative: Combined task1+task2 LoRA backbone (preferred for new runs)

A LoRA-fine-tuned variant trained on combined `task_1`+`task_2` data lives at:

```text
push-that-thing/cosmos-task1-task2:posttraining/video2world/v2w_push_combined_lora_rank32_lr1.778e-04_bsz32/checkpoints/
  model/iter_000002500.pt        # unfused — needs fusion step before use
```

These checkpoints are **LoRA-only** (separate `lora_A`/`lora_B` adapters), not
fused into the base weights. The action-decoder pipeline expects a fused
checkpoint, so the setup script auto-fuses on download when the requested
filename ends in `_fused.pt`:

1. Looks for `iter_NNN_fused.pt` on HF — not found in the combined repo
2. Falls back to `iter_NNN.pt` (unfused), downloads it
3. Runs `mimic-video/model/scripts/fuse_lora_ckpt.py` to produce
   `iter_NNN_fused.pt` (ALPHA=32 hardcoded, matches the LoRA training)
4. Moves the fused output to the expected video-backbone path
5. Cleans up the unfused source (saves ~4 GB)

To use this backbone, set these env vars before `bash .brev/setup.sh`:

```bash
export HF_MODEL_REPO=push-that-thing/cosmos-task1-task2
export VIDEO_CKPT_FILE=posttraining/video2world/v2w_push_combined_lora_rank32_lr1.778e-04_bsz32/checkpoints/model/iter_000002500_fused.pt
export VIDEO_CKPT_NAME=v2w_push_combined_lora_rank32_lr1.778e-04_bsz32_iter_000002500_fused
```

Setup will download the ~4 GB unfused source, fuse it, and leave the resulting
~3.9 GB `<VIDEO_CKPT_NAME>.pt` in
`mimic-video/model/checkpoints/video_backbone/`. Fusion takes ~30-60 seconds
on H100.

Then for training, pass the same name as `VIDEO_CKPT`:

```bash
VIDEO_CKPT=v2w_push_combined_lora_rank32_lr1.778e-04_bsz32_iter_000002500_fused \
JOB_NAME=w2a_so101_combined_lora32_iter2500 \
bash scripts/train_action_decoder.sh trainer.max_iter=20000
```

**Important:** swapping backbones changes the layer-20 feature distribution
the action decoder is conditioned on. Previously-trained action decoder
weights (e.g., iter_1000 of the `v2w_push_that_thing` chain run) will not
transfer. **A new backbone requires training the action decoder from
scratch** with a fresh `JOB_NAME`.

The previous chain run's stats cache CAN be reused — the normalizer
statistics depend on the dataset only, not the backbone.

## Required Branches

Use the `jessie` branch in both repos:

```bash
git clone --recurse-submodules --branch jessie git@github.com:push-that-thing/pdt-mimic.git
cd pdt-mimic
git submodule update --init --recursive
```

Verify the normalizer fix is present:

```bash
git -C mimic-video log -1 --oneline
```

Expected:

```text
83725eb Preserve self-mapped normalizer groups
```

Do not train from a commit before this fix. Earlier runs were effectively using
raw actions because the self-mapped concat group removed the normalizer params.

## GPU Choice

Two options that have been validated:

```text
1x A100 80GB        cheaper, slower
2x H100 80GB PCIe   faster wall-clock, ~2.5-3.5x A100 with matched effective batch
```

Pick A100 for cheap iteration on smoke tests and short runs. Pick 2x H100 for
the real validation run on the full 204-episode dataset where iteration speed
matters. Effective batch is preserved across the two by adjusting
`GRAD_ACCUM_ITER` (see the launch sections below).

Avoid L40S for the real run because 48GB VRAM is risky for this pipeline.

## One-Time Setup on Brev

Set the Hugging Face token for private dataset/checkpoint access:

```bash
export HF_TOKEN="hf_..."
```

Run setup:

```bash
cd ~/workspace/pdt-mimic
bash .brev/setup.sh
```

Login to W&B:

```bash
wandb login
```

Verify W&B:

```bash
wandb status
```

The repo includes a non-secret W&B config at:

```text
configs/wandb/action_decoder.env
```

It sets:

```bash
WANDB_ENTITY=elias-andreoli-eth-z-rich
WANDB_PROJECT=vam
WANDB_MODE=online
```

Do not commit HF or W&B API tokens.

## Verify the Pipeline Before Training

Check GPU:

```bash
nvidia-smi
```

Check the fine-tuned video checkpoint:

```bash
ls -lh mimic-video/model/checkpoints/video_backbone/v2w_push_that_thing.pt
```

### Confirm Dataset Episode Count

`push-that-thing/task_1` has been growing over time. As of 2026-05-12 it has
**204 episodes**. The dataset wrapper trusts whatever the LeRobot loader
reports for `num_episodes`, so a stale local cache from a previous instance
can silently train on a smaller snapshot.

> Real incident: an earlier A100 run trained on 57 episodes because the
> instance had cached an older snapshot of `task_1` at
> `~/.cache/huggingface/lerobot/push-that-thing/task_1/`. Always confirm the
> count before launching.

```bash
cd ~/workspace/pdt-mimic/mimic-video/model
source .venv/bin/activate
PYTHONPATH=$PWD:../../lerobot/src python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
d = LeRobotDataset("push-that-thing/task_1")
print("num_episodes:", d.num_episodes)
PY
```

Expect `num_episodes: 204` (or higher if the dataset has grown again).

If the number is too low, nuke the cache and re-check:

```bash
rm -rf ~/.cache/huggingface/lerobot/push-that-thing/task_1
```

Check the dataset adapter:

```bash
cd ~/workspace/pdt-mimic/mimic-video/model
source .venv/bin/activate
PYTHONPATH=$PWD:../../lerobot/src python - <<'PY'
from cosmos_predict2.data.action.dataset_lerobot import LeRobotDataset

ds = LeRobotDataset("push-that-thing/task_1", train=True)
print("len", len(ds))
x = ds[0]
for k, v in x.items():
    print(k, v.shape, v.dtype)
PY
```

Expected shapes:

```text
obs/workspace_rgb (3, 5, 480, 640) float32
obs/lowdim_concat (1, 6) float32
obs/language_embedding (1, 512, 1024) float32
action/workspace_rgb (3, 56, 480, 640) float32
action/lowdim_concat (90, 6) float32
```

## Model Parameters

Current SO-101 action decoder parameters (~75M total params):

```text
max_horizon = 91
in_channels = 6
out_channels = 6
model_channels = 512
num_blocks = 12
num_heads = 8
mlp_ratio = 4.0
crossattn_emb_channels = 2048
use_adaln_lora = True
adaln_lora_dim = 128
pair_timestep_feature_rank = 512
atten_backend = flash_attn_no_cp
sac_config = none
```

Defined in `mimic-video/model/cosmos_predict2/configs/defaults/world2action_pipe.py`,
the `"so101"` entry of `ACTION_DECODER_NETS`. Matched by data_configs that
start with `"so101"` (e.g., `so101_lerobot`).

Earlier chain runs (`w2a_so101_h100_chain_bsz2_accum8_full204` on HF) used a
much smaller config (~17M params, `model_channels=256`, `num_blocks=7`,
`mlp_ratio=3`, `use_adaln_lora=False`). Those checkpoints are **not
compatible** with the current 75M architecture — any retraining from a fresh
session starts the action decoder from scratch.

Data horizon:

```text
observation images: 5 frames at 5 Hz
action images: 56 frames at 5 Hz
lowdim actions: 90 steps at 30 Hz
action shift: 0.2 seconds
action dimension: 6
```

Normalization:

```text
obs/lowdim_concat: VARIANCE
action/lowdim_concat: VARIANCE
images: no lowdim normalizer, image values mapped to [-1, 1]
```

The action decoder trains in normalized action space and unnormalizes outputs at
inference.

## Batch Size Terms

Keep:

```bash
BSZ=1
```

This is the registered Hydra experiment batch key. Do not set `BSZ=2` unless a
matching Hydra experiment exists.

Actual training batch is:

```text
effective batch size = NGPU * MICRO_BSZ * GRAD_ACCUM_ITER
```

For one A100:

```text
NGPU=1
MICRO_BSZ=1
GRAD_ACCUM_ITER=16
effective batch size=16
```

`MICRO_BSZ` increases GPU memory use. `GRAD_ACCUM_ITER` increases the number of
micro-steps before each optimizer update and is much safer for memory.

## Smoke Test

Run this before any real training:

```bash
cd ~/workspace/pdt-mimic

WANDB_MODE=offline \
SMOKE=1 \
NGPU=1 \
BSZ=1 \
MICRO_BSZ=1 \
GRAD_ACCUM_ITER=16 \
JOB_NAME=normfix_a100_smoke \
bash scripts/train_action_decoder.sh
```

For a faster memory probe, try batch 2:

```bash
WANDB_MODE=offline \
SMOKE=1 \
NGPU=1 \
BSZ=1 \
MICRO_BSZ=2 \
GRAD_ACCUM_ITER=8 \
JOB_NAME=normfix_a100_bsz2_smoke \
bash scripts/train_action_decoder.sh
```

Expected good signs:

- The video backbone loads from `v2w_push_that_thing.pt`.
- T5 embeddings are loaded from cache or Hugging Face.
- Dataset statistics compute or load.
- `Var_inst[x_0]` is roughly order `1`, not hundreds or thousands.
- The smoke run completes 2 iterations.

If `Var_inst[x_0]` is huge, stop. That means action normalization is not active.

## Real A100 Launch

Use tmux so training survives SSH drops:

```bash
tmux new -s action_decoder
```

Safe A100 run:

```bash
cd ~/workspace/pdt-mimic

NGPU=1 \
BSZ=1 \
MICRO_BSZ=1 \
GRAD_ACCUM_ITER=16 \
SAVE_ITER=250 \
JOB_NAME=w2a_so101_a100_normfix_bsz1_accum16 \
bash scripts/train_action_decoder.sh trainer.max_iter=20000 2>&1 | tee -a /tmp/action_decoder_a100_normfix.log
```

If `MICRO_BSZ=2` passed smoke and memory looks comfortable, use:

```bash
cd ~/workspace/pdt-mimic

NGPU=1 \
BSZ=1 \
MICRO_BSZ=2 \
GRAD_ACCUM_ITER=8 \
SAVE_ITER=250 \
JOB_NAME=w2a_so101_a100_normfix_bsz2_accum8 \
bash scripts/train_action_decoder.sh trainer.max_iter=20000 2>&1 | tee -a /tmp/action_decoder_a100_normfix_bsz2.log
```

Detach tmux:

```text
Ctrl-b, then d
```

Reattach:

```bash
tmux attach -t action_decoder
```

## W&B

Expected W&B project:

```text
https://wandb.ai/elias-andreoli-eth-z-rich/vam
```

Watch:

```text
train/loss
train/Var_inst[x_0]
system GPU utilization
system GPU memory
```

Because `loss_scale=10.0`, the logged loss is 10x the raw MSE.

## Checkpoints

With:

```bash
SAVE_ITER=250
```

checkpoints should appear every 250 iterations.

For the safe A100 run:

```text
mimic-video/model/checkpoints/vam/so101/w2a_so101_a100_normfix_bsz1_accum16/checkpoints/model/
```

For the batch 2 run:

```text
mimic-video/model/checkpoints/vam/so101/w2a_so101_a100_normfix_bsz2_accum8/checkpoints/model/
```

List latest checkpoints:

```bash
find mimic-video/model/checkpoints/vam/so101 -path '*/checkpoints/model/*.pt' | sort | tail
```

### Preserving Checkpoints Across Instances

Brev instances are ephemeral. Anything on the local disk is lost when the
instance is torn down. Push the checkpoints (and any eval artifacts) to the
HF model repo `push-that-thing/model` before spinning the instance down.

```bash
cd ~/workspace/pdt-mimic
export HF_TOKEN="hf_..."

SRC=mimic-video/model/checkpoints/vam/so101/<JOB_NAME>/checkpoints/model
DST=action_decoder/<JOB_NAME>

hf upload push-that-thing/model $SRC/iter_000000250.pt $DST/iter_000000250.pt
hf upload push-that-thing/model $SRC/iter_000000500.pt $DST/iter_000000500.pt

# Optional: also push offline eval metrics + plots
hf upload push-that-thing/model \
  mimic-video/model/eval/action_decoder/<JOB_NAME> \
  $DST/eval --repo-type=model
```

Notes on the CLI:

- Use `hf upload`, not the deprecated `huggingface-cli upload`.
- Each `hf upload` call must fit on a single shell line. If your terminal
  inserts newlines on long pasted lines, set `SRC` / `DST` as variables first
  (as above) so each upload call stays short.
- If a path-in-repo argument gets dropped (visible in the URL printed by
  `hf upload`), the file lands at the repo root. Re-upload to the correct
  destination and delete the misplaced copy:

```bash
hf repo-files delete push-that-thing/model iter_000000250.pt iter_000000500.pt
```

Verify in the browser before terminating the instance:

```text
https://huggingface.co/push-that-thing/model/tree/main/action_decoder/<JOB_NAME>
```

### A100 Baseline Reference

The first A100 run (job `w2a_so101_a100_normfix_bsz2_accum8`) is archived at:

```text
push-that-thing/model:action_decoder/a100_bsz2_accum8/
  iter_000000250.pt
  iter_000000500.pt
  eval/iter_000000250/{metrics.json,plots/}
  eval/iter_000000500/{metrics.json,plots/}
```

Note: this run trained on a stale 57-episode cached snapshot of `task_1`, not
on the full 204-episode dataset. The iter_500 checkpoint scored
`mse_vs_train_mean_ratio = 3.42` — worse than predicting the train-set mean
action. Treat it as a sanity-check baseline; the H100 chain run on the full
dataset is the real reference point.

### H100 Chain Run Session 1 Reference

The first session of the chain run (job
`w2a_so101_h100_chain_bsz2_accum8_full204`) ran for ~1000 iters on 1x H100 PCIe.
Full state is archived at:

```text
push-that-thing/model:action_decoder/chain/w2a_so101_h100_chain_bsz2_accum8_full204/
  checkpoints/
    latest_checkpoint.txt          # tells resume logic which iter to load
    model/iter_000000050.pt ... iter_000001000.pt
    optim/...                      # Adam state, same iter set
    scheduler/...
    trainer/...
  eval/
    iter_000000400/{metrics.json,plots/,predictions.npz}
    iter_000000500/{metrics.json,plots/,predictions.npz}
    iter_000000900/{metrics.json,plots/,predictions.npz}
    iter_000001000/{metrics.json,plots/,predictions.npz}

push-that-thing/model:action_decoder/stats_cache/
  (normalization stats hash files — pull before next session)
```

Headline numbers at session 1 close (32 val samples, stop=25/30 averaged):

```text
iter   val/loss   ratio(stop=30)   pred_std/target_std   deploy<1
500    11.0       1.96             0.86                  25%
900    7.5        2.15             0.56                  27%   # noisy dip
1000   ~7         1.96             0.62                  34%   # best so far
```

**Current best deploy candidate: iter 1000.** Notes:

- iter 900 looked like mode collapse but iter 1000 rebounded — single-eval
  signals are too noisy to trust; eval every 200-300 iters and look for
  consistent trend, not point-to-point change.
- val/loss alone is misleading for checkpoint selection — it can keep dropping
  while sampled-trajectory ratio gets worse. Always cross-check with the
  offline eval.
- For SO-101 (fixed-setup push from A to B), the `train_mean` baseline is
  unusually strong because most of the trajectory is similar across demos.
  Ratio > 1 here doesn't necessarily mean the policy is undeployable in
  practice — robot rollout is the real test.

## Multi-Session Chain Workflow

Brev hours are limited, so a full training run is split across multiple ~7-hour
sessions. Each session resumes from the previous session's last checkpoint by
keeping the same `JOB_NAME` and pulling the saved state down from HF.

### Per-Session Budget on 1x H100 PCIe

```text
Iter time (steady state)     ~22-26 s/iter  # 75M action decoder
                             ~19.3 s/iter   # legacy 17M action decoder
Stats compute (first time)   ~30 min        # cached after first run
First training iter warmup   ~30 s extra
Iters per 7-hour session     ~850-950       # at 75M, after subtracting overhead
                             ~1100-1200     # at legacy 17M
Eval (32 samples, 2 stops)   ~10-12 min     # slows training while running
```

For the 75M model targeted at **10k iters**, plan for ~**10 sessions of 7 hours
each**. See "Single Task Convergence Expectations" below for the model-quality
side of that estimate.

#### Checkpoint disk math at 75M

Per saved iter you write `model + optim + scheduler + trainer` totaling
~300 MB (vs ~100 MB at 17M). With `SAVE_ITER=50` over 10k iters that's 200
checkpoints × 300 MB = **~60 GB on disk and on HF**.

To keep this manageable, either:

- Bump `SAVE_ITER=100` or `SAVE_ITER=200` (20-50 checkpoints total over 10k
  iters), or
- Periodically prune older checkpoints between sessions:

  ```bash
  # Keep every 4th checkpoint
  find $SRC/checkpoints -name 'iter_*.pt' \
    | sort | awk -F'iter_|.pt' 'NR%4 != 0' | xargs rm
  ```

The recommended setting going forward is `SAVE_ITER=100`.

### Resume Flow (Session N+1, N ≥ 1)

On a fresh H100 Brev instance after one-time setup
(`bash .brev/setup.sh`, `hf auth login`, `wandb` auth, GitHub SSH key):

#### 1. Set variables and create target dirs

```bash
cd ~/pdt-mimic
JOB=w2a_so101_h100_chain_bsz2_accum8_full204
DST=mimic-video/model/checkpoints/vam/so101/$JOB
mkdir -p "$DST"
mkdir -p ~/.cache/mimic_lerobot_stats
```

#### 2. Pull the checkpoint tree

```bash
hf download push-that-thing/model --repo-type=model \
  --include "action_decoder/chain/$JOB/*" \
  --local-dir /tmp/ckpt-restore
cp -r /tmp/ckpt-restore/action_decoder/chain/$JOB/checkpoints/* "$DST/checkpoints/" 2>/dev/null || \
  cp -r /tmp/ckpt-restore/action_decoder/chain/$JOB/* "$DST/"
```

Verify the resume marker exists:

```bash
cat "$DST/checkpoints/latest_checkpoint.txt"
ls -lh "$DST/checkpoints/model/" | tail -5
```

#### 3. Pull the stats cache (skips 30-min compute)

```bash
hf download push-that-thing/model --repo-type=model \
  --include "action_decoder/stats_cache/*" \
  --local-dir /tmp/sc
cp -r /tmp/sc/action_decoder/stats_cache/* ~/.cache/mimic_lerobot_stats/
```

#### 4. Launch with the same JOB_NAME

```bash
tmux new -s train
cd ~/pdt-mimic
source mimic-video/model/.venv/bin/activate

NGPU=1 \
BSZ=1 \
MICRO_BSZ=2 \
GRAD_ACCUM_ITER=8 \
VAL_RATIO=0.10 \
RUN_VALIDATION=True \
VALIDATION_MODE=loss_only \
VALIDATION_ITER=100 \
MAX_VAL_ITER=16 \
SAVE_ITER=50 \
WANDB_MODE=online \
JOB_NAME=w2a_so101_h100_chain_bsz2_accum8_full204 \
bash scripts/train_action_decoder.sh trainer.max_iter=20000 2>&1 | tee /tmp/action_decoder_chain.log
```

Same `JOB_NAME` is essential. The checkpointer logs:

```text
Loading checkpoint from .../iter_000001000.pt
Resuming from iteration 1000
```

If you see `Training from scratch.` instead, the resume failed — kill it and
check the checkpoint directory structure.

#### 5. End-of-session upload

Before tearing down:

```bash
JOB=w2a_so101_h100_chain_bsz2_accum8_full204
SRC=mimic-video/model/checkpoints/vam/so101/$JOB

# Stop training cleanly
pkill -INT -f torchrun
sleep 10

# Upload latest checkpoints (idempotent — existing iters skip)
hf upload push-that-thing/model $SRC action_decoder/chain/$JOB --repo-type=model

# Upload any new eval results
hf upload push-that-thing/model \
  mimic-video/model/eval/action_decoder/$JOB \
  action_decoder/chain/$JOB/eval --repo-type=model

# Stats cache (idempotent — skips if unchanged)
hf upload push-that-thing/model ~/.cache/mimic_lerobot_stats \
  action_decoder/stats_cache --repo-type=model
```

Each `hf upload` must be a single shell line — line continuation via `\` can
break under bracketed-paste autoindent in tmux.

### Offline Eval Across the Chain

Every 200-300 iters, eval the latest checkpoint with stop_step scan:

```bash
cd ~/pdt-mimic/mimic-video/model
source .venv/bin/activate

PYTHONPATH=$PWD:../../lerobot/src python scripts/evaluate_action_decoder_checkpoint.py \
  --job-name w2a_so101_h100_chain_bsz2_accum8_full204 \
  --iteration <N> \
  --num-samples 32 \
  --stop-steps "25,30" \
  --no-plots
```

Notes:

- `--stop-steps "25,30"` is the deploy-realistic range. Add lower values
  (e.g., `"5,10,15,20,25,30"`) once to characterize the model, then stick to
  high stops for routine eval. Each stop_step adds time linearly.
- `--num-samples 32` keeps eval-time variance manageable. Below 16 samples,
  per-eval noise dominates point-to-point trend.
- Eval shares the GPU with training and slows it ~30-50% for the duration.
  Don't eval at every checkpoint; ~every 200 iters is enough.
- The `--wandb` flag fails with a 401 if WANDB_API_KEY is the 128-char new
  format (legacy CLI rejects). Skip `--wandb` and rely on local
  `metrics.json` files; HF-upload eval folder at end of session.

### What to Watch For Across Sessions

#### Healthy trajectory

```text
val/loss     descending across sessions, with noise per-session
ratio        descending across eval points (every 200-300 iters)
pred_std     stays above 0.5 * target_std (mode collapse warning if below)
deploy<1     fraction of samples crosses 50%+ for truly deployable
per-dim      dims 0, 4, 5 converge first; dims 1, 2 are hardest for SO-101
```

#### Failure mode: persistent mode collapse

Signal: `pred_std/target_std` drops below 0.5 and **stays there** for 2+
consecutive eval points, while ratio either stays flat or worsens.

If this happens, stop and apply intervention:

- Lower LR (try `LR=5.0e-5` instead of default `1.0e-4`)
- Enable EMA (currently `ema.enabled: False` in the model config)
- Larger effective batch (e.g., `GRAD_ACCUM_ITER=16` → effective 32) for
  smoother gradients
- If still bad: smaller architecture or fewer training iters before deploy test

Do not let a single bad eval (like our iter_900) trigger intervention.
Confirm with the next eval point first.

#### Failure mode: val/loss and ratio diverge

Signal: val/loss continues descending but offline ratio gets worse.

This is the most common signal that proxy ≠ deploy quality. Action: stop
using val/loss for checkpoint selection. Rely on offline eval ratio instead.
Pick the best checkpoint by ratio at high stop_step, not by val/loss.

#### Failure mode: per-dim MSE one dim runaway

Signal: one of dims 1, 2, 4 keeps growing while others descend. Indicates the
model is solving easy dims at the cost of hard ones.

Action: usually means more training is needed, not less. Hard dims often
descend later. Only worry if a hard dim has been growing for 1000+ iters.

### Single Task Convergence Expectations

`push-that-thing/task_1` is a single-task dataset (push object from fixed A to
fixed B). Realistic convergence:

```text
                          iters to deployable   H100 wall-clock @ 19s/iter
Optimistic                1-2k                  5-11 hours    (1-2 sessions)
Realistic                 2-5k                  11-26 hours   (2-4 sessions)
Pessimistic               5-10k                 26-53 hours   (4-8 sessions)
```

The proxy `mse_vs_train_mean_ratio` is unusually harsh for fixed-setup tasks
because the baseline (play back average trajectory) is already a coherent
motion. Ratio < 1 is the *strict* deploy threshold; in practice ratio < 1.5 at
stop=30 with `pred_std/target_std > 0.7` and 40%+ samples deploying might
already roll out usefully on the actual robot. **The robot rollout is the
real ground truth**, not the proxy metric.

### Common Gotchas From Real Runs

#### SMOKE=1 environment leak

Symptom: launch echo reports `save_iter=50` but Hydra dump shows
`save_iter: 99999999` and `run_validation: False`. Training proceeds but never
saves or validates.

Cause: `SMOKE=1` is set in the parent shell from a previous smoke test, and
`bash scripts/train_action_decoder.sh` inherits it as a sub-process env. The
script's `if [[ "${SMOKE:-0}" == "1" ]]` block then overrides everything.

Fix:

```bash
unset SMOKE
echo 'unset SMOKE' >> ~/.bashrc
```

Always verify after launch with:

```bash
grep -E "save_iter|run_validation" /tmp/action_decoder_chain.log | head
```

Must show `save_iter: 50` and `run_validation: True`. If you see
`99999999` or `False`, stop and clean up.

#### WANDB_MODE leak

Symptom: launch echo reports `wandb=1 (offline)` instead of `online`. Run
logs locally but never syncs to W&B.

Cause: `WANDB_MODE=offline` leaked from a prior smoke test.

Fix:

```bash
unset WANDB_MODE
```

Always set `WANDB_MODE=online` inline in the launch command so even a leaked
env var can't override.

#### W&B 128-char API key rejection

Symptom: `wandb login` interactive prompt rejects new-format keys:
`ValueError: API key must be 40 characters long, yours was 128`.

Cause: Legacy `wandb login` validator hardcodes 40-char keys; new keys are 128.

Fix:

```bash
export WANDB_API_KEY=<paste-128-char-key>
echo "export WANDB_API_KEY=<paste-128-char-key>" >> ~/.bashrc
```

The `WANDB_API_KEY` env var skips the validator. Add to `~/.bashrc` so future
tmux panes inherit it.

#### Bracketed-paste line splitting

Symptom: a single multi-line shell command with `\` line continuations gets
split into multiple commands by the terminal, dropping some env vars and
breaking arg counts. `hf upload` lands with one too few args, etc.

Cause: tmux/iTerm bracketed-paste interacts with shell autoindent.

Fix: keep each shell command as a single physical line, or use shell variables
for long paths. Multi-line `\`-continued commands work best when the whole
block is pasted in one operation, not retyped piecewise.

#### Stale dataset cache

Symptom: `num_episodes` reports an outdated count (e.g., 57 instead of 204)
because the LeRobot cache from a previous box was populated before the dataset
was extended.

Fix:

```bash
rm -rf ~/.cache/huggingface/lerobot/push-that-thing/task_1
```

Always run the episode count check before launching:

```bash
cd ~/pdt-mimic/mimic-video/model
PYTHONPATH=$PWD:../../lerobot/src python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('num_episodes:', LeRobotDataset('push-that-thing/task_1').num_episodes)"
```

Expect `num_episodes: 204` (or higher if the dataset has grown).

## How to Tune Based on Observations

### Next Run: 10% Validation and Early Stopping

With 204 episodes, the wrapper holds out `ceil(204 * 0.10) = 21` validation
episodes with the configured seed (default `42`) and trains on the remaining
183. The episode split is deterministic from `seed`. Normalizer statistics are
recomputed over the training split only (the stats cache key includes
`val_ratio`/`seed`/`train`), so there is no validation leakage.

Use loss-only validation for training-time early stopping. The "full"
validation path samples actions across ~36 video-noise sigmas and additionally
*generates* future video before sampling again — it is tens of × more
expensive than a train step and not suitable for periodic in-training checks.
Reserve it for offline checkpoint eval.

#### 2x H100 launch (recommended for full-dataset runs)

Effective batch matches the A100 batch-2 run (`2 * 2 * 4 = 16`):

```bash
cd ~/workspace/pdt-mimic
tmux new -s action_decoder_val

source mimic-video/model/.venv/bin/activate

NGPU=2 \
BSZ=1 \
MICRO_BSZ=2 \
GRAD_ACCUM_ITER=4 \
VAL_RATIO=0.10 \
EARLY_STOPPING=1 \
EARLY_STOP_PATIENCE=5 \
EARLY_STOP_MIN_DELTA=0.05 \
EARLY_STOP_START_ITER=2000 \
VALIDATION_ITER=500 \
MAX_VAL_ITER=64 \
SAVE_ITER=250 \
JOB_NAME=w2a_so101_2xh100_val10_bsz2_accum4 \
bash scripts/train_action_decoder.sh trainer.max_iter=20000 2>&1 | tee -a /tmp/action_decoder_2xh100_val10.log
```

Expect ~2.5-3.5x faster wall-clock than 1x A100, not a clean 2x — video
decoding, validation, DDP overhead, and PCIe eat some of the gain.

#### 1x A100 launch (cheap-iteration fallback)

```bash
cd ~/workspace/pdt-mimic
tmux new -s action_decoder_val

source mimic-video/model/.venv/bin/activate

NGPU=1 \
BSZ=1 \
MICRO_BSZ=2 \
GRAD_ACCUM_ITER=8 \
VAL_RATIO=0.10 \
EARLY_STOPPING=1 \
EARLY_STOP_PATIENCE=5 \
EARLY_STOP_MIN_DELTA=0.05 \
EARLY_STOP_START_ITER=2000 \
VALIDATION_ITER=500 \
MAX_VAL_ITER=64 \
SAVE_ITER=250 \
JOB_NAME=w2a_so101_a100_normfix_bsz2_accum8_val10 \
bash scripts/train_action_decoder.sh trainer.max_iter=20000 2>&1 | tee -a /tmp/action_decoder_val10.log
```

#### Validation settings

```text
VAL_RATIO=0.10              # hold out about 10% of episodes (21 of 204)
VALIDATION_MODE=loss_only   # implied by EARLY_STOPPING=1
VALIDATION_ITER=500         # check val loss every 500 optimizer steps
MAX_VAL_ITER=64             # cap validation batches so training keeps moving
EARLY_STOP_PATIENCE=5       # stop after 5 non-improving validations
EARLY_STOP_MIN_DELTA=0.05   # require at least this much val/loss improvement
EARLY_STOP_START_ITER=2000  # do not stop during the very early phase
```

Each validation check uses up to `MAX_VAL_ITER * MICRO_BSZ * NGPU` held-out
windows, not the full 21-episode validation set. For a slower but more
complete estimate, increase `MAX_VAL_ITER` or remove it. Do this only if the
extra runtime is acceptable.

### If It OOMs

Use safer memory settings:

```bash
MICRO_BSZ=1
GRAD_ACCUM_ITER=16
```

If it still OOMs:

```bash
MICRO_BSZ=1
GRAD_ACCUM_ITER=8
```

This reduces effective batch size to 8 but should be easier to run.

### If GPU Memory Has Room

Try:

```bash
MICRO_BSZ=2
GRAD_ACCUM_ITER=8
```

This keeps effective batch size 16 but may be faster than accumulating 16
single-sample micro-steps.

### If Loss Is Very Noisy

Increase effective batch size through accumulation:

```bash
MICRO_BSZ=1
GRAD_ACCUM_ITER=32
```

Only do this if training speed is acceptable. Larger effective batches usually
make the curve smoother, but each optimizer update takes longer.

### If Loss Is Huge and Var_inst Is Huge

Stop. This points to action normalization failing.

Check:

```bash
git -C mimic-video log -1 --oneline
```

Must include:

```text
83725eb Preserve self-mapped normalizer groups
```

Then rerun smoke.

### If Loss Plateaus

First check deploy/eval behavior on intermediate checkpoints. If the policy
does not improve:

- Try longer training before changing architecture.
- Compare checkpoints at 250, 500, 1000, 2000, 5000 iterations.
- Consider lowering LR only if loss is unstable after normalization is confirmed.

Current LR:

```text
1.0e-4
```

Reasonable alternatives:

```text
5.0e-5 for more conservative training
2.0e-4 only if learning is clearly too slow and stable
```

### If W&B Fails With Personal Entity Disabled

Use the team entity:

```bash
export WANDB_ENTITY=elias-andreoli-eth-z-rich
export WANDB_PROJECT=vam
export WANDB_MODE=online
```

Or rely on:

```text
configs/wandb/action_decoder.env
```

## What Counts as Pipeline Verified

The pipeline is verified when:

1. Dataset sample shapes match the expected SO-101 shapes.
2. Fine-tuned video backbone loads successfully.
3. T5 embeddings load without recomputing unexpectedly.
4. Normalized action variance metric is sane: `Var_inst[x_0]` around order `1`.
5. Training runs for a smoke test and writes a checkpoint.
6. W&B logs to the team project.
7. A real run saves intermediate checkpoints at the configured interval.
