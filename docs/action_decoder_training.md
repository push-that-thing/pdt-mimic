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

Recommended for the current stage:

```text
1x A100 80GB
```

This is enough for training and much cheaper than H100. H100 is faster if the
price gap is acceptable, but not required. Avoid L40S for the real run because
48GB VRAM is risky for this pipeline.

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

Current SO-101 action decoder parameters:

```text
max_horizon = 91
in_channels = 6
out_channels = 6
model_channels = 256
num_blocks = 7
num_heads = 4
mlp_ratio = 3.0
crossattn_emb_channels = 2048
use_adaln_lora = False
adaln_lora_dim = 0
pair_timestep_feature_rank = 256
atten_backend = flash_attn_no_cp
sac_config = none
```

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

## How to Tune Based on Observations

### Next Run: 10% Validation and Early Stopping

If `push-that-thing/task_1` has 204 episodes, use a 10% validation split for
the next clean run. The dataset wrapper will hold out `ceil(204 * 0.10) = 21`
episodes with the configured seed, train on the remaining episodes, and run
validation on the held-out split.

Use loss-only validation for training-time early stopping. The full validation
path also samples actions across video-noise settings and is too expensive for
routine in-training checks on one A100.

Recommended launch:

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

Validation settings:

```text
VAL_RATIO=0.10              # hold out about 10% of episodes
VALIDATION_MODE=loss_only   # implied by EARLY_STOPPING=1
VALIDATION_ITER=500         # check val loss every 500 optimizer steps
MAX_VAL_ITER=64             # cap validation batches so training keeps moving
EARLY_STOP_PATIENCE=5       # stop after 5 non-improving validations
EARLY_STOP_MIN_DELTA=0.05   # require at least this much val/loss improvement
EARLY_STOP_START_ITER=2000  # do not stop during the very early phase
```

For a slower but more complete validation estimate, increase `MAX_VAL_ITER` or
remove it. Do this only if the extra runtime is acceptable.

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
