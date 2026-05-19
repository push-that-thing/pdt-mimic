# pdt-mimic on Jetson Thor

This repo deploys the mimic-video policy on an SO-101 follower arm. The host is a
Jetson Thor (aarch64, CUDA 13) — **not** the x86_64 cu126 setup the upstream
`mimic-video` README assumes. The `thor_inference` branch carries the patches
that make the stack import and run on Thor.

## Layout

- `mimic-video/` — submodule, vendored cosmos_predict2 + models. On branch
  `thor_inference`; the commit makes transformer-engine, apex, and decord
  optional (Thor has none of them) and adds a pure-PyTorch
  `apply_rotary_pos_emb` fallback.
- `lerobot/` — submodule pinned near v0.4.0. On branch `thor_inference` with a
  one-line fix for a Python 3.12 dataclass strictness bug in
  `policies/groot/groot_n1.py`.
- `lerobot_policy_mimic_video/` — the policy wrapper. Defaults (in
  `configuration_mimic_video.py`) point at `iter_000003400` + the combined-LoRA
  backbone + `stop_after_step=15`.
- `scripts/` — entry points (see below).

All three repos are on branch `thor_inference`. The main repo's `thor_inference`
branches off `main` (last shared commit: `6b18a69`).

## Activating the env

```
source scripts/mimic_env.sh
```

This script handles both layouts — x86_64 cu126 wheels (`nvidia/<pkg>/lib`) and
the Thor cu13 wheels (`nvidia/cu13/lib`) — and loads the HF token from
`$HF_HOME/token`. The venv lives at `mimic-video/model/.venv` (Python 3.12 on
Thor, 3.10 on the x86 build). Do not call `pip` directly; use `uv pip` against
the venv if you need to add a package.

## Bringing a fresh Thor up

`scripts/setup_thor.sh` (idempotent; `DRY_RUN=1` to preview) records the install
recipe — torch / flash-attn / cuDSS from the `jetson-ai-lab.io sbsa/cu130` wheel
index, plus the four x86-only cosmos deps that are skipped on Thor (apex,
transformer-engine, natten, triton-from-cosmos-index).

## Running inference

- `scripts/smoke_load.py` — loads the policy and runs a synthetic observation
  through it. ~82 s for the first call, ~72 s steady-state.
- `scripts/deploy_so101.sh` — real-hardware deploy via `lerobot_record_mimic`.
  Tees all output to `/tmp/pdt-mimic-logs/deploy_so101_<ts>.log` and prints the
  last 40 lines on crash. Tunable via `ROBOT_PORT`, `CAMERA_IDX`,
  `NUM_EPISODES`, `EPISODE_TIME_S`.
- `scripts/deploy_so101_no_keyboard.sh` — fallback for when only 2 USB-A ports
  are available; prompts via terminal beeps and timed banners so devices can be
  hot-swapped without typing.

## Known latency issue

Inference produces a `(1, 90, 6)` bfloat16 action chunk = 3 s of robot motion at
30 Hz. Steady-state generation takes ~72 s, so the arm is idle ~24× longer than
it moves. The pipeline is end-to-end correct (output is image-conditioned, not
fixed), but **not real-time-deployable** until this is brought down. Don't claim
deploy works without addressing it.

## Hardware constraints to keep in mind

- Only 2 USB-A ports, no working hub. Arm + camera fills both; the keyboard has
  to come out once the arm is plugged in. SSH or USB-C tether is the
  workflow — see "Remote access".
- The SO-101 controller appears as `/dev/ttyACM*` (typically `ttyACM0`).
  `lerobot-find-port` needs you to physically unplug between its two prompts; if
  the diff is empty the arm wasn't enumerating in the first place (check
  `dmesg` for `cdc_acm`).

## Remote access (no keyboard on Thor)

- WiFi: Thor was at `10.6.2.134` on `wlP1p1s0`, DHCP — the IP can change on
  reboot or when switching networks. Check with `ip -4 addr show wlP1p1s0` if
  SSH suddenly fails.
- USB-C tether: Thor exposes a USB-Ethernet gadget at `192.168.55.1` on
  `l4tbr0`. Plug a USB-C data cable laptop ↔ Thor, then
  `ssh ethrc-thor@192.168.55.1` from the laptop. Works without shared WiFi.
- Authorized key for `jessli@ethz.ch` is installed at
  `~/.ssh/authorized_keys` (mode 600).

## Conventions

- Author for commits on this box: `JessieeeNotLi <jessli@ethz.com>`. No global
  git identity is set — pass `-c user.name=… -c user.email=…` per commit.
- HF assets cache at `$HF_HOME` (defaults to `~/.cache/huggingface`). The token
  is read from `$HF_HOME/token`.
- Logs go under `/tmp/pdt-mimic-logs/`.
