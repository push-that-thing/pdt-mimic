# MIMIC Video LeRobot Policy

This package exposes the custom MIMIC video SO101 policy through LeRobot's
bring-your-own-policy interface.

Install it from the repository root after the MIMIC/LeRobot environment is set
up:

```bash
pip install -e ./lerobot_policy_mimic_video
```

The policy type is `mimic_video`. It expects the MIMIC model code in
`mimic-video/model`, a video backbone checkpoint, an action decoder checkpoint,
and MIMIC-format low-dimensional statistics.

The LeRobot processors are implemented, but the default processor pipeline does
not normalize `observation.state` or unnormalize `action`: MIMIC already does
that inside `World2ActionPipeline`.
