"""Thin wrapper around `lerobot_record` that registers our BYOP policy.

LeRobot v0.4.0's policy factory is a hardcoded if/elif chain — it has no plugin
discovery for `lerobot_policy_*` packages (unlike robots/cameras/teleoperators).
Importing `lerobot_policy_mimic_video` registers `MimicVideoConfig` with
draccus's ChoiceRegistry (so the CLI accepts `--policy.type=mimic_video`), but
`make_policy()` / `make_pre_post_processors()` still dispatch off hardcoded
branches. We wrap those factory functions here to delegate `mimic_video` to
our package.
"""
import logging

# Suppress the per-tick clamp warning from ensure_safe_goal_position. It fires
# every control step when max_relative_target is engaged, and the pformat dump
# makes the log unreadable. Filter on substring against the root logger.
class _ClampWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Relative goal position magnitude had to be clamped" not in record.getMessage()

logging.getLogger().addFilter(_ClampWarningFilter())

import lerobot_policy_mimic_video  # noqa: F401 — registers MimicVideoConfig
from lerobot_policy_mimic_video import (
    MimicVideoConfig,
    MimicVideoPolicy,
    make_mimic_video_pre_post_processors,
)

from lerobot.policies import factory as _policy_factory

_orig_get_policy_class = _policy_factory.get_policy_class
_orig_make_policy_config = _policy_factory.make_policy_config
_orig_make_pre_post_processors = _policy_factory.make_pre_post_processors


def _get_policy_class(name):
    if name == "mimic_video":
        return MimicVideoPolicy
    return _orig_get_policy_class(name)


def _make_policy_config(policy_type, **kwargs):
    if policy_type == "mimic_video":
        return MimicVideoConfig(**kwargs)
    return _orig_make_policy_config(policy_type, **kwargs)


def _make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
    if isinstance(policy_cfg, MimicVideoConfig) and not pretrained_path:
        return make_mimic_video_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )
    return _orig_make_pre_post_processors(policy_cfg, pretrained_path=pretrained_path, **kwargs)


_policy_factory.get_policy_class = _get_policy_class
_policy_factory.make_policy_config = _make_policy_config
_policy_factory.make_pre_post_processors = _make_pre_post_processors

# `lerobot_record` does `from … import make_pre_post_processors`, binding the
# original by name in its own namespace — patch that binding too.
from lerobot.scripts import lerobot_record as _record_mod  # noqa: E402
from lerobot.scripts.lerobot_record import main  # noqa: E402

_record_mod.make_pre_post_processors = _make_pre_post_processors

if __name__ == "__main__":
    main()
