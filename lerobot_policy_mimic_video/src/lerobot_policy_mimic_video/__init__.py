"""LeRobot BYOP package for the MIMIC video SO101 policy."""

import sys

try:
    import lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover - import-time integration guard
    raise ImportError(
        "lerobot is not installed. Install LeRobot before using lerobot_policy_mimic_video."
    ) from exc

from .configuration_mimic_video import MimicVideoConfig
from .modeling_mimic_video import MimicVideoPolicy
from .processor_mimic_video import make_mimic_video_pre_post_processors


def _register_with_lerobot() -> None:
    """Teach lerobot's hardcoded factory about the mimic_video plugin policy.

    lerobot 0.4.0 auto-discovers third-party robot/camera/teleop plugins, but it
    resolves *policies* through hardcoded if/elif chains in
    ``lerobot.policies.factory`` (``get_policy_class`` / ``make_pre_post_processors``
    / ``make_policy_config``) with no plugin hook. ``MimicVideoConfig`` already
    registers via ``@PreTrainedConfig.register_subclass("mimic_video")`` so the CLI
    parses ``--policy.type=mimic_video`` — but those three factory lookups would
    still raise ``NotImplementedError``. We wrap them here.

    The deploy wrapper imports this package *before* ``lerobot.scripts.lerobot_record``,
    so record.py's ``from ...factory import make_pre_post_processors`` binds to the
    patched function. The ``sys.modules`` fix-up below also covers the reverse order.
    """
    from lerobot.policies import factory

    if getattr(factory, "_mimic_video_registered", False):
        return

    _orig_get_policy_class = factory.get_policy_class
    _orig_make_pre_post = factory.make_pre_post_processors
    _orig_make_policy_config = factory.make_policy_config

    def get_policy_class(name):
        if name == "mimic_video":
            return MimicVideoPolicy
        return _orig_get_policy_class(name)

    def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
        if isinstance(policy_cfg, MimicVideoConfig):
            return make_mimic_video_pre_post_processors(
                config=policy_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
            )
        return _orig_make_pre_post(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    def make_policy_config(policy_type, **kwargs):
        if policy_type == "mimic_video":
            return MimicVideoConfig(**kwargs)
        return _orig_make_policy_config(policy_type, **kwargs)

    factory.get_policy_class = get_policy_class
    factory.make_pre_post_processors = make_pre_post_processors
    factory.make_policy_config = make_policy_config
    factory._mimic_video_registered = True

    # If lerobot_record was imported first, rebind its direct `from ... import`
    # reference too (get_policy_class is resolved via the factory module global
    # inside make_policy, so it needs no fix-up).
    _rec = sys.modules.get("lerobot.scripts.lerobot_record")
    if _rec is not None and hasattr(_rec, "make_pre_post_processors"):
        _rec.make_pre_post_processors = make_pre_post_processors


_register_with_lerobot()

__all__ = [
    "MimicVideoConfig",
    "MimicVideoPolicy",
    "make_mimic_video_pre_post_processors",
]
