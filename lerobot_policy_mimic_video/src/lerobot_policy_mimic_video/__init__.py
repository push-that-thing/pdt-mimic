"""LeRobot BYOP package for the MIMIC video SO101 policy."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover - import-time integration guard
    raise ImportError(
        "lerobot is not installed. Install LeRobot before using lerobot_policy_mimic_video."
    ) from exc

from .configuration_mimic_video import MimicVideoConfig
from .modeling_mimic_video import MimicVideoPolicy
from .processor_mimic_video import make_mimic_video_pre_post_processors

__all__ = [
    "MimicVideoConfig",
    "MimicVideoPolicy",
    "make_mimic_video_pre_post_processors",
]
