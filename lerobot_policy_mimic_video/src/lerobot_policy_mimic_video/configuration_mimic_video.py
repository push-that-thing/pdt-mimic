from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
except ImportError:  # LeRobot has moved these types across releases.
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

from lerobot.configs.policies import PreTrainedConfig


def _identity_normalization_mapping() -> dict[FeatureType, NormalizationMode]:
    return {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.STATE: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }


@PreTrainedConfig.register_subclass("mimic_video")
@dataclass
class MimicVideoConfig(PreTrainedConfig):
    """Configuration for deploying the MIMIC video SO101 policy through LeRobot."""

    # LeRobot/robot-facing keys.
    camera_key: str = "observation.images.front"
    state_key: str = "observation.state"
    task: str = "push the object"
    observation_rename_map: dict[str, str] = field(default_factory=dict)

    # MIMIC timing and tensor contract.
    image_height: int = 480
    image_width: int = 640
    obs_image_horizon: int = 5
    source_fps: int = 30
    image_target_fps: int = 5
    action_dim: int = 6
    horizon: int = 90
    n_action_steps: int = 10
    action_start_offset_steps: int = 0

    # MIMIC model/config paths. Relative paths are resolved from the current
    # working directory and then from this repository root.
    mimic_model_dir: str = "mimic-video/model"
    experiment_name: str = "w2a_so101_lerobot_v2w_push_that_thing_lr1.000e-04_layer20_bsz1"
    video_checkpoint_path: str = "mimic-video/model/checkpoints/video_backbone/v2w_push_that_thing.pt"
    action_checkpoint_path: str = (
        "mimic-video/model/checkpoints/vam/so101/"
        "w2a_so101_h100_chain_bsz2_accum8_full204/checkpoints/model/iter_000001000.pt"
    )
    dataset_statistics_path: str | None = None
    stats_cache_dir: str | None = "~/.cache/mimic_lerobot_stats"
    recompute_statistics_if_missing: bool = False

    # Dataset settings used only if statistics must be recomputed.
    dataset_repo_id: str = "push-that-thing/task_1"
    dataset_root: str | None = None
    dataset_val_ratio: float | None = 0.1
    dataset_seed: int = 42

    # Inference knobs.
    num_sampling_step: int = 35
    stop_after_step: int | None = 30
    seed: int = 0
    increment_seed_per_chunk: bool = False
    use_cuda_graphs: bool = False
    torch_dtype: str = "bfloat16"
    load_on_init: bool = True
    disable_guardrails: bool = True

    # The base config also owns this field; keep it identity by default because
    # the MIMIC world-to-action pipeline normalizes/unnormalizes internally.
    normalization_mapping: dict[FeatureType, NormalizationMode] = field(
        default_factory=_identity_normalization_mapping
    )

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.obs_image_horizon not in {1, 5}:
            raise ValueError("MIMIC video inference supports obs_image_horizon of 1 or 5.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive.")
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps cannot exceed horizon.")
        if self.action_start_offset_steps < 0:
            raise ValueError("action_start_offset_steps must be non-negative.")

    def validate_features(self) -> None:
        if self.input_features and self.camera_key not in self.input_features:
            raise ValueError(f"MIMIC video requires image feature '{self.camera_key}'.")
        if self.input_features and self.state_key not in self.input_features:
            raise ValueError(f"MIMIC video requires state feature '{self.state_key}'.")
        if self.output_features and self.action_feature is None:
            raise ValueError("MIMIC video requires an action output feature.")

    def get_optimizer_preset(self) -> Any:
        try:
            from lerobot.optim.optimizers import AdamWConfig
        except ImportError:
            from lerobot.common.optim.optimizers import AdamWConfig

        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> list[int] | None:
        if self.obs_image_horizon <= 1:
            return None
        stride = max(1, round(self.source_fps / self.image_target_fps))
        return [-(self.obs_image_horizon - 1 - i) * stride for i in range(self.obs_image_horizon)]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def default_input_features(self) -> dict[str, PolicyFeature]:
        return {
            self.camera_key: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, self.image_height, self.image_width),
            ),
            self.state_key: PolicyFeature(type=FeatureType.STATE, shape=(self.action_dim,)),
        }

    @property
    def default_output_features(self) -> dict[str, PolicyFeature]:
        return {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
        }
