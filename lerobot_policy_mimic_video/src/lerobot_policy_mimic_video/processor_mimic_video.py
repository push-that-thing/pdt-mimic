from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

try:
    from lerobot.configs import PipelineFeatureType, PolicyFeature
except ImportError:
    from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
try:
    from lerobot.processor import policy_action_to_transition, transition_to_policy_action
except ImportError:
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

try:
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
except ImportError:
    POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
    POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

from .configuration_mimic_video import MimicVideoConfig


@dataclass
@ProcessorStepRegistry.register(name="mimic_video_observation_processor")
class MimicVideoObservationProcessorStep(ObservationProcessorStep):
    """Prepare LeRobot observations for the MIMIC policy boundary.

    The processor intentionally avoids low-dimensional normalization. It only
    makes image/state tensors well-formed so MIMIC can perform its own internal
    preprocessing and normalizer logic.
    """

    image_key: str
    state_key: str
    image_height: int = 480
    image_width: int = 640
    convert_images_to_uint8: bool = True

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        processed = dict(observation)
        if self.image_key in processed:
            processed[self.image_key] = self._process_image(processed[self.image_key])
        if self.state_key in processed:
            processed[self.state_key] = torch.as_tensor(processed[self.state_key], dtype=torch.float32)
        return processed

    def _process_image(self, image: Any) -> torch.Tensor:
        tensor = torch.as_tensor(image)
        tensor = self._channels_first(tensor)
        tensor = self._resize(tensor)
        if self.convert_images_to_uint8:
            tensor = self._to_uint8(tensor)
        elif tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        return tensor.contiguous()

    def _channels_first(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            if tensor.shape[-1] in {1, 3, 4}:
                tensor = tensor.permute(2, 0, 1)
            if tensor.shape[0] == 4:
                tensor = tensor[:3]
            return tensor
        if tensor.ndim == 4:
            if tensor.shape[-1] in {1, 3, 4}:  # BHWC or THWC
                tensor = tensor.permute(0, 3, 1, 2)
            if tensor.shape[1] == 4:
                tensor = tensor[:, :3]
            return tensor
        if tensor.ndim == 5:
            if tensor.shape[-1] in {1, 3, 4}:  # BTHWC
                tensor = tensor.permute(0, 1, 4, 2, 3)
            if tensor.shape[2] == 4:
                tensor = tensor[:, :, :3]
            return tensor
        raise ValueError(f"Unsupported image shape for MIMIC video policy: {tuple(tensor.shape)}")

    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        target = (self.image_height, self.image_width)
        if tensor.shape[-2:] == target:
            return tensor

        original_dtype = tensor.dtype
        if tensor.ndim == 3:
            resized = F.interpolate(tensor.float().unsqueeze(0), size=target, mode="bilinear", align_corners=False)
            return resized.squeeze(0).to(dtype=original_dtype)
        if tensor.ndim == 4:
            return F.interpolate(tensor.float(), size=target, mode="bilinear", align_corners=False).to(
                dtype=original_dtype
            )
        if tensor.ndim == 5:
            b, t, c, _h, _w = tensor.shape
            flat = tensor.reshape(b * t, c, tensor.shape[-2], tensor.shape[-1])
            resized = F.interpolate(flat.float(), size=target, mode="bilinear", align_corners=False)
            return resized.reshape(b, t, c, *target).to(dtype=original_dtype)
        return tensor

    @staticmethod
    def _to_uint8(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.uint8:
            return tensor
        if not tensor.is_floating_point():
            return tensor.to(dtype=torch.uint8)
        min_val = float(tensor.min().detach().cpu())
        max_val = float(tensor.max().detach().cpu())
        if min_val >= -1.0001 and max_val <= 1.0001 and min_val < 0.0:
            tensor = (tensor + 1.0) * 127.5
        elif min_val >= -0.0001 and max_val <= 1.0001:
            tensor = tensor * 255.0
        return tensor.round().clamp(0, 255).to(dtype=torch.uint8)

    def get_config(self) -> dict[str, Any]:
        return {
            "image_key": self.image_key,
            "state_key": self.state_key,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "convert_images_to_uint8": self.convert_images_to_uint8,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="mimic_video_action_processor")
class MimicVideoActionProcessorStep(PolicyActionProcessorStep):
    """Convert the policy action tensor back to the robot-facing single action."""

    squeeze_batch: bool = True
    check_finite: bool = True

    def action(self, action: PolicyAction) -> PolicyAction:
        if self.check_finite and not torch.isfinite(action).all():
            raise RuntimeError("MIMIC video policy produced a non-finite action.")
        if self.squeeze_batch and action.ndim == 2 and action.shape[0] == 1:
            return action.squeeze(0)
        return action

    def get_config(self) -> dict[str, Any]:
        return {
            "squeeze_batch": self.squeeze_batch,
            "check_finite": self.check_finite,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_mimic_video_pre_post_processors(
    config: MimicVideoConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    del dataset_stats  # MIMIC loads its own stats into World2ActionPipeline.

    input_steps = [
        RenameObservationsProcessorStep(rename_map=config.observation_rename_map),
        MimicVideoObservationProcessorStep(
            image_key=config.camera_key,
            state_key=config.state_key,
            image_height=config.image_height,
            image_width=config.image_width,
            convert_images_to_uint8=True,
        ),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [
        MimicVideoActionProcessorStep(),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
