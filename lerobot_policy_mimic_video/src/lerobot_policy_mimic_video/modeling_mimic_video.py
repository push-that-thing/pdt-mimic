from __future__ import annotations

import collections
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from lerobot.policies.pretrained import PreTrainedPolicy
except ImportError:
    from lerobot.common.policies.pretrained import PreTrainedPolicy

from .configuration_mimic_video import MimicVideoConfig

LOGGER = logging.getLogger(__name__)


class MimicVideoPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper around the MIMIC video-to-action pipeline."""

    config_class = MimicVideoConfig
    name = "mimic_video"

    def __init__(self, config: MimicVideoConfig, dataset_stats: dict[str, Any] | None = None):
        super().__init__(config, dataset_stats)
        del dataset_stats
        config.validate_features()
        self.config = config
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.model: nn.Module | None = None
        self._prompt_embedding: torch.Tensor | None = None
        self._image_history: collections.deque[torch.Tensor] = collections.deque(maxlen=config.obs_image_horizon)
        self._action_queue: collections.deque[torch.Tensor] = collections.deque()
        self._chunks_generated = 0

        if config.load_on_init:
            self.load_mimic_pipeline()

    def reset(self) -> None:
        self._image_history.clear()
        self._action_queue.clear()
        self._chunks_generated = 0

    def get_optim_params(self) -> dict[str, Any]:
        return {"params": [self._dummy]}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del batch
        raise NotImplementedError("MimicVideoPolicy is an inference/deployment wrapper, not a trainer.")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        chunk = self._run_mimic(batch, **kwargs)
        start = min(self.config.action_start_offset_steps, max(chunk.shape[1] - 1, 0))
        return chunk[:, start:, :]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            if chunk.shape[0] != 1:
                raise ValueError("MimicVideoPolicy deployment currently supports batch size 1.")
            execute_steps = min(self.config.n_action_steps, chunk.shape[1])
            for action in chunk[0, :execute_steps]:
                self._action_queue.append(action)
        return self._action_queue.popleft()

    def load_mimic_pipeline(self) -> None:
        mimic_model_dir = self._resolve_path(self.config.mimic_model_dir, expect_exists=True)
        video_checkpoint = self._resolve_path(self.config.video_checkpoint_path, expect_exists=True)
        action_checkpoint = self._resolve_path(self.config.action_checkpoint_path, expect_exists=True)

        mimic_model_dir_str = str(mimic_model_dir)
        if mimic_model_dir_str not in sys.path:
            sys.path.insert(0, mimic_model_dir_str)

        from cosmos_predict2.configs.config import make_config
        from cosmos_predict2.data.action.utils import extract_normalization_types
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
        from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
        from cosmos_predict2.pipelines.world2action import World2ActionPipeline
        from imaginaire.lazy_config import instantiate
        from imaginaire.utils.config_helper import override

        dtype = self._torch_dtype(self.config.torch_dtype)
        device = self.config.device or "cuda"

        mimic_config = override(make_config(), ["--", f"experiment={self.config.experiment_name}"])
        if self.config.disable_guardrails:
            mimic_config.model.config.video_pipe_config.guardrail_config.enabled = False
        mimic_config.model.config.video_dit_path = str(video_checkpoint)
        if self.config.action_num_denoising_steps is not None:
            mimic_config.model.config.pipe_config.scheduler.num_denoising_steps = (
                self.config.action_num_denoising_steps
            )

        if not self.config.use_text_encoder:
            self._prompt_embedding = self._load_prompt_embedding(dtype=dtype, device=device)

        video_pipe = self._load_video_pipe(
            Video2WorldPipeline,
            mimic_config,
            video_checkpoint,
            device=device,
            dtype=dtype,
        )
        action_pipe = World2ActionPipeline.from_config(
            mimic_config.model.config.pipe_config,
            dit_path=str(action_checkpoint),
            device=device,
            dtype=dtype,
        )

        data_config = instantiate(mimic_config.data_config)
        stats = self._load_statistics(mimic_config=mimic_config, instantiate=instantiate)
        action_pipe.normalizer.build_from_stats(
            stats,
            normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
            concat_groups=data_config.policy_io.concat_groups,
            device=device,
            dtype=dtype,
        )

        pipeline = Video2World2ActionPipeline(video_pipe, action_pipe)
        if str(device).startswith("cuda"):
            pipeline = pipeline.cuda()
        self.model = pipeline.eval()

    def _load_video_pipe(
        self,
        video_pipe_cls,
        mimic_config,
        video_checkpoint: Path,
        *,
        device: str,
        dtype: torch.dtype,
    ):
        kwargs = {
            "config": mimic_config.model.config.video_pipe_config,
            "dit_path": str(video_checkpoint),
            "device": device,
            "torch_dtype": dtype,
            "load_ema_to_reg": False,
        }
        use_te = bool(self.config.use_text_encoder)
        try:
            return video_pipe_cls.from_config(**kwargs, use_text_encoder=use_te)
        except TypeError:
            if not use_te:
                raise RuntimeError(
                    "Cosmos Video2WorldPipeline.from_config does not accept use_text_encoder=False; "
                    "cannot skip T5 loading on this build."
                )
            return video_pipe_cls.from_config(**kwargs)

    def _load_prompt_embedding(self, *, dtype: torch.dtype, device: str) -> torch.Tensor:
        task = self.config.task
        explicit = self.config.prompt_embedding_path
        if explicit:
            path = self._resolve_path(explicit, expect_exists=True)
        else:
            cache_dir = Path(self.config.prompt_embedding_cache_dir or "~/.cache/mimic_lerobot_t5/.t5_embeddings").expanduser()
            if not cache_dir.is_dir():
                raise FileNotFoundError(
                    f"T5 embedding cache dir not found: {cache_dir}. "
                    "Run data_preprocessing/action/precompute_t5_lerobot.py first, "
                    "or set use_text_encoder=True / prompt_embedding_path."
                )
            cache_key = hashlib.sha256(str(sorted([task])).encode()).hexdigest()
            preferred = cache_dir / f"{cache_key}.npz"
            if preferred.exists():
                path = preferred
            else:
                path = None
                for cand in sorted(cache_dir.glob("*.npz")):
                    with np.load(cand, allow_pickle=False) as data:
                        if task in data.files:
                            path = cand
                            break
                if path is None:
                    raise FileNotFoundError(
                        f"No T5 embedding for task {task!r} in {cache_dir}. "
                        f"Tried hash {cache_key}.npz and scanned all .npz files."
                    )

        with np.load(path, allow_pickle=False) as data:
            if task not in data.files:
                raise KeyError(f"Task {task!r} not found in {path}; keys: {list(data.files)}")
            emb = np.asarray(data[task])
        tensor = torch.as_tensor(emb).to(device=device, dtype=dtype)
        LOGGER.info("Loaded precomputed T5 embedding from %s (shape=%s)", path, tuple(tensor.shape))
        return tensor

    def _load_statistics(self, *, mimic_config, instantiate) -> dict[str, dict[str, np.ndarray]]:
        stats_path = self._find_statistics_path()
        if stats_path is not None:
            with stats_path.open() as f:
                return self._numpy_stats(json.load(f))

        if not self.config.recompute_statistics_if_missing:
            raise FileNotFoundError(
                "MIMIC statistics were not found. Set dataset_statistics_path, restore "
                "action_decoder/stats_cache into stats_cache_dir, or set "
                "recompute_statistics_if_missing=True."
            )

        train_dataset = instantiate(mimic_config.dataloader_train.dataset)
        return self._numpy_stats(train_dataset.get_statistics())

    def _find_statistics_path(self) -> Path | None:
        candidates: list[Path] = []
        if self.config.dataset_statistics_path:
            candidates.append(self._resolve_path(self.config.dataset_statistics_path, expect_exists=False))
        if self.config.stats_cache_dir:
            candidates.append(self._resolve_path(self.config.stats_cache_dir, expect_exists=False))

        for candidate in candidates:
            candidate = candidate.expanduser()
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                files = [p for p in candidate.iterdir() if p.is_file()]
                if files:
                    newest = max(files, key=lambda p: p.stat().st_mtime)
                    LOGGER.info("Using newest MIMIC stats cache file: %s", newest)
                    return newest
        return None

    @staticmethod
    def _numpy_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
        return {
            key: {sub_key: np.asarray(value, dtype=np.float32) for sub_key, value in sub_stats.items()}
            for key, sub_stats in stats.items()
        }

    def _run_mimic(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        if self.model is None:
            self.load_mimic_pipeline()

        assert self.model is not None
        input_vid = self._prepare_video(batch)
        state = self._prepare_state(batch)
        prompt = self._extract_prompt(batch, kwargs)
        seed = self.config.seed + self._chunks_generated if self.config.increment_seed_per_chunk else self.config.seed
        self._chunks_generated += 1

        return self.model(
            input_vid=input_vid,
            state_B_HO_O=state,
            prompt=prompt,
            prompt_embedding=self._prompt_embedding,
            num_sampling_step=self.config.num_sampling_step,
            stop_after_step=self.config.stop_after_step,
            seed=seed,
            use_cuda_graphs=self.config.use_cuda_graphs,
        )

    def _prepare_video(self, batch: dict[str, Any]) -> torch.Tensor:
        image = self._get_required(batch, self.config.camera_key)
        tensor = torch.as_tensor(image)
        frames = self._to_cthw(tensor)

        if frames.shape[1] == 1:
            self._image_history.append(frames[:, 0].detach())
            history = list(self._image_history)
            while len(history) < self.config.obs_image_horizon:
                history.insert(0, history[0])
            frames = torch.stack(history[-self.config.obs_image_horizon :], dim=1)
        elif frames.shape[1] > self.config.obs_image_horizon:
            frames = frames[:, -self.config.obs_image_horizon :, :, :]

        if frames.shape[1] not in {1, 5}:
            raise ValueError(f"MIMIC expects 1 or 5 image frames, got {frames.shape[1]}.")

        return frames.unsqueeze(0).contiguous()

    def _prepare_state(self, batch: dict[str, Any]) -> torch.Tensor:
        state = torch.as_tensor(self._get_required(batch, self.config.state_key), dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim == 3 and state.shape[1] == 1:
            return self._to_model_device(state)
        if state.ndim != 2:
            raise ValueError(f"Expected state shape (D,) or (B,D), got {tuple(state.shape)}.")
        if state.shape[-1] != self.config.action_dim:
            raise ValueError(f"Expected SO101 state dim {self.config.action_dim}, got {state.shape[-1]}.")
        return self._to_model_device(state.unsqueeze(1))

    def _to_cthw(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self._strip_batch_dim(tensor)
        if tensor.ndim == 3:
            tensor = self._chw(tensor).unsqueeze(1)
        elif tensor.ndim == 4:
            tensor = self._temporal_cthw(tensor)
        elif tensor.ndim == 5:
            tensor = self._temporal_cthw(self._strip_batch_dim(tensor))
        else:
            raise ValueError(f"Unsupported image tensor shape: {tuple(tensor.shape)}")

        tensor = self._resize_cthw(tensor)
        if tensor.dtype != torch.uint8 and tensor.is_floating_point():
            tensor = self._float_image_to_uint8(tensor)
        elif tensor.dtype != torch.uint8:
            tensor = tensor.to(dtype=torch.uint8)
        return tensor

    @staticmethod
    def _strip_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim >= 4 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        return tensor

    @staticmethod
    def _chw(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] in {1, 3, 4}:
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] == 4:
            tensor = tensor[:3]
        if tensor.shape[0] != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got shape {tuple(tensor.shape)}.")
        return tensor

    def _temporal_cthw(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"Expected temporal image tensor with 4 dims, got {tuple(tensor.shape)}.")
        if tensor.shape[0] == 3:  # CTHW
            return tensor
        if tensor.shape[1] == 3:  # TCHW
            return tensor.permute(1, 0, 2, 3)
        if tensor.shape[-1] == 3:  # THWC
            return tensor.permute(3, 0, 1, 2)
        raise ValueError(f"Could not infer temporal image layout for shape {tuple(tensor.shape)}.")

    def _resize_cthw(self, tensor: torch.Tensor) -> torch.Tensor:
        target = (self.config.image_height, self.config.image_width)
        if tensor.shape[-2:] == target:
            return tensor
        c, t, h, w = tensor.shape
        flat = tensor.permute(1, 0, 2, 3).reshape(t, c, h, w).float()
        resized = F.interpolate(flat, size=target, mode="bilinear", align_corners=False)
        return resized.reshape(t, c, *target).permute(1, 0, 2, 3).to(dtype=tensor.dtype)

    @staticmethod
    def _float_image_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
        min_val = float(tensor.min().detach().cpu())
        max_val = float(tensor.max().detach().cpu())
        if min_val >= -1.0001 and max_val <= 1.0001 and min_val < 0.0:
            tensor = (tensor + 1.0) * 127.5
        elif min_val >= -0.0001 and max_val <= 1.0001:
            tensor = tensor * 255.0
        return tensor.round().clamp(0, 255).to(dtype=torch.uint8)

    def _to_model_device(self, tensor: torch.Tensor) -> torch.Tensor:
        device = self.config.device or "cuda"
        dtype = self._torch_dtype(self.config.torch_dtype)
        return tensor.to(device=device, dtype=dtype)

    def _extract_prompt(self, batch: dict[str, Any], kwargs: dict[str, Any]) -> str:
        value = kwargs.get("task") or kwargs.get("task_description") or batch.get("task") or self.config.task
        if isinstance(value, (list, tuple)):
            value = value[0]
        if isinstance(value, torch.Tensor):
            raise ValueError("Task prompt must be text, not a tensor.")
        return str(value)

    @staticmethod
    def _get_required(batch: dict[str, Any], key: str) -> Any:
        if key not in batch:
            raise KeyError(f"Missing required MIMIC video policy input '{key}'.")
        return batch[key]

    @staticmethod
    def _torch_dtype(name: str) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported torch_dtype '{name}'.")
        return mapping[name]

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def _resolve_path(self, value: str, *, expect_exists: bool) -> Path:
        path = Path(value).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.extend(
                [
                    Path.cwd() / path,
                    self._repo_root() / path,
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        resolved = candidates[-1].resolve()
        if expect_exists:
            raise FileNotFoundError(resolved)
        return resolved
