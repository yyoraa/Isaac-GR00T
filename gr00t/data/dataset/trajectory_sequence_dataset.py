# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Episode-safe trajectory windows and temporal collation for RoboTTT."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig

from .lerobot_episode_loader import LeRobotEpisodeLoader


@dataclass(frozen=True)
class TrajectoryWindow:
    episode_id: int
    start: int
    length: int


class TrajectorySequenceDataset(ShardedDataset):
    """Expose consecutive windows while preserving each episode boundary."""

    def __init__(
        self,
        episode_loader: Any,
        step_factory: Callable[[Any, int], dict[str, Any]],
        context_length: int,
        stride: int,
        action_horizon: int,
    ):
        super().__init__(getattr(episode_loader, "dataset_path", None))
        if context_length <= 0 or stride <= 0 or action_horizon <= 0:
            raise ValueError("context_length, stride, and action_horizon must be positive")
        self.episode_loader = episode_loader
        self.step_factory = step_factory
        self.context_length = context_length
        self.stride = stride
        self.action_horizon = action_horizon
        self.windows = self._build_windows(episode_loader.episode_lengths)

    @classmethod
    def from_lerobot(
        cls,
        dataset_path: str,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        context_length: int,
        stride: int,
        action_horizon: int,
        allow_padding: bool = False,
    ) -> "TrajectorySequenceDataset":
        loader = LeRobotEpisodeLoader(dataset_path, modality_configs)
        dataset = cls(loader, lambda _episode, _step: {}, context_length, stride, action_horizon)
        dataset.dataset_path = dataset_path
        dataset.embodiment_tag = embodiment_tag
        dataset.modality_configs = modality_configs
        dataset.allow_padding = allow_padding
        dataset.processor = None
        dataset.step_factory = dataset._process_lerobot_step
        return dataset

    def _process_lerobot_step(self, episode: Any, step: int) -> dict[str, Any]:
        from .sharded_single_step_dataset import extract_step_data

        if self.processor is None:
            raise RuntimeError("processor must be set before loading trajectory windows")
        content = extract_step_data(
            episode,
            step,
            self.modality_configs,
            self.embodiment_tag,
            self.allow_padding,
        )
        return self.processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])

    def _build_windows(self, episode_lengths: Sequence[int]) -> list[TrajectoryWindow]:
        windows = []
        for episode_id, raw_length in enumerate(episode_lengths):
            effective_length = max(0, int(raw_length) - self.action_horizon + 1)
            if effective_length == 0:
                continue
            starts = (
                [0]
                if effective_length <= self.context_length
                else range(0, effective_length, self.stride)
            )
            for start in starts:
                windows.append(
                    TrajectoryWindow(
                        episode_id=episode_id,
                        start=start,
                        length=min(self.context_length, effective_length - start),
                    )
                )
        if not windows:
            raise ValueError("no episode is long enough for the configured action horizon")
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def get_shard_length(self, idx: int) -> int:
        self.windows[idx]
        return 1

    def get_shard(self, idx: int) -> list[dict[str, Any]]:
        window = self.windows[idx]
        episode = self.episode_loader[window.episode_id]
        steps = [
            self.step_factory(episode, step)
            for step in range(window.start, window.start + window.length)
        ]
        return [
            {
                "episode_id": window.episode_id,
                "start": window.start,
                "length": window.length,
                "trajectory_steps": steps,
            }
        ]

    def get_dataset_statistics(self) -> dict[str, Any]:
        return self.episode_loader.get_dataset_statistics()


class TrajectoryCollator:
    """Pad trajectory time, then delegate per-step VLM collation."""

    def __init__(self, base_collator: Callable[[list[dict[str, Any]]], Any]):
        self.base_collator = base_collator
        self.context_length: int | None = None

    def set_context_length(self, context_length: int) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.context_length = context_length

    def __call__(self, features: list[dict[str, Any]]) -> BatchFeature:
        if not features:
            raise ValueError("cannot collate an empty trajectory batch")
        batch_size = len(features)
        lengths = [int(feature["length"]) for feature in features]
        if self.context_length is not None:
            lengths = [min(length, self.context_length) for length in lengths]
        max_length = max(lengths)
        flat_steps = []
        valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
        positions = torch.zeros(batch_size, max_length, dtype=torch.long)
        episode_ids = torch.empty(batch_size, dtype=torch.long)

        for batch_index, (feature, length) in enumerate(zip(features, lengths)):
            all_steps = feature["trajectory_steps"]
            if int(feature["length"]) != len(all_steps) or length <= 0:
                raise ValueError("trajectory length does not match trajectory_steps")
            steps = all_steps[:length]
            padded = list(steps) + [steps[-1]] * (max_length - length)
            flat_steps.extend(padded)
            valid_mask[batch_index, :length] = True
            start = int(feature["start"])
            positions[batch_index, :length] = torch.arange(start, start + length)
            positions[batch_index, length:] = start + length - 1
            episode_ids[batch_index] = int(feature["episode_id"])

        collated = self.base_collator(flat_steps)
        flat_inputs = collated["inputs"]
        inputs = {}
        flat_batch = batch_size * max_length
        vlm_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        for key, value in flat_inputs.items():
            if (
                key not in vlm_keys
                and isinstance(value, torch.Tensor)
                and value.shape[0] == flat_batch
            ):
                inputs[key] = value.reshape(batch_size, max_length, *value.shape[1:])
            else:
                inputs[key] = value

        reset_mask = torch.zeros_like(valid_mask)
        reset_mask[:, 0] = True
        inputs.update(
            {
                "valid_mask": valid_mask,
                "action_loss_mask": valid_mask.clone(),
                "episode_reset_mask": reset_mask,
                "temporal_positions": positions,
                "episode_id": episode_ids,
                "trajectory_shape": torch.tensor([batch_size, max_length]),
            }
        )
        return BatchFeature(data={"inputs": inputs})
