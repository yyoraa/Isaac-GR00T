# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Exact public RoboTTT stage presets and optimizer helpers."""

from dataclasses import dataclass
from typing import Literal

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class RoboTTTTrainingConfig:
    stage: Literal["stage1", "stage2"]
    max_steps: int = 20_000
    learning_rate: float = 5e-5
    scheduler: Literal["wsd", "cosine"] = "cosine"
    weight_decay: float = 1e-5
    context_length: int = 1024
    tbptt_steps: int = 128
    use_lora: bool = False
    gradient_checkpointing: bool = True
    deepspeed_config: str = "gr00t/configs/deepspeed/robottt_zero3_offload.json"

    def __post_init__(self) -> None:
        if self.stage not in ("stage1", "stage2"):
            raise ValueError(f"stage must be stage1 or stage2, got {self.stage}")
        if self.use_lora:
            raise ValueError("RoboTTT reproduction uses exact full/new-parameter tuning, not LoRA")
        if self.context_length <= 0 or self.tbptt_steps <= 0:
            raise ValueError("context_length and tbptt_steps must be positive")
        if self.stage == "stage2" and self.context_length != 1024:
            raise ValueError("RoboTTT Stage 2 requires context_length=1024")

    @classmethod
    def for_stage(cls, stage: Literal["stage1", "stage2"]) -> "RoboTTTTrainingConfig":
        if stage == "stage1":
            return cls(
                stage="stage1",
                max_steps=30_000,
                learning_rate=2e-5,
                scheduler="wsd",
                context_length=8192,
            )
        if stage == "stage2":
            return cls(stage="stage2")
        raise ValueError(f"stage must be stage1 or stage2, got {stage}")


def set_robottt_stage_trainability(model: nn.Module, stage: str) -> None:
    """Apply the paper's Stage 1 freeze or Stage 2 full-tuning policy."""
    if stage == "stage1":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.endswith("register_tokens") or ".robottt." in name
        return
    if stage == "stage2":
        model.requires_grad_(True)
        return
    raise ValueError(f"stage must be stage1 or stage2, got {stage}")


def build_wsd_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
) -> LambdaLR:
    """Build linear warmup, stable plateau, and linear decay scheduling."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or decay_steps <= 0 or warmup_steps + decay_steps > total_steps:
        raise ValueError("invalid WSD phase lengths")
    decay_start = total_steps - decay_steps

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / warmup_steps
        if step <= decay_start:
            return 1.0
        return max(0.0, (total_steps - step) / decay_steps)

    return LambdaLR(optimizer, factor)
