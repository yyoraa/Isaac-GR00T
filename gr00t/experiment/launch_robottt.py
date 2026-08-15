#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Launch the public GR00T N1.7 + RoboTTT reproduction on RoboCasa365."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Literal

import tyro

from gr00t.configs.base_config import Config, get_default_config
from gr00t.configs.robottt_training import RoboTTTTrainingConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.experiment import run


@dataclass
class RoboTTTLaunchConfig:
    stage: Literal["stage1", "stage2"]
    base_model_path: str
    dataset_path: str
    manifest_path: str
    embodiment_tag: str = "robocasa365_panda_omron"
    output_dir: str = "./outputs/robottt-robocasa365"
    experiment_name: str | None = None
    resume_from_checkpoint: bool = False
    num_gpus: int = 1
    gradient_accumulation_steps: int = 8
    save_steps: int = 1_000
    save_total_limit: int = 5
    use_wandb: bool = False
    dataloader_num_workers: int = 0
    tbptt_steps: int = 1
    backbone_micro_batch_size: int = 8
    max_steps: int | None = None
    context_length: int | None = None


def _manifest_sha256(path: str | Path) -> str:
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"RoboCasa365 manifest does not exist: {manifest}")
    digest = hashlib.sha256()
    with manifest.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_robottt_config(launch: RoboTTTLaunchConfig) -> Config:
    preset = RoboTTTTrainingConfig.for_stage(launch.stage)
    tag_name = (
        "robocasa365_panda_omron"
        if launch.embodiment_tag.lower() == "panda_omron"
        else launch.embodiment_tag
    )
    embodiment = EmbodimentTag.resolve(tag_name).value
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [
                            path for path in launch.dataset_path.split(os.pathsep) if path
                        ],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.robottt_enabled = True
    config.model.robottt_num_register_tokens = 16
    config.model.robottt_inner_dim = 3072
    config.model.robottt_inner_lr = 0.1
    config.model.robottt_rope_theta = 10_000.0
    config.model.robottt_gate_init = 0.001
    if launch.tbptt_steps <= 0:
        raise ValueError("tbptt_steps must be positive")
    if launch.backbone_micro_batch_size <= 0:
        raise ValueError("backbone_micro_batch_size must be positive")
    if launch.max_steps is not None and launch.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if launch.context_length is not None and launch.context_length <= 0:
        raise ValueError("context_length must be positive")
    config.model.robottt_tbptt_steps = launch.tbptt_steps
    config.model.robottt_backbone_micro_batch_size = (
        launch.backbone_micro_batch_size if launch.stage == "stage1" else None
    )
    config.model.robottt_analytic_inner_update = True
    config.model.robottt_compile_inner_update = True
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.load_bf16 = launch.stage == "stage1"
    config.model.backbone_trainable_params_fp32 = launch.stage == "stage2"
    config.model.use_relative_action = True

    config.data.sequence_mode = True
    context_length = launch.context_length or preset.context_length
    config.data.context_length = context_length
    config.data.sequence_stride = context_length
    config.data.tbptt_steps = launch.tbptt_steps

    config.training.start_from_checkpoint = launch.base_model_path
    config.training.output_dir = launch.output_dir
    config.training.experiment_name = launch.experiment_name
    config.training.max_steps = launch.max_steps or preset.max_steps
    config.training.learning_rate = preset.learning_rate
    config.training.lr_scheduler_type = preset.scheduler
    if launch.stage == "stage1" and launch.max_steps is not None:
        config.training.robottt_wsd_decay_steps = min(
            config.training.robottt_wsd_decay_steps,
            max(1, config.training.max_steps // 10),
        )
        if config.training.max_steps == 1:
            config.training.warmup_ratio = 0
    config.training.weight_decay = preset.weight_decay
    config.training.global_batch_size = launch.num_gpus
    config.training.gradient_accumulation_steps = launch.gradient_accumulation_steps
    config.training.gradient_checkpointing = preset.gradient_checkpointing
    config.training.optim = "adamw_torch"
    config.training.save_steps = launch.save_steps
    config.training.save_total_limit = launch.save_total_limit
    config.training.save_only_model = False
    config.training.resume_from_checkpoint = launch.resume_from_checkpoint
    config.training.use_wandb = launch.use_wandb
    config.training.dataloader_num_workers = launch.dataloader_num_workers
    config.training.num_gpus = launch.num_gpus
    config.training.use_ddp = launch.stage == "stage1" and launch.num_gpus > 1
    config.training.robottt_stage = launch.stage
    config.training.robottt_manifest_hash = _manifest_sha256(launch.manifest_path)
    if launch.context_length is not None:
        config.training.robottt_curriculum_buckets = [launch.context_length]
    else:
        config.training.robottt_curriculum_buckets = (
            [128, 512, 1024, 2048, 4096, 8192] if launch.stage == "stage1" else [1024]
        )
    config.training.deepspeed_config_path = (
        preset.deepspeed_config if launch.stage == "stage2" else None
    )
    return config


if __name__ == "__main__":
    os.environ.setdefault("LOGURU_LEVEL", "INFO")
    run(build_robottt_config(tyro.cli(RoboTTTLaunchConfig, description=__doc__)))
