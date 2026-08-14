# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""RoboTTT curriculum, checkpoint metadata, and Trainer integration."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from transformers.trainer import get_last_checkpoint
from transformers.trainer_callback import TrainerCallback

from gr00t.configs.robottt_training import build_wsd_scheduler
from gr00t.experiment.trainer import Gr00tTrainer


ROBOTTT_STATE_NAME = "robottt_state.json"


@dataclass(frozen=True)
class RoboTTTCurriculum:
    buckets: tuple[int, ...]
    total_steps: int

    def __post_init__(self) -> None:
        if not self.buckets or any(value <= 0 for value in self.buckets):
            raise ValueError("curriculum buckets must be positive")
        if tuple(sorted(set(self.buckets))) != self.buckets:
            raise ValueError("curriculum buckets must be strictly increasing")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")

    def bucket_for_step(self, step: int) -> tuple[int, int]:
        if step < 0:
            raise ValueError("step must be non-negative")
        phase_steps = math.ceil(self.total_steps / len(self.buckets))
        index = min(step // phase_steps, len(self.buckets) - 1)
        return index, self.buckets[index]


@dataclass(frozen=True)
class RoboTTTCheckpointState:
    stage: str
    global_step: int
    curriculum_bucket: int
    context_length: int
    manifest_sha256: str
    sampler_state: dict[str, Any]
    rng_state_file: str = "rng_state.pth"

    def save(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        temporary = path / f".{ROBOTTT_STATE_NAME}.tmp"
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        temporary.replace(path / ROBOTTT_STATE_NAME)

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> "RoboTTTCheckpointState":
        path = Path(checkpoint_dir) / ROBOTTT_STATE_NAME
        if not path.is_file():
            raise ValueError(f"RoboTTT checkpoint metadata is missing: {path}")
        return cls(**json.loads(path.read_text()))

    def validate(self, *, stage: str, manifest_sha256: str) -> None:
        if self.stage != stage:
            raise ValueError(f"checkpoint stage is {self.stage}, requested {stage}")
        if self.manifest_sha256 != manifest_sha256:
            raise ValueError("dataset manifest hash differs from the checkpoint")


class _RoboTTTStateCallback(TrainerCallback):
    def __init__(self, trainer: "RoboTTTTrainer"):
        self.trainer = trainer

    def on_step_begin(self, args, state, control, **kwargs):
        self.trainer.apply_curriculum(state.global_step)

    def on_save(self, args, state, control, **kwargs):
        self.trainer.save_robottt_state(state.global_step)


class RoboTTTTrainer(Gr00tTrainer):
    """Gr00tTrainer with WSD and fail-closed RoboTTT resume metadata."""

    def __init__(
        self,
        *args,
        robottt_stage: str,
        robottt_manifest_hash: str,
        robottt_curriculum: RoboTTTCurriculum,
        robottt_wsd_decay_steps: int = 1_000,
        **kwargs,
    ):
        self.robottt_stage = robottt_stage
        self.robottt_manifest_hash = robottt_manifest_hash
        self.robottt_curriculum = robottt_curriculum
        self.robottt_wsd_decay_steps = robottt_wsd_decay_steps
        super().__init__(*args, **kwargs)
        self.add_callback(_RoboTTTStateCallback(self))

    def apply_curriculum(self, step: int) -> None:
        _, context_length = self.robottt_curriculum.bucket_for_step(step)
        if hasattr(self.data_collator, "set_context_length"):
            self.data_collator.set_context_length(context_length)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.robottt_stage != "stage1":
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is None:
            optimizer = self.optimizer if optimizer is None else optimizer
            warmup_steps = self.args.get_warmup_steps(num_training_steps)
            self.lr_scheduler = build_wsd_scheduler(
                optimizer,
                total_steps=num_training_steps,
                warmup_steps=warmup_steps,
                decay_steps=self.robottt_wsd_decay_steps,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def save_robottt_state(self, global_step: int) -> None:
        if not self.args.should_save:
            return
        bucket, context_length = self.robottt_curriculum.bucket_for_step(global_step)
        dataset = self.train_dataset
        if hasattr(dataset, "state_dict"):
            sampler_state = dataset.state_dict()
        else:
            sampler_state = {
                key: getattr(dataset, key)
                for key in ("seed", "epoch", "curr_shard_index")
                if hasattr(dataset, key)
            }
        RoboTTTCheckpointState(
            stage=self.robottt_stage,
            global_step=global_step,
            curriculum_bucket=bucket,
            context_length=context_length,
            manifest_sha256=self.robottt_manifest_hash,
            sampler_state=sampler_state,
        ).save(Path(self.args.output_dir) / f"checkpoint-{global_step}")

    def train(self, resume_from_checkpoint=None, **kwargs):
        checkpoint = resume_from_checkpoint
        if checkpoint is True:
            checkpoint = get_last_checkpoint(self.args.output_dir)
        if checkpoint not in (None, False):
            state = RoboTTTCheckpointState.load(checkpoint)
            state.validate(
                stage=self.robottt_stage,
                manifest_sha256=self.robottt_manifest_hash,
            )
            if hasattr(self.train_dataset, "load_state_dict"):
                self.train_dataset.load_state_dict(state.sampler_state)
            self.apply_curriculum(state.global_step)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
