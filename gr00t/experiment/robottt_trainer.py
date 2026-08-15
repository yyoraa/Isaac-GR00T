# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""RoboTTT curriculum, checkpoint metadata, and Trainer integration."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature
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

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        payload = inputs.get("inputs", {})
        if "trajectory_shape" not in payload:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        outputs = model(**inputs)
        loss = outputs["loss"]
        self.loss = loss
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _slice_trajectory_inputs(inputs: dict[str, Any], start: int, end: int) -> dict[str, Any]:
        payload = inputs["inputs"]
        batch_size, trajectory_length = [int(value) for value in payload["trajectory_shape"]]
        segment_length = end - start
        sliced = {}
        for key, value in payload.items():
            if key == "trajectory_shape":
                sliced[key] = value.new_tensor([batch_size, segment_length])
            elif (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and tuple(value.shape[:2])
                == (
                    batch_size,
                    trajectory_length,
                )
            ):
                sliced[key] = value[:, start:end]
            elif (
                isinstance(value, torch.Tensor)
                and value.ndim >= 1
                and value.shape[0] % (batch_size * trajectory_length) == 0
                and value.shape[0] != batch_size
            ):
                multiplicity = value.shape[0] // (batch_size * trajectory_length)
                reshaped = value.reshape(
                    batch_size, trajectory_length, multiplicity, *value.shape[1:]
                )
                sliced[key] = reshaped[:, start:end].reshape(
                    batch_size * segment_length * multiplicity, *value.shape[1:]
                )
            else:
                sliced[key] = value
        return {"inputs": sliced}

    def training_step(self, model, inputs, num_items_in_batch=None):
        payload = inputs.get("inputs", {})
        if "trajectory_shape" not in payload:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        micro_batch_size = getattr(
            model.config,
            "robottt_backbone_micro_batch_size",
            None,
        )
        if micro_batch_size is not None:
            return self._training_step_with_cached_backbone(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
                micro_batch_size=int(micro_batch_size),
            )

        inputs = self._prepare_inputs(inputs)
        _, trajectory_length = [int(value) for value in inputs["inputs"]["trajectory_shape"]]
        segment_length = int(getattr(model.config, "robottt_tbptt_steps", 1))
        fast_state = None
        reported_loss = torch.zeros((), device=self.args.device)
        for start in range(0, trajectory_length, segment_length):
            end = min(start + segment_length, trajectory_length)
            segment = self._slice_trajectory_inputs(inputs, start, end)
            segment["robottt_state"] = fast_state
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(
                    model,
                    segment,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )
            weight = (end - start) / trajectory_length
            scaled_loss = loss * weight / self.current_gradient_accumulation_steps
            kwargs = {}
            if self.accelerator.distributed_type.value == "DEEPSPEED":
                kwargs["scale_wrt_gas"] = False
            self.accelerator.backward(scaled_loss, **kwargs)
            reported_loss = reported_loss + loss.detach() * weight
            fast_state = outputs.robottt_state.detach()
        return reported_loss

    @staticmethod
    def _slice_cached_backbone_output(
        output: BatchFeature,
        start: int,
        end: int,
        batch_size: int,
        trajectory_length: int,
    ) -> BatchFeature:
        sliced = {}
        for key, value in output.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and tuple(value.shape[:2]) == (batch_size, trajectory_length)
            ):
                sliced[key] = value[:, start:end]
            else:
                sliced[key] = value
        return BatchFeature(data=sliced)

    def _training_step_with_cached_backbone(
        self,
        model,
        inputs,
        *,
        num_items_in_batch=None,
        micro_batch_size: int,
    ):
        if micro_batch_size <= 0:
            raise ValueError("robottt_backbone_micro_batch_size must be positive")
        unwrapped_model = self.accelerator.unwrap_model(model)
        if any(parameter.requires_grad for parameter in unwrapped_model.backbone.parameters()):
            raise RuntimeError("cached RoboTTT training requires a fully frozen backbone")

        payload = inputs["inputs"]
        batch_size, trajectory_length = [int(value) for value in payload["trajectory_shape"]]
        tbptt_steps = int(getattr(model.config, "robottt_tbptt_steps", 1))
        if tbptt_steps != 1:
            raise ValueError("cached RoboTTT training currently requires robottt_tbptt_steps=1")
        chunk_length = max(1, micro_batch_size // batch_size)
        fast_state = None
        reported_loss = torch.zeros((), device=self.args.device)

        for chunk_start in range(0, trajectory_length, chunk_length):
            chunk_end = min(chunk_start + chunk_length, trajectory_length)
            raw_chunk = self._slice_trajectory_inputs(inputs, chunk_start, chunk_end)
            prepared_chunk = self._prepare_inputs(raw_chunk)
            backbone_inputs, action_inputs = unwrapped_model.prepare_input(prepared_chunk["inputs"])
            current_chunk_length = chunk_end - chunk_start
            with self.compute_loss_context_manager(), torch.no_grad():
                backbone_outputs = unwrapped_model.backbone(backbone_inputs)
            unwrapped_model._reshape_sequence_backbone_output(
                backbone_outputs,
                batch_size,
                current_chunk_length,
            )

            for local_start in range(current_chunk_length):
                local_end = local_start + 1
                cached_backbone = self._slice_cached_backbone_output(
                    backbone_outputs,
                    local_start,
                    local_end,
                    batch_size,
                    current_chunk_length,
                )
                cached_action = BatchFeature(
                    data=self._slice_trajectory_inputs(
                        {"inputs": action_inputs},
                        local_start,
                        local_end,
                    )["inputs"]
                )
                with self.compute_loss_context_manager():
                    outputs = model(
                        cached_backbone_output=cached_backbone,
                        cached_action_input=cached_action,
                        robottt_state=fast_state,
                    )
                    loss = outputs["loss"]
                weight = 1.0 / trajectory_length
                scaled_loss = loss * weight / self.current_gradient_accumulation_steps
                kwargs = {}
                if self.accelerator.distributed_type.value == "DEEPSPEED":
                    kwargs["scale_wrt_gas"] = False
                self.accelerator.backward(scaled_loss, **kwargs)
                reported_loss = reported_loss + loss.detach() * weight
                fast_state = outputs.robottt_state.detach()

        return reported_loss

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
        else:
            self.apply_curriculum(0)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
