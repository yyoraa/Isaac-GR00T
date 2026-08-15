from contextlib import nullcontext
import json
from types import SimpleNamespace

from gr00t.configs.robottt_training import (
    RoboTTTTrainingConfig,
    build_wsd_scheduler,
    set_robottt_stage_trainability,
)
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.trajectory_sequence_dataset import TrajectorySequenceDataset
from gr00t.experiment.launch_robottt import RoboTTTLaunchConfig, build_robottt_config
from gr00t.experiment.robottt_trainer import (
    ROBOTTT_STATE_NAME,
    RoboTTTCheckpointState,
    RoboTTTCurriculum,
    RoboTTTTrainer,
)
from gr00t.experiment.trainer import Gr00tTrainer
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature
from transformers.trainer_utils import SchedulerType


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.action_head = nn.Module()
        self.action_head.register_tokens = nn.Parameter(torch.zeros(16, 2))
        block = nn.Module()
        block.robottt = nn.Linear(2, 2)
        self.action_head.model = nn.Module()
        self.action_head.model.transformer_blocks = nn.ModuleList([block])
        self.action_head.action_decoder = nn.Linear(2, 2)


class _RecordingCollator:
    context_length = None

    def set_context_length(self, context_length):
        self.context_length = context_length


class _BatchFeatureModel(nn.Module):
    def forward(self, **inputs):
        return BatchFeature(
            data={
                "loss": torch.tensor(2.5, requires_grad=True),
                "robottt_state": "next-state",
            }
        )


def test_stage_presets_match_public_robottt_schedule():
    stage1 = RoboTTTTrainingConfig.for_stage("stage1")
    stage2 = RoboTTTTrainingConfig.for_stage("stage2")

    assert (stage1.max_steps, stage1.learning_rate, stage1.scheduler) == (
        30_000,
        2e-5,
        SchedulerType.WARMUP_STABLE_DECAY.value,
    )
    assert (stage2.max_steps, stage2.learning_rate, stage2.scheduler) == (
        20_000,
        5e-5,
        "cosine",
    )
    assert stage2.context_length == 1024
    assert stage1.weight_decay == stage2.weight_decay == 1e-5


def test_stage1_trains_only_registers_and_robottt_parameters():
    model = _TinyModel()
    set_robottt_stage_trainability(model, "stage1")

    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable == {
        "action_head.register_tokens",
        "action_head.model.transformer_blocks.0.robottt.weight",
        "action_head.model.transformer_blocks.0.robottt.bias",
    }


def test_stage2_trains_every_parameter():
    model = _TinyModel()
    set_robottt_stage_trainability(model, "stage1")
    set_robottt_stage_trainability(model, "stage2")

    assert all(value.requires_grad for value in model.parameters())


def test_wsd_scheduler_has_warmup_stable_and_decay_phases():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5)
    scheduler = build_wsd_scheduler(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        decay_steps=3,
    )

    factors = [scheduler.lr_lambdas[0](step) for step in range(11)]
    assert factors[0] == 0.0
    assert factors[2] == 1.0
    assert factors[6] == 1.0
    assert factors[10] == 0.0


def test_stage2_rejects_lora_or_non_1024_context():
    for kwargs in ({"use_lora": True}, {"context_length": 512}):
        try:
            RoboTTTTrainingConfig(stage="stage2", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid Stage 2 config accepted: {kwargs}")


def test_curriculum_and_checkpoint_state_round_trip(tmp_path):
    curriculum = RoboTTTCurriculum((128, 512, 1024, 2048, 4096, 8192), 30_000)
    assert curriculum.bucket_for_step(0) == (0, 128)
    assert curriculum.bucket_for_step(5_000) == (1, 512)
    assert curriculum.bucket_for_step(29_999) == (5, 8192)

    state = RoboTTTCheckpointState(
        stage="stage1",
        global_step=5_000,
        curriculum_bucket=1,
        context_length=512,
        manifest_sha256="a" * 64,
        sampler_state={"seed": 42, "epoch": 3, "cursor": 17},
    )
    state.save(tmp_path)
    restored = RoboTTTCheckpointState.load(tmp_path)

    assert restored == state
    assert json.loads((tmp_path / ROBOTTT_STATE_NAME).read_text())["sampler_state"]["cursor"] == 17
    restored.validate(stage="stage1", manifest_sha256="a" * 64)


def test_fresh_training_applies_first_curriculum_bucket_before_parent_train(monkeypatch):
    trainer = object.__new__(RoboTTTTrainer)
    trainer.robottt_curriculum = RoboTTTCurriculum((128, 512), 10)
    trainer.data_collator = _RecordingCollator()

    monkeypatch.setattr(
        Gr00tTrainer,
        "train",
        lambda self, **kwargs: self.data_collator.context_length,
    )

    assert trainer.train() == 128


def test_robottt_compute_loss_accepts_batch_feature_outputs(monkeypatch):
    trainer = object.__new__(RoboTTTTrainer)

    def reject_parent(*args, **kwargs):
        raise AssertionError("RoboTTT sequence loss must not use the mapping-only parent path")

    monkeypatch.setattr(Gr00tTrainer, "compute_loss", reject_parent)

    loss, outputs = trainer.compute_loss(
        _BatchFeatureModel(),
        {"inputs": {"trajectory_shape": torch.tensor([1, 1])}},
        return_outputs=True,
    )

    assert loss.item() == 2.5
    assert outputs.robottt_state == "next-state"
    assert trainer.loss is loss


def test_resume_state_rejects_stage_or_manifest_mismatch(tmp_path):
    state = RoboTTTCheckpointState(
        stage="stage1",
        global_step=1,
        curriculum_bucket=0,
        context_length=128,
        manifest_sha256="b" * 64,
        sampler_state={},
    )
    state.save(tmp_path)
    restored = RoboTTTCheckpointState.load(tmp_path)

    for kwargs in (
        {"stage": "stage2", "manifest_sha256": "b" * 64},
        {"stage": "stage1", "manifest_sha256": "c" * 64},
    ):
        try:
            restored.validate(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"incompatible resume state accepted: {kwargs}")


def test_launcher_maps_stage2_to_full_tuning_and_single_gpu_zero3(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"episode_id": "ep-1"}\n')
    launch = RoboTTTLaunchConfig(
        stage="stage2",
        base_model_path="stage1-checkpoint",
        dataset_path="robocasa365-lerobot",
        manifest_path=str(manifest),
        embodiment_tag="panda_omron",
    )

    config = build_robottt_config(launch)

    assert config.model.robottt_enabled is True
    assert config.data.sequence_mode is True
    assert config.data.context_length == 1024
    assert config.training.robottt_stage == "stage2"
    assert config.training.max_steps == 20_000
    assert config.training.learning_rate == 5e-5
    assert config.training.deepspeed_config_path.endswith("robottt_zero3_offload.json")
    assert config.training.num_gpus == 1
    assert config.training.robottt_manifest_hash
    assert config.model.load_bf16 is False
    assert config.model.backbone_trainable_params_fp32 is True


def test_launcher_loads_frozen_stage1_base_in_bf16(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"episode_id": "ep-1"}\n')
    config = build_robottt_config(
        RoboTTTLaunchConfig(
            stage="stage1",
            base_model_path="public-groot",
            dataset_path="robocasa365-lerobot",
            manifest_path=str(manifest),
        )
    )

    assert config.model.load_bf16 is True
    assert config.model.backbone_trainable_params_fp32 is False
    assert config.model.robottt_backbone_micro_batch_size == 8
    assert config.model.robottt_analytic_inner_update is True
    assert config.model.robottt_compile_inner_update is True


def test_stage1_launcher_uses_wsd_and_honors_quick_overrides(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"episode_id": "ep-1"}\n')

    config = build_robottt_config(
        RoboTTTLaunchConfig(
            stage="stage1",
            base_model_path="public-groot",
            dataset_path="robocasa365-lerobot",
            manifest_path=str(manifest),
            max_steps=100,
            context_length=128,
        )
    )

    assert config.training.lr_scheduler_type == SchedulerType.WARMUP_STABLE_DECAY.value
    assert config.training.max_steps == 100
    assert config.training.robottt_wsd_decay_steps == 10
    assert config.data.context_length == 128
    assert config.data.sequence_stride == 128


def test_one_step_smoke_has_a_valid_wsd_schedule(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"episode_id": "ep-1"}\n')

    config = build_robottt_config(
        RoboTTTLaunchConfig(
            stage="stage1",
            base_model_path="public-groot",
            dataset_path="robocasa365-lerobot",
            manifest_path=str(manifest),
            max_steps=1,
            context_length=128,
        )
    )

    assert config.training.warmup_ratio == 0
    assert config.training.robottt_wsd_decay_steps == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (("max_steps", 0), ("context_length", 0)),
)
def test_launcher_rejects_nonpositive_quick_overrides(tmp_path, field, value):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"episode_id": "ep-1"}\n')

    kwargs = {
        "stage": "stage1",
        "base_model_path": "public-groot",
        "dataset_path": "robocasa365-lerobot",
        "manifest_path": str(manifest),
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        build_robottt_config(RoboTTTLaunchConfig(**kwargs))


def test_stage1_launcher_preserves_effective_batch_across_eight_ranks(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"episode_id": "ep-1"}\n')

    config = build_robottt_config(
        RoboTTTLaunchConfig(
            stage="stage1",
            base_model_path="public-groot",
            dataset_path="robocasa365-lerobot",
            manifest_path=str(manifest),
            num_gpus=8,
            gradient_accumulation_steps=1,
        )
    )

    assert config.training.num_gpus == 8
    assert config.training.global_batch_size == 8
    assert config.training.global_batch_size // config.training.num_gpus == 1
    assert config.training.gradient_accumulation_steps == 1
    assert config.training.accumulated_batch_size == 8
    assert config.training.use_ddp is True


def test_stage2_disables_frozen_backbone_cache_but_keeps_compiled_inner_update(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"episode_id": "ep-1"}\n')

    config = build_robottt_config(
        RoboTTTLaunchConfig(
            stage="stage2",
            base_model_path="stage1-checkpoint",
            dataset_path="robocasa365-lerobot",
            manifest_path=str(manifest),
        )
    )

    assert config.model.robottt_backbone_micro_batch_size is None
    assert config.model.robottt_analytic_inner_update is True
    assert config.model.robottt_compile_inner_update is True


def test_trajectory_sampler_state_restores_exact_next_shard():
    mixture = ShardedMixtureDataset.__new__(ShardedMixtureDataset)
    mixture.datasets = [TrajectorySequenceDataset.__new__(TrajectorySequenceDataset)]
    mixture.seed = 42
    mixture.epoch = 3
    mixture.curr_shard_index = 16
    mixture.world_size = 1
    mixture.generate_shard_sampling_schedule = lambda: [(0, index) for index in range(100)]

    state = mixture.state_dict()
    mixture.load_state_dict(state)

    assert state == {"seed": 42, "epoch": 3, "next_shard_index": 17}
    assert mixture.curr_shard_index == 16
    assert mixture._resume_next_shard_index == 17


def test_tbptt_slicer_preserves_temporal_and_flat_vlm_alignment():
    inputs = {
        "inputs": {
            "trajectory_shape": torch.tensor([1, 4]),
            "state": torch.arange(4).reshape(1, 4, 1),
            "input_ids": torch.arange(4).reshape(4, 1),
            "pixel_values": torch.arange(8).reshape(8, 1),
            "episode_id": torch.tensor([9]),
        }
    }

    segment = RoboTTTTrainer._slice_trajectory_inputs(inputs, 1, 3)["inputs"]

    assert segment["trajectory_shape"].tolist() == [1, 2]
    assert segment["state"].flatten().tolist() == [1, 2]
    assert segment["input_ids"].flatten().tolist() == [1, 2]
    assert segment["pixel_values"].flatten().tolist() == [2, 3, 4, 5]
    assert segment["episode_id"].tolist() == [9]


class _GenerationState:
    def __init__(self, generation):
        self.generation = generation

    def detach(self):
        return _GenerationState(self.generation + 1)


class _CountingFrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = []

    def forward(self, inputs):
        ids = inputs.input_ids
        self.calls.append(ids.flatten().tolist())
        batch = ids.shape[0]
        return BatchFeature(
            data={
                "backbone_features": ids.float().reshape(batch, 1, 1),
                "backbone_attention_mask": torch.ones(batch, 1, dtype=torch.bool),
                "image_mask": torch.zeros(batch, 1, dtype=torch.bool),
            }
        )


class _CachedTrainingModel(nn.Module):
    def __init__(self, micro_batch_size=3):
        super().__init__()
        self.config = SimpleNamespace(
            robottt_tbptt_steps=1,
            robottt_backbone_micro_batch_size=micro_batch_size,
        )
        self.backbone = _CountingFrozenBackbone()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.action_calls = []

    def prepare_input(self, payload):
        backbone = BatchFeature(data={"input_ids": payload["input_ids"]})
        action = BatchFeature(
            data={
                "trajectory_shape": payload["trajectory_shape"],
                "state": payload["state"],
            }
        )
        return backbone, action

    @staticmethod
    def _reshape_sequence_backbone_output(output, batch_size, trajectory_length):
        flat_batch = batch_size * trajectory_length
        for key, value in list(output.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == flat_batch:
                output[key] = value.reshape(batch_size, trajectory_length, *value.shape[1:])
        return output

    def forward(
        self,
        inputs=None,
        robottt_state=None,
        cached_backbone_output=None,
        cached_action_input=None,
    ):
        if cached_backbone_output is None or cached_action_input is None:
            raise AssertionError("cached training must use the cached model branch")
        position = int(cached_action_input.state.flatten()[0])
        prior_generation = None if robottt_state is None else robottt_state.generation
        self.action_calls.append((position, prior_generation))
        next_generation = 10 if robottt_state is None else robottt_state.generation + 10
        return BatchFeature(
            data={
                "loss": self.scale * (position + 1),
                "robottt_state": _GenerationState(next_generation),
            }
        )


class _TestAccelerator:
    distributed_type = SimpleNamespace(value="NO")

    def __init__(self):
        self.backward_calls = 0

    def unwrap_model(self, model):
        return model

    def backward(self, loss, **_kwargs):
        self.backward_calls += 1
        loss.backward()


def _cached_trainer():
    trainer = object.__new__(RoboTTTTrainer)
    trainer.args = SimpleNamespace(device=torch.device("cpu"))
    trainer.current_gradient_accumulation_steps = 1
    trainer.accelerator = _TestAccelerator()
    trainer.compute_loss_context_manager = nullcontext
    trainer._prepare_inputs = lambda value: value
    return trainer


def _seven_step_inputs():
    return {
        "inputs": {
            "trajectory_shape": torch.tensor([1, 7]),
            "input_ids": torch.arange(7).reshape(7, 1),
            "state": torch.arange(7).reshape(1, 7, 1),
        }
    }


def test_cached_backbone_is_micro_batched_once_then_consumed_in_order():
    trainer = _cached_trainer()
    model = _CachedTrainingModel(micro_batch_size=3)

    loss = trainer.training_step(model, _seven_step_inputs())

    assert model.backbone.calls == [[0, 1, 2], [3, 4, 5], [6]]
    assert model.action_calls == [
        (0, None),
        (1, 11),
        (2, 22),
        (3, 33),
        (4, 44),
        (5, 55),
        (6, 66),
    ]
    torch.testing.assert_close(loss, torch.tensor(4.0))
    torch.testing.assert_close(model.scale.grad, torch.tensor(4.0))
    assert trainer.accelerator.backward_calls == 7


def test_cached_backbone_rejects_trainable_parameters():
    trainer = _cached_trainer()
    model = _CachedTrainingModel(micro_batch_size=3)
    model.backbone.anchor.requires_grad_(True)

    with pytest.raises(RuntimeError, match="requires a fully frozen backbone"):
        trainer.training_step(model, _seven_step_inputs())
