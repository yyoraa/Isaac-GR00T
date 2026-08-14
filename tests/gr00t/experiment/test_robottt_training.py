from gr00t.configs.robottt_training import (
    RoboTTTTrainingConfig,
    build_wsd_scheduler,
    set_robottt_stage_trainability,
)
import torch
from torch import nn


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


def test_stage_presets_match_public_robottt_schedule():
    stage1 = RoboTTTTrainingConfig.for_stage("stage1")
    stage2 = RoboTTTTrainingConfig.for_stage("stage2")

    assert (stage1.max_steps, stage1.learning_rate, stage1.scheduler) == (
        30_000,
        2e-5,
        "wsd",
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
