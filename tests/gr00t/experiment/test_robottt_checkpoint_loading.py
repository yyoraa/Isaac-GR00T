import math
from types import SimpleNamespace

from gr00t.model.gr00t_n1d7 import setup as _setup
from gr00t.model.modules.robottt import RoboTTTLayer
import torch
from torch import nn


class _FilteredConfig:
    def to_filtered_json(self):
        return "{}"


def _pipeline(tmp_path):
    pipeline = object.__new__(_setup.Gr00tN1d7Pipeline)
    pipeline.save_cfg_dir = tmp_path
    pipeline.transformers_loading_kwargs = {}
    pipeline.config = SimpleNamespace(
        training=SimpleNamespace(start_from_checkpoint="public-groot", skip_weight_loading=False),
        model=SimpleNamespace(
            tune_llm=False,
            tune_visual=False,
            tune_projector=True,
            tune_diffusion_model=True,
            tune_vlln=True,
            state_dropout_prob=0.0,
            backbone_trainable_params_fp32=True,
            load_bf16=True,
            robottt_enabled=True,
            robottt_num_register_tokens=16,
            robottt_inner_dim=3072,
            robottt_inner_lr=0.1,
            robottt_rope_theta=10_000.0,
            robottt_gate_init=0.001,
            robottt_tbptt_steps=1,
            robottt_backbone_micro_batch_size=8,
            robottt_analytic_inner_update=True,
            robottt_compile_inner_update=True,
        ),
    )
    return pipeline


def test_checkpoint_load_enables_robottt_without_overriding_checkpoint_depth(
    monkeypatch, tmp_path
):
    captured = {}
    model = nn.Module()
    model.anchor = nn.Parameter(torch.zeros(()))
    model.action_head = nn.Module()
    model.action_head.register_parameter("mask_token", None)
    model.action_head.register_tokens = nn.Parameter(torch.full((16, 4), float("nan")))
    model.config = _FilteredConfig()

    def fake_from_pretrained(path, **kwargs):
        captured.update(kwargs)
        return model, {
            "missing_keys": [
                "action_head.register_tokens",
            ],
            "unexpected_keys": [],
            "mismatched_keys": [],
        }

    monkeypatch.setattr(_setup.AutoModel, "from_pretrained", fake_from_pretrained)
    pipeline = _pipeline(tmp_path)

    assert pipeline._create_model() is model
    assert captured["robottt_enabled"] is True
    assert captured["robottt_num_register_tokens"] == 16
    assert captured["robottt_inner_dim"] == 3072
    assert captured["robottt_inner_lr"] == 0.1
    assert captured["robottt_rope_theta"] == 10_000.0
    assert captured["robottt_gate_init"] == 0.001
    assert captured["robottt_tbptt_steps"] == 1
    assert captured["robottt_backbone_micro_batch_size"] == 8
    assert captured["robottt_analytic_inner_update"] is True
    assert captured["robottt_compile_inner_update"] is True
    assert "diffusion_model_cfg" not in captured


def test_checkpoint_load_reinitializes_missing_robottt_parameters(monkeypatch, tmp_path):
    model = nn.Module()
    model.anchor = nn.Parameter(torch.zeros(()))
    model.action_head = nn.Module()
    model.action_head.register_parameter("mask_token", None)
    model.action_head.register_tokens = nn.Parameter(torch.full((16, 4), float("nan")))
    model.action_head.model = nn.Module()
    block = nn.Module()
    block.robottt = RoboTTTLayer(dim=4, inner_dim=8, gate_init=0.001)
    model.action_head.model.transformer_blocks = nn.ModuleList([block])
    model.config = _FilteredConfig()

    with torch.no_grad():
        for parameter in block.robottt.parameters():
            parameter.fill_(float("nan"))

    missing_keys = ["action_head.register_tokens"] + [
        f"action_head.model.transformer_blocks.0.robottt.{name}"
        for name, _ in block.robottt.named_parameters()
    ]

    monkeypatch.setattr(
        _setup.AutoModel,
        "from_pretrained",
        lambda path, **kwargs: (
            model,
            {"missing_keys": missing_keys, "unexpected_keys": [], "mismatched_keys": []},
        ),
    )

    loaded = _pipeline(tmp_path)._create_model()

    assert torch.isfinite(loaded.action_head.register_tokens).all()
    assert loaded.action_head.register_tokens.abs().sum() > 0
    assert all(torch.isfinite(value).all() for value in block.robottt.parameters())
    assert block.robottt.w0_w1.abs().sum() > 0
    assert block.robottt.q_proj.weight.abs().sum() > 0
    assert torch.count_nonzero(block.robottt.w0_b1) == 0
    assert loaded.action_head.register_tokens.dtype == torch.bfloat16
    assert all(value.dtype == torch.bfloat16 for value in block.robottt.parameters())
    torch.testing.assert_close(
        block.robottt.gate,
        torch.full_like(block.robottt.gate, math.atanh(0.001)),
    )
