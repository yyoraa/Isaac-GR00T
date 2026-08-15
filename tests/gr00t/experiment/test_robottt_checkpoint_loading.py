from types import SimpleNamespace

from gr00t.model.gr00t_n1d7 import setup as _setup
import torch
from torch import nn


class _FilteredConfig:
    def to_filtered_json(self):
        return "{}"


def test_checkpoint_load_enables_robottt_without_overriding_checkpoint_depth(monkeypatch, tmp_path):
    captured = {}
    model = nn.Module()
    model.anchor = nn.Parameter(torch.zeros(()))
    model.action_head = nn.Module()
    model.action_head.register_parameter("mask_token", None)
    model.config = _FilteredConfig()

    def fake_from_pretrained(path, **kwargs):
        captured.update(kwargs)
        return model, {
            "missing_keys": [
                "action_head.register_tokens",
                "action_head.model.transformer_blocks.0.robottt.fast_mlp.w1",
            ],
            "unexpected_keys": [],
            "mismatched_keys": [],
        }

    monkeypatch.setattr(_setup.AutoModel, "from_pretrained", fake_from_pretrained)
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
        ),
    )

    assert pipeline._create_model() is model
    assert captured["robottt_enabled"] is True
    assert captured["robottt_num_register_tokens"] == 16
    assert captured["robottt_inner_dim"] == 3072
    assert captured["robottt_inner_lr"] == 0.1
    assert captured["robottt_rope_theta"] == 10_000.0
    assert captured["robottt_gate_init"] == 0.001
    assert captured["robottt_tbptt_steps"] == 1
    assert "diffusion_model_cfg" not in captured
