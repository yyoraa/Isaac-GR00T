from types import SimpleNamespace

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


class _ExplodingBackbone(nn.Module):
    def forward(self, _inputs):
        raise AssertionError("cached forward must not call the backbone")


class _RecordingActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_sequence(self, backbone_output, action_input, robottt_state=None, tbptt_steps=None):
        self.calls.append((backbone_output, action_input, robottt_state, tbptt_steps))
        return BatchFeature(data={"loss": torch.tensor(1.25), "robottt_state": "next-state"})


def _model_without_weights():
    model = object.__new__(Gr00tN1d7)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(robottt_enabled=True, robottt_tbptt_steps=1)
    model.backbone = _ExplodingBackbone()
    model.action_head = _RecordingActionHead()
    return model


def test_cached_forward_skips_backbone_and_carries_fast_state():
    model = _model_without_weights()
    backbone = BatchFeature(
        data={
            "backbone_features": torch.randn(1, 1, 3, 4),
            "backbone_attention_mask": torch.ones(1, 1, 3, dtype=torch.bool),
            "image_mask": torch.ones(1, 1, 3, dtype=torch.bool),
        }
    )
    action = BatchFeature(data={"trajectory_shape": torch.tensor([1, 1])})

    output = model(
        cached_backbone_output=backbone,
        cached_action_input=action,
        robottt_state="prior-state",
    )

    assert output.loss.item() == 1.25
    assert output.robottt_state == "next-state"
    assert model.action_head.calls == [(backbone, action, "prior-state", 1)]


def test_cached_forward_rejects_partial_or_mixed_inputs():
    model = _model_without_weights()
    backbone = BatchFeature(data={})
    action = BatchFeature(data={})

    with pytest.raises(ValueError, match="provided together"):
        model(cached_backbone_output=backbone)
    with pytest.raises(ValueError, match="mutually exclusive"):
        model(inputs={}, cached_backbone_output=backbone, cached_action_input=action)


def test_sequence_backbone_reshape_preserves_flattened_order():
    flat = torch.arange(24).reshape(6, 2, 2)
    output = BatchFeature(
        data={
            "backbone_features": flat.clone(),
            "backbone_attention_mask": torch.ones(6, 2, dtype=torch.bool),
            "image_mask": torch.zeros(6, 2, dtype=torch.bool),
            "unchanged": torch.tensor([9]),
        }
    )

    reshaped = Gr00tN1d7._reshape_sequence_backbone_output(output, 2, 3)

    assert reshaped.backbone_features.shape == (2, 3, 2, 2)
    torch.testing.assert_close(reshaped.backbone_features.reshape(6, 2, 2), flat)
    assert reshaped.unchanged.tolist() == [9]
