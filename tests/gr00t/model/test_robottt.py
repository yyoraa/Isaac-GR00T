from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.robottt import RoboTTTLayer
import torch


def test_robottt_defaults_match_paper():
    config = Gr00tN1d7Config(robottt_enabled=True)

    assert config.robottt_num_register_tokens == 16
    assert config.robottt_inner_dim == 3072
    assert config.robottt_inner_lr == 0.1
    assert config.robottt_rope_theta == 10000.0
    assert config.robottt_gate_init == 0.001


def test_initial_fast_state_is_batched_and_meta_learnable():
    layer = RoboTTTLayer(dim=8, inner_dim=16)

    state = layer.initial_state(batch_size=2)

    assert state.w1.shape == (2, 8, 16)
    assert state.b1.shape == (2, 16)
    assert state.w2.shape == (2, 16, 8)
    assert state.b2.shape == (2, 8)

    state.w1.sum().backward()
    assert layer.w0_w1.grad is not None
    assert torch.count_nonzero(layer.w0_w1.grad) == layer.w0_w1.numel()


def test_detach_preserves_values_and_removes_graph_history():
    layer = RoboTTTLayer(dim=4, inner_dim=6)
    state = layer.initial_state(batch_size=1)

    detached = state.detach()

    for original, result in zip(state.tensors(), detached.tensors()):
        torch.testing.assert_close(result, original)
        assert result.grad_fn is None
        assert result.requires_grad


def test_initial_state_rejects_nonpositive_batch_size():
    layer = RoboTTTLayer(dim=4, inner_dim=6)

    try:
        layer.initial_state(batch_size=0)
    except ValueError as exc:
        assert "batch_size must be positive" in str(exc)
    else:
        raise AssertionError("initial_state accepted an empty batch")
