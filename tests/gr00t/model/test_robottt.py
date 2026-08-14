import math

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.robottt import FastMLPState, RoboTTTLayer, apply_temporal_rope
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


def test_temporal_rope_matches_hand_computed_rotation():
    tokens = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]], [[1.0, 0.0, 1.0, 0.0]]]])
    positions = torch.tensor([[0, 1]])

    rotated = apply_temporal_rope(tokens, positions, theta=10000.0)

    expected = torch.tensor(
        [
            [
                [[1.0, 0.0, 1.0, 0.0]],
                [[math.cos(1.0), math.sin(1.0), math.cos(0.01), math.sin(0.01)]],
            ]
        ]
    )
    torch.testing.assert_close(rotated, expected)


def _manual_fast_forward(tokens: torch.Tensor, state: FastMLPState) -> torch.Tensor:
    hidden = torch.nn.functional.gelu(
        torch.einsum("bnd,bdh->bnh", tokens, state.w1) + state.b1[:, None]
    )
    return torch.einsum("bnh,bhd->bnd", hidden, state.w2) + state.b2[:, None]


def test_step_updates_before_applying_fast_model():
    layer = RoboTTTLayer(dim=2, inner_dim=2, inner_lr=0.1, gate_init=0.5)
    with torch.no_grad():
        identity = torch.eye(2)
        layer.q_proj.weight.copy_(identity)
        layer.k_proj.weight.copy_(identity)
        layer.v_proj.weight.copy_(identity)
        layer.w0_w1.copy_(identity)
        layer.w0_b1.zero_()
        layer.w0_w2.copy_(identity)
        layer.w0_b2.zero_()

    tokens = torch.tensor([[[1.0, -0.5]]])
    initial = layer.initial_state(batch_size=1)

    output, updated, _ = layer.step(
        tokens,
        initial,
        positions=torch.tensor([0]),
        update_mask=torch.tensor([True]),
    )

    post_update_value = _manual_fast_forward(tokens, updated)
    pre_update_value = _manual_fast_forward(tokens, initial)
    expected = tokens + 0.5 * post_update_value
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, tokens + 0.5 * pre_update_value)


def test_outer_loss_reaches_w0_qkv_gate_and_learning_rate():
    layer = RoboTTTLayer(dim=4, inner_dim=6)
    tokens = torch.randn(2, 3, 4)

    output, _, metrics = layer.step(
        tokens,
        layer.initial_state(batch_size=2),
        positions=torch.tensor([2, 5]),
        update_mask=torch.tensor([True, True]),
    )
    (output.square().mean() + metrics["inner_loss"]).backward()

    parameters = [
        layer.w0_w1,
        layer.q_proj.weight,
        layer.k_proj.weight,
        layer.v_proj.weight,
        layer.gate,
        layer.inner_lr_log_multiplier,
    ]
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_scan_does_not_apply_or_update_on_padding():
    layer = RoboTTTLayer(dim=4, inner_dim=6, gate_init=0.2)
    tokens = torch.randn(1, 3, 2, 4)
    initial = layer.initial_state(batch_size=1)

    output, final_state, metrics = layer.scan(
        tokens,
        initial,
        positions=torch.tensor([[0, 1, 2]]),
        valid_mask=torch.tensor([[True, False, True]]),
        update_mask=torch.tensor([[True, True, False]]),
    )

    torch.testing.assert_close(output[:, 1], tokens[:, 1])
    assert metrics["num_updates"].item() == 1
    assert any(not torch.equal(a, b) for a, b in zip(initial.tensors(), final_state.tensors()))
