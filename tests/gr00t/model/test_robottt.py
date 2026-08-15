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

    normalized = torch.nn.functional.layer_norm(tokens, (2,))
    post_update_value = _manual_fast_forward(normalized, updated)
    pre_update_value = _manual_fast_forward(normalized, initial)
    expected = tokens + 0.5 * post_update_value
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, tokens + 0.5 * pre_update_value)


def test_inner_update_normalizes_large_backbone_residuals():
    torch.manual_seed(0)
    layer = RoboTTTLayer(dim=8, inner_dim=16, inner_lr=0.1)
    state = layer.initial_state(batch_size=1)

    for position in range(2):
        tokens = torch.randn(1, 4, 8) * 1_000
        output, state, metrics = layer.step(
            tokens,
            state,
            positions=torch.tensor([position]),
            update_mask=torch.tensor([True]),
        )

        assert torch.isfinite(output).all()
        assert torch.isfinite(metrics["inner_loss"])
        assert all(torch.isfinite(value).all() for value in state.tensors())
        assert max(value.detach().abs().max() for value in state.tensors()) < 10


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


def _run_step_and_backward(layer, tokens, positions, update_mask):
    output, state, metrics = layer.step(
        tokens,
        layer.initial_state(tokens.shape[0]),
        positions=positions,
        update_mask=update_mask,
    )
    (output.square().mean() + metrics["inner_loss"]).backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in layer.named_parameters()
        if parameter.grad is not None
    }
    return output, state, metrics, gradients


def test_analytic_inner_update_matches_autograd_reference():
    torch.manual_seed(19)
    reference = RoboTTTLayer(dim=4, inner_dim=7, gate_init=0.2).double().train()
    analytic = RoboTTTLayer(
        dim=4,
        inner_dim=7,
        gate_init=0.2,
        analytic_inner_update=True,
    ).double().train()
    analytic.load_state_dict(reference.state_dict())
    tokens = torch.randn(2, 3, 4, dtype=torch.float64)
    positions = torch.tensor([2, 5])
    mask = torch.tensor([True, False])

    reference_result = reference.step(tokens, reference.initial_state(2), positions, mask)
    analytic_result = analytic.step(tokens, analytic.initial_state(2), positions, mask)

    torch.testing.assert_close(analytic_result[0], reference_result[0], rtol=1e-10, atol=1e-10)
    for actual, expected in zip(analytic_result[1].tensors(), reference_result[1].tensors()):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        analytic_result[2]["inner_loss"],
        reference_result[2]["inner_loss"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert analytic_result[2]["num_updates"].item() == 1


def test_analytic_outer_gradients_match_autograd_reference():
    torch.manual_seed(29)
    reference = RoboTTTLayer(dim=4, inner_dim=7, gate_init=0.2).double().train()
    analytic = RoboTTTLayer(
        dim=4,
        inner_dim=7,
        gate_init=0.2,
        analytic_inner_update=True,
    ).double().train()
    analytic.load_state_dict(reference.state_dict())
    tokens = torch.randn(2, 3, 4, dtype=torch.float64)
    positions = torch.tensor([2, 5])
    mask = torch.tensor([True, False])

    reference_result = _run_step_and_backward(reference, tokens, positions, mask)
    analytic_result = _run_step_and_backward(analytic, tokens, positions, mask)

    assert analytic_result[3].keys() == reference_result[3].keys()
    for name in reference_result[3]:
        torch.testing.assert_close(
            analytic_result[3][name], reference_result[3][name], rtol=1e-9, atol=1e-10
        )


def test_analytic_all_masked_step_preserves_state_bitwise():
    layer = RoboTTTLayer(dim=4, inner_dim=7, analytic_inner_update=True).double().train()
    state = layer.initial_state(2)
    tokens = torch.randn(2, 3, 4, dtype=torch.float64)

    _, updated, metrics = layer.step(
        tokens,
        state,
        positions=torch.tensor([0, 1]),
        update_mask=torch.tensor([False, False]),
    )

    assert all(torch.equal(actual, expected) for actual, expected in zip(updated.tensors(), state.tensors()))
    assert metrics["num_updates"].item() == 0


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
