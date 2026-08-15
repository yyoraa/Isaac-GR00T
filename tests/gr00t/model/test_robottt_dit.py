import types

from gr00t.model.modules.dit import AlternateVLDiT, BasicTransformerBlock
import torch
from torch import nn


class _RecordingTTT(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def scan(self, tokens, state, positions, valid_mask, update_mask):
        self.inputs.append(tokens.detach().clone())
        return (
            tokens * 2,
            "next-state",
            {"inner_loss": tokens.new_zeros(()), "num_updates": valid_mask.sum()},
        )


def _small_block() -> BasicTransformerBlock:
    return BasicTransformerBlock(
        dim=4,
        num_attention_heads=1,
        attention_head_dim=4,
        dropout=0.0,
        norm_type="layer_norm",
        final_dropout=False,
    )


def test_legacy_forward_composes_attention_then_feed_forward():
    block = _small_block().eval()
    tokens = torch.randn(2, 3, 4)

    expected = block.forward_feed_forward(block.forward_attention(tokens))
    actual = block(tokens)

    torch.testing.assert_close(actual, expected)


def test_sequence_forward_places_robottt_between_attention_and_feed_forward():
    block = _small_block()
    recorder = _RecordingTTT()
    block.robottt = recorder

    def attention(_self, hidden_states, **_kwargs):
        return hidden_states + 1

    def feed_forward(_self, hidden_states):
        return hidden_states - 3

    block.forward_attention = types.MethodType(attention, block)
    block.forward_feed_forward = types.MethodType(feed_forward, block)
    tokens = torch.zeros(1, 2, 3, 4)

    output, state, metrics = block.forward_sequence(
        tokens,
        robottt_state="initial-state",
        temporal_positions=torch.tensor([[0, 1]]),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        update_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(recorder.inputs[0], torch.ones_like(tokens))
    torch.testing.assert_close(output, torch.full_like(tokens, -1))
    assert state == "next-state"
    assert metrics["num_updates"].item() == 2


def test_alternate_vl_dit_builds_robottt_in_every_layer():
    model = AlternateVLDiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=4,
        dropout=0.0,
        attention_bias=True,
        activation_fn="gelu",
        norm_type="ada_norm",
        norm_elementwise_affine=False,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=True,
        cross_attention_dim=8,
        robottt_enabled=True,
        robottt_inner_dim=16,
        robottt_analytic_inner_update=True,
        robottt_compile_inner_update=True,
    )

    assert len(model.transformer_blocks) == 4
    assert all(block.robottt is not None for block in model.transformer_blocks)
    assert all(block.robottt.analytic_inner_update for block in model.transformer_blocks)
    assert all(block.robottt.compile_inner_update for block in model.transformer_blocks)
    state = model.initial_robottt_state(batch_size=2)
    assert len(state.layers) == 4
    assert all(layer_state.w1.shape == (2, 8, 16) for layer_state in state.layers)


def test_disabled_dit_has_no_robottt_parameters():
    model = AlternateVLDiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        attention_bias=True,
        activation_fn="gelu",
        norm_type="ada_norm",
        norm_elementwise_affine=False,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=True,
        cross_attention_dim=8,
        robottt_enabled=False,
    )

    assert all(block.robottt is None for block in model.transformer_blocks)
    assert not any("robottt" in name for name, _ in model.named_parameters())


def test_alternate_vl_dit_scans_robot_time_in_all_layers():
    model = AlternateVLDiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=4,
        dropout=0.0,
        attention_bias=True,
        activation_fn="gelu",
        norm_type="ada_norm",
        norm_elementwise_affine=False,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=True,
        cross_attention_dim=8,
        robottt_enabled=True,
        robottt_inner_dim=16,
    ).eval()
    hidden = torch.randn(1, 2, 3, 8)
    encoder = torch.randn(1, 2, 4, 8)
    timestep = torch.zeros(1, 2, dtype=torch.long)
    image_mask = torch.tensor([[[True, True, False, False], [True, True, False, False]]])
    attention_mask = torch.ones(1, 2, 4, dtype=torch.bool)
    state = model.initial_robottt_state(batch_size=1)

    output, next_state, metrics = model(
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        timestep=timestep,
        image_mask=image_mask,
        backbone_attention_mask=attention_mask,
        robottt_state=state,
        temporal_positions=torch.tensor([[7, 8]]),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        update_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    assert output.shape == (1, 2, 3, 8)
    assert len(next_state.layers) == 4
    torch.testing.assert_close(metrics["num_updates"], torch.full((4,), 2))
