from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


def _config() -> Gr00tN1d7Config:
    return Gr00tN1d7Config(
        backbone_embedding_dim=8,
        hidden_size=8,
        input_embedding_dim=8,
        max_state_dim=3,
        max_action_dim=3,
        action_horizon=4,
        state_history_length=1,
        max_num_embodiments=2,
        add_pos_embed=True,
        max_seq_len=8,
        use_vlln=True,
        use_alternate_vl_dit=True,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        robottt_enabled=True,
        robottt_num_register_tokens=16,
        robottt_inner_dim=16,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 8,
            "interleave_self_attention": True,
        },
    )


class _CaptureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_states = None

    def forward(self, hidden_states, robottt_state=None, **_kwargs):
        self.hidden_states = hidden_states.detach().clone()
        metrics = {
            "inner_loss": hidden_states.new_zeros(2),
            "num_updates": hidden_states.new_zeros(2, dtype=torch.long),
        }
        return hidden_states, robottt_state, metrics


def _inputs(config: Gr00tN1d7Config):
    backbone = BatchFeature(
        data={
            "backbone_features": torch.randn(1, 2, 5, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(1, 2, 5, dtype=torch.bool),
            "image_mask": torch.ones(1, 2, 5, dtype=torch.bool),
        }
    )
    action = BatchFeature(
        data={
            "state": torch.randn(1, 2, 1, config.max_state_dim),
            "action": torch.randn(1, 2, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(1, 2, dtype=torch.long),
            "action_mask": torch.ones(1, 2, config.action_horizon, config.max_action_dim),
            "valid_mask": torch.ones(1, 2, dtype=torch.bool),
            "action_loss_mask": torch.ones(1, 2, dtype=torch.bool),
            "temporal_positions": torch.tensor([[4, 5]]),
        }
    )
    return backbone, action


def test_action_head_owns_exactly_sixteen_register_tokens():
    config = _config()
    head = Gr00tN1d7ActionHead(config)

    assert head.register_tokens.shape == (16, 8)
    assert head.register_tokens.requires_grad


def test_sequence_time_is_independent_across_robot_timesteps():
    head = Gr00tN1d7ActionHead(_config())
    torch.manual_seed(7)

    sampled = head.sample_sequence_time(2, 4, device=torch.device("cpu"), dtype=torch.float32)

    assert sampled.shape == (2, 4, 1, 1)
    assert torch.unique(sampled).numel() > 1


def test_forward_sequence_prepends_registers_and_returns_only_action_loss():
    config = _config()
    head = Gr00tN1d7ActionHead(config)
    capture = _CaptureModel()
    head.model = capture
    backbone, action = _inputs(config)

    output = head.forward_sequence(backbone, action)

    assert capture.hidden_states.shape == (1, 2, 16 + 1 + 4, 8)
    expected_registers = head.register_tokens.view(1, 1, 16, 8).expand(1, 2, 16, 8)
    torch.testing.assert_close(capture.hidden_states[:, :, :16], expected_registers)
    assert output["action_loss"].shape == (1, 2, 4, 3)
    assert output["pred_actions"].shape == (1, 2, 4, 3)


def test_action_loss_mask_excludes_context_only_timestep():
    config = _config()
    head = Gr00tN1d7ActionHead(config)
    head.model = _CaptureModel()
    backbone, action = _inputs(config)
    action.action_loss_mask[:, 0] = False
    torch.manual_seed(11)
    first = head.forward_sequence(backbone, action)["loss"]
    action.action[:, 0] = 10000
    torch.manual_seed(11)
    second = head.forward_sequence(backbone, action)["loss"]

    torch.testing.assert_close(first, second)


class _DetachableState:
    def __init__(self, generation=0):
        self.generation = generation

    def detach(self):
        return _DetachableState(self.generation + 1)


class _SegmentCapture(_CaptureModel):
    def __init__(self):
        super().__init__()
        self.seen_states = []

    def forward(self, hidden_states, robottt_state=None, **kwargs):
        self.seen_states.append(robottt_state)
        output, _, metrics = super().forward(hidden_states, robottt_state, **kwargs)
        return output, _DetachableState(len(self.seen_states)), metrics


def test_tbptt_carries_and_detaches_fast_state_between_segments():
    config = _config()
    head = Gr00tN1d7ActionHead(config)
    capture = _SegmentCapture()
    head.model = capture
    backbone, action = _inputs(config)

    output = head.forward_sequence(backbone, action, tbptt_steps=1)

    assert len(capture.seen_states) == 2
    assert capture.seen_states[0] is None
    assert capture.seen_states[1].generation == 2
    assert output["robottt_state"].generation == 2
