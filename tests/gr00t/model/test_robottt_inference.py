from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


class _State:
    def __init__(self, generation=0):
        self.generation = generation

    def detach(self):
        return _State(self.generation)


class _InferenceCapture(nn.Module):
    def __init__(self):
        super().__init__()
        self.update_masks = []
        self.incoming_states = []

    def forward(self, hidden_states, robottt_state=None, update_mask=None, **_kwargs):
        update = bool(update_mask.item())
        self.update_masks.append(update)
        self.incoming_states.append(robottt_state)
        generation = 0 if robottt_state is None else robottt_state.generation
        if update:
            generation += 1
        metrics = {
            "inner_loss": hidden_states.new_zeros(2),
            "num_updates": hidden_states.new_full((2,), int(update), dtype=torch.long),
        }
        return hidden_states, _State(generation), metrics


def _head_and_inputs():
    config = Gr00tN1d7Config(
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
        num_inference_timesteps=4,
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
    head = Gr00tN1d7ActionHead(config).eval()
    capture = _InferenceCapture()
    head.model = capture
    backbone = BatchFeature(
        data={
            "backbone_features": torch.randn(1, 5, 8),
            "backbone_attention_mask": torch.ones(1, 5, dtype=torch.bool),
            "image_mask": torch.ones(1, 5, dtype=torch.bool),
        }
    )
    action = BatchFeature(
        data={
            "state": torch.randn(1, 1, 3),
            "embodiment_id": torch.zeros(1, dtype=torch.long),
        }
    )
    return head, capture, backbone, action


def test_four_denoising_evaluations_update_once_per_observation():
    head, capture, backbone, action = _head_and_inputs()

    result = head.get_action(backbone, action)

    assert result.action_pred.shape == (1, 4, 3)
    assert capture.update_masks == [True, False, False, False]
    assert head.robottt_observation_count == 1
    assert head.robottt_fast_state.generation == 1


def test_fast_state_carries_between_observations_and_reset_clears_it():
    head, capture, backbone, action = _head_and_inputs()

    head.get_action(backbone, action)
    head.get_action(backbone, action)

    assert capture.update_masks == [True, False, False, False] * 2
    assert capture.incoming_states[4].generation == 1
    assert head.robottt_fast_state.generation == 2
    assert head.robottt_observation_count == 2

    head.reset_robottt_state()
    assert head.robottt_fast_state is None
    assert head.robottt_observation_count == 0


def test_update_off_ablation_runs_architecture_without_fast_updates():
    head, capture, backbone, action = _head_and_inputs()
    head.set_robottt_online_updates(False)

    head.get_action(backbone, action)

    assert capture.update_masks == [False, False, False, False]
    assert head.robottt_fast_state.generation == 0
