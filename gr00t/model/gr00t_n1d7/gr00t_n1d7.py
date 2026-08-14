# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from gr00t.model.modules.robottt import RoboTTTState


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
                robottt_enabled=config.robottt_enabled,
                robottt_inner_dim=config.robottt_inner_dim,
                robottt_inner_lr=config.robottt_inner_lr,
                robottt_rope_theta=config.robottt_rope_theta,
                robottt_gate_init=config.robottt_gate_init,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        if config.robottt_enabled:
            self.register_tokens = nn.Parameter(
                torch.empty(config.robottt_num_register_tokens, self.input_embedding_dim)
            )
            nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        else:
            self.register_parameter("register_tokens", None)
        self._robottt_fast_state: RoboTTTState | None = None
        self._robottt_observation_count = 0
        self._robottt_batch_size: int | None = None

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly. The action head can
        # be instantiated under a meta / no_init_weights default-device context
        # (e.g. nested from_pretrained). A Beta built from bare Python floats
        # would then place its concentration tensors on the meta device (or in
        # the active default dtype, e.g. bf16). With validate_args enabled that
        # already fails here in __init__ (Beta's internal .item() check cannot
        # run on meta); even with validation off, sample_time would later raise
        # or return garbage. Explicit device/dtype here makes the sampler depend
        # only on the config, not on the construction-time device/dtype context,
        # so the noise schedule is identical across SDPA/FA2/FA4 and meta vs.
        # real-device loads. config is the canonical source for these values.
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def sample_sequence_time(self, batch_size, trajectory_length, device, dtype):
        """Sample independent flow times for every robot timestep."""
        sample = self.beta_dist.sample([batch_size, trajectory_length]).to(device, dtype=dtype)
        return ((1 - sample) * self.config.noise_s)[:, :, None, None]

    @property
    def robottt_fast_state(self) -> RoboTTTState | None:
        """Current rollout-local fast state, exposed for diagnostics."""
        return self._robottt_fast_state

    @property
    def robottt_observation_count(self) -> int:
        return self._robottt_observation_count

    def reset_robottt_state(self) -> None:
        """Reset online adaptation at an episode boundary."""
        self._robottt_fast_state = None
        self._robottt_observation_count = 0
        self._robottt_batch_size = None

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def forward_sequence(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        robottt_state: RoboTTTState | None = None,
        tbptt_steps: int | None = None,
    ) -> BatchFeature:
        """Train RoboTTT on an episode-aligned sequence of robot timesteps."""
        if not self.config.robottt_enabled:
            raise RuntimeError("forward_sequence requires robottt_enabled=True")
        self.set_frozen_modules_to_eval_mode()

        vl_embeds = backbone_output.backbone_features
        if vl_embeds.ndim != 4:
            raise ValueError(
                "sequence backbone_features must have shape [B,T,S,D], "
                f"got {tuple(vl_embeds.shape)}"
            )
        batch_size, trajectory_length, vl_length, vl_dim = vl_embeds.shape
        flat_batch = batch_size * trajectory_length
        vl_embeds = self.vlln(vl_embeds.reshape(flat_batch, vl_length, vl_dim))
        vl_embeds = self.vl_self_attention(vl_embeds)
        vl_embeds = vl_embeds.reshape(batch_size, trajectory_length, vl_length, vl_dim)

        embodiment_id = action_input.embodiment_id
        expected_time_shape = (batch_size, trajectory_length)
        if embodiment_id.shape != expected_time_shape:
            raise ValueError(
                f"embodiment_id must have shape {expected_time_shape}, "
                f"got {tuple(embodiment_id.shape)}"
            )
        flat_embodiment = embodiment_id.reshape(flat_batch)

        state = action_input.state
        expected_state_shape = (
            batch_size,
            trajectory_length,
            self.config.state_history_length,
            self.config.max_state_dim,
        )
        if state.shape != expected_state_shape:
            raise ValueError(
                f"state must have shape {expected_state_shape}, got {tuple(state.shape)}"
            )
        flat_state = state.reshape(flat_batch, 1, -1)
        state_features = self.state_encoder(flat_state, flat_embodiment).reshape(
            batch_size, trajectory_length, 1, self.input_embedding_dim
        )

        if self.training and self.state_dropout_prob > 0:
            keep = (
                torch.rand(batch_size, trajectory_length, device=state.device)
                >= self.state_dropout_prob
            )
            state_features = state_features * keep[:, :, None, None].to(state_features.dtype)

        actions = action_input.action
        expected_action_shape = (
            batch_size,
            trajectory_length,
            self.action_horizon,
            self.action_dim,
        )
        if actions.shape != expected_action_shape:
            raise ValueError(
                f"action must have shape {expected_action_shape}, got {tuple(actions.shape)}"
            )
        noise = torch.randn_like(actions)
        flow_time = self.sample_sequence_time(
            batch_size, trajectory_length, actions.device, actions.dtype
        )
        noisy_trajectory = (1 - flow_time) * noise + flow_time * actions
        velocity = actions - noise
        timestep = (flow_time[:, :, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(
            noisy_trajectory.reshape(flat_batch, self.action_horizon, self.action_dim),
            timestep.reshape(flat_batch),
            flat_embodiment,
        ).reshape(batch_size, trajectory_length, self.action_horizon, self.input_embedding_dim)

        if self.config.add_pos_embed:
            position_ids = torch.arange(self.action_horizon, device=actions.device)
            action_features = action_features + self.position_embedding(position_ids).view(
                1, 1, self.action_horizon, self.input_embedding_dim
            )

        registers = self.register_tokens.view(
            1, 1, self.config.robottt_num_register_tokens, self.input_embedding_dim
        ).expand(batch_size, trajectory_length, -1, -1)
        sequence_tokens = torch.cat((registers, state_features, action_features), dim=2)
        valid_mask = action_input.get(
            "valid_mask", torch.ones(expected_time_shape, device=actions.device, dtype=torch.bool)
        ).bool()
        temporal_positions = action_input.get(
            "temporal_positions",
            torch.arange(trajectory_length, device=actions.device)
            .view(1, -1)
            .expand(batch_size, -1),
        )

        segment_length = trajectory_length if tbptt_steps is None else tbptt_steps
        if segment_length <= 0:
            raise ValueError(f"tbptt_steps must be positive, got {segment_length}")
        segment_outputs = []
        segment_losses = []
        segment_updates = []
        current_state = robottt_state
        for start in range(0, trajectory_length, segment_length):
            end = min(start + segment_length, trajectory_length)
            segment_output, current_state, segment_metrics = self.model(
                hidden_states=sequence_tokens[:, start:end],
                encoder_hidden_states=vl_embeds[:, start:end],
                timestep=timestep[:, start:end],
                image_mask=backbone_output.image_mask[:, start:end],
                backbone_attention_mask=backbone_output.backbone_attention_mask[:, start:end],
                robottt_state=current_state,
                temporal_positions=temporal_positions[:, start:end],
                valid_mask=valid_mask[:, start:end],
                update_mask=valid_mask[:, start:end],
            )
            segment_outputs.append(segment_output)
            segment_losses.append(segment_metrics["inner_loss"])
            segment_updates.append(segment_metrics["num_updates"])
            if end < trajectory_length:
                current_state = current_state.detach()

        model_output = torch.cat(segment_outputs, dim=1)
        next_state = current_state
        robottt_metrics = {
            "inner_loss": torch.stack(segment_losses).mean(dim=0),
            "num_updates": torch.stack(segment_updates).sum(dim=0),
        }
        flat_output = model_output.reshape(flat_batch, model_output.shape[2], model_output.shape[3])
        prediction = self.action_decoder(flat_output, flat_embodiment).reshape(
            batch_size, trajectory_length, model_output.shape[2], self.action_dim
        )
        pred_actions = prediction[:, :, -self.action_horizon :]

        action_mask = action_input.action_mask
        action_loss_mask = action_input.get("action_loss_mask", valid_mask).bool()
        effective_mask = (
            action_mask
            * valid_mask[:, :, None, None].to(action_mask.dtype)
            * action_loss_mask[:, :, None, None].to(action_mask.dtype)
        )
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * effective_mask
        loss = action_loss.sum() / effective_mask.sum().clamp_min(1)
        return BatchFeature(
            data={
                "loss": loss,
                "action_loss": action_loss,
                "action_mask": effective_mask,
                "pred_actions": pred_actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
                "robottt_state": next_state,
                "robottt_metrics": robottt_metrics,
            }
        )

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        candidate_robottt_state = self._robottt_fast_state
        if self.config.robottt_enabled and self._robottt_batch_size not in (None, batch_size):
            self.reset_robottt_state()
            candidate_robottt_state = None

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join registers, state, and action embeddings along sequence dimension.
            if self.config.robottt_enabled:
                registers = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
                sa_embs = torch.cat((registers, state_features, action_features), dim=1)
            else:
                sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                if self.config.robottt_enabled:
                    model_output, candidate_robottt_state, _ = self.model(
                        hidden_states=sa_embs[:, None],
                        encoder_hidden_states=vl_embeds[:, None],
                        timestep=timesteps_tensor[:, None],
                        image_mask=backbone_output.image_mask[:, None],
                        backbone_attention_mask=backbone_output.backbone_attention_mask[:, None],
                        robottt_state=candidate_robottt_state,
                        temporal_positions=torch.full(
                            (batch_size, 1), self._robottt_observation_count, device=device
                        ),
                        valid_mask=torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                        update_mask=torch.full(
                            (batch_size, 1), t == 0, dtype=torch.bool, device=device
                        ),
                    )
                    model_output = model_output[:, 0]
                else:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        image_mask=backbone_output.image_mask,
                        backbone_attention_mask=backbone_output.backbone_attention_mask,
                    )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        if self.config.robottt_enabled:
            self._robottt_fast_state = candidate_robottt_state.detach()
            self._robottt_observation_count += 1
            self._robottt_batch_size = batch_size

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        if self.config.robottt_enabled and "trajectory_shape" in action_inputs:
            batch_size, trajectory_length = [
                int(value) for value in action_inputs.trajectory_shape.tolist()
            ]
            flat_batch = batch_size * trajectory_length
            for key, value in list(backbone_outputs.items()):
                if isinstance(value, torch.Tensor) and value.shape[0] == flat_batch:
                    backbone_outputs[key] = value.reshape(
                        batch_size, trajectory_length, *value.shape[1:]
                    )
            action_outputs = self.action_head.forward_sequence(
                backbone_outputs,
                action_inputs,
                tbptt_steps=getattr(self.config, "robottt_tbptt_steps", 128),
            )
        else:
            action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
