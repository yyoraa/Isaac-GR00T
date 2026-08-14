# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fast-weight state used by RoboTTT sequence adaptation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


def apply_temporal_rope(
    tokens: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0
) -> torch.Tensor:
    """Rotate feature pairs using robot-timestep positions.

    ``tokens`` has shape ``[..., num_tokens, dim]`` and ``positions`` has the
    matching leading shape ``[...]``. All tokens from one robot timestep use
    the same temporal position.
    """
    dim = tokens.shape[-1]
    if dim % 2:
        raise ValueError(f"RoboTTT RoPE requires an even feature dimension, got {dim}")
    if tuple(positions.shape) != tuple(tokens.shape[:-2]):
        raise ValueError(
            "positions must match token leading dimensions: "
            f"got {tuple(positions.shape)} for {tuple(tokens.shape)}"
        )
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")

    frequencies = theta ** (
        -torch.arange(0, dim, 2, device=tokens.device, dtype=torch.float32) / dim
    )
    angles = positions.to(device=tokens.device, dtype=torch.float32).unsqueeze(-1) * frequencies
    cos = angles.cos().to(tokens.dtype).unsqueeze(-2)
    sin = angles.sin().to(tokens.dtype).unsqueeze(-2)
    even = tokens[..., 0::2]
    odd = tokens[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


@dataclass(frozen=True)
class FastMLPState:
    """Per-example parameters for a two-layer fast MLP."""

    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the state in stable optimizer order."""
        return self.w1, self.b1, self.w2, self.b2

    def detach(self) -> FastMLPState:
        """Truncate history while retaining inner-loop differentiability."""

        def detach_tensor(value: torch.Tensor) -> torch.Tensor:
            return value.detach().requires_grad_(value.requires_grad)

        return FastMLPState(*(detach_tensor(value) for value in self.tensors()))


@dataclass(frozen=True)
class RoboTTTState:
    """Fast states for all action-transformer layers."""

    layers: tuple[FastMLPState, ...]

    def detach(self) -> RoboTTTState:
        """Truncate temporal history for every layer at a TBPTT boundary."""
        return RoboTTTState(tuple(layer.detach() for layer in self.layers))


class RoboTTTLayer(nn.Module):
    """One RoboTTT layer with a learned initial fast model."""

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        inner_lr: float = 0.1,
        rope_theta: float = 10000.0,
        gate_init: float = 0.001,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if inner_dim <= 0:
            raise ValueError(f"inner_dim must be positive, got {inner_dim}")
        if inner_lr <= 0:
            raise ValueError(f"inner_lr must be positive, got {inner_lr}")
        if not -1.0 < gate_init < 1.0:
            raise ValueError(f"gate_init must be between -1 and 1, got {gate_init}")

        self.dim = dim
        self.inner_dim = inner_dim
        self.inner_lr = inner_lr
        self.rope_theta = rope_theta
        self.w0_w1 = nn.Parameter(torch.empty(dim, inner_dim))
        self.w0_b1 = nn.Parameter(torch.zeros(inner_dim))
        self.w0_w2 = nn.Parameter(torch.empty(inner_dim, dim))
        self.w0_b2 = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.w0_w1)
        nn.init.xavier_uniform_(self.w0_w2)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.inner_lr_log_multiplier = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.gate = nn.Parameter(torch.full((dim,), math.atanh(gate_init)))

    def initial_state(self, batch_size: int) -> FastMLPState:
        """Expand learned W0 into independent fast weights for each example."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        def expand(value: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                return value.unsqueeze(0).expand(batch_size, *value.shape).clone()

        return FastMLPState(
            w1=expand(self.w0_w1),
            b1=expand(self.w0_b1),
            w2=expand(self.w0_w2),
            b2=expand(self.w0_b2),
        )

    @staticmethod
    def _fast_forward(tokens: torch.Tensor, state: FastMLPState) -> torch.Tensor:
        hidden = F.gelu(torch.einsum("bnd,bdh->bnh", tokens, state.w1) + state.b1[:, None])
        return torch.einsum("bnh,bhd->bnd", hidden, state.w2) + state.b2[:, None]

    @staticmethod
    def _blend_state(
        updated: FastMLPState, previous: FastMLPState, mask: torch.Tensor
    ) -> FastMLPState:
        values = []
        for new_value, old_value in zip(updated.tensors(), previous.tensors()):
            broadcast_mask = mask.reshape(mask.shape[0], *([1] * (new_value.ndim - 1)))
            values.append(torch.where(broadcast_mask, new_value, old_value))
        return FastMLPState(*values)

    def step(
        self,
        tokens: torch.Tensor,
        state: FastMLPState,
        positions: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, FastMLPState, dict[str, torch.Tensor]]:
        """Update the fast MLP from K/V, then apply it to Q."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens must have shape [B,N,{self.dim}], got {tuple(tokens.shape)}")
        batch_size = tokens.shape[0]
        if positions.shape != (batch_size,):
            raise ValueError(
                f"positions must have shape [{batch_size}], got {tuple(positions.shape)}"
            )
        if update_mask.shape != (batch_size,):
            raise ValueError(
                f"update_mask must have shape [{batch_size}], got {tuple(update_mask.shape)}"
            )
        update_mask = update_mask.to(device=tokens.device, dtype=torch.bool)

        query = apply_temporal_rope(self.q_proj(tokens), positions, self.rope_theta)
        key = apply_temporal_rope(self.k_proj(tokens), positions, self.rope_theta)
        value = self.v_proj(tokens)

        with torch.enable_grad():
            prediction = self._fast_forward(key, state)
            per_example_loss = (prediction - value).square().mean(dim=(1, 2))
            if not torch.isfinite(per_example_loss).all():
                raise FloatingPointError("RoboTTT inner loss contains NaN or Inf")

            if update_mask.any():
                objective = (per_example_loss * update_mask).sum()
                gradients = torch.autograd.grad(
                    objective,
                    state.tensors(),
                    create_graph=self.training,
                    allow_unused=False,
                )
                step_size = self.inner_lr * F.softplus(self.inner_lr_log_multiplier)
                candidate = FastMLPState(
                    *(
                        parameter - step_size * gradient
                        for parameter, gradient in zip(state.tensors(), gradients)
                    )
                )
                updated = self._blend_state(candidate, state, update_mask)
            else:
                updated = state

            adapted = self._fast_forward(query, updated)

        output = tokens + torch.tanh(self.gate).view(1, 1, -1) * adapted
        active_count = update_mask.sum()
        mean_inner_loss = (per_example_loss * update_mask).sum() / active_count.clamp_min(1)
        return output, updated, {"inner_loss": mean_inner_loss, "num_updates": active_count}

    def scan(
        self,
        tokens: torch.Tensor,
        state: FastMLPState,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, FastMLPState, dict[str, torch.Tensor]]:
        """Scan one episode-aligned sequence in temporal order."""
        if tokens.ndim != 4 or tokens.shape[-1] != self.dim:
            raise ValueError(
                f"tokens must have shape [B,T,N,{self.dim}], got {tuple(tokens.shape)}"
            )
        batch_size, trajectory_length = tokens.shape[:2]
        expected = (batch_size, trajectory_length)
        for name, value in (
            ("positions", positions),
            ("valid_mask", valid_mask),
            ("update_mask", update_mask),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")

        outputs = []
        inner_loss_sum = tokens.new_zeros(())
        num_updates = torch.zeros((), device=tokens.device, dtype=torch.long)
        current = state
        for timestep in range(trajectory_length):
            valid = valid_mask[:, timestep].to(device=tokens.device, dtype=torch.bool)
            step_output, current, metrics = self.step(
                tokens[:, timestep],
                current,
                positions=positions[:, timestep],
                update_mask=valid & update_mask[:, timestep].to(tokens.device, torch.bool),
            )
            outputs.append(torch.where(valid[:, None, None], step_output, tokens[:, timestep]))
            inner_loss_sum = inner_loss_sum + metrics["inner_loss"] * metrics["num_updates"]
            num_updates = num_updates + metrics["num_updates"]

        mean_inner_loss = inner_loss_sum / num_updates.clamp_min(1)
        return (
            torch.stack(outputs, dim=1),
            current,
            {
                "inner_loss": mean_inner_loss,
                "num_updates": num_updates,
            },
        )
