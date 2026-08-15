# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fast-weight state used by RoboTTT sequence adaptation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F


def _gelu_exact_derivative(value: torch.Tensor) -> torch.Tensor:
    """Derivative of ``torch.nn.functional.gelu`` with ``approximate='none'``."""
    normal_cdf = 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))
    normal_pdf = torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)
    return normal_cdf + value * normal_pdf


def _analytic_fast_mlp_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    update_mask: torch.Tensor,
    step_size: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    """Apply one exact analytic gradient step to the per-example fast MLP."""
    pre_activation = torch.einsum("bnd,bdh->bnh", key, w1) + b1[:, None]
    hidden = F.gelu(pre_activation)
    prediction = torch.einsum("bnh,bhd->bnd", hidden, w2) + b2[:, None]
    residual = prediction - value
    per_example_loss = residual.square().mean(dim=(1, 2))

    scale = 2.0 / (prediction.shape[1] * prediction.shape[2])
    grad_prediction = (
        residual * scale * update_mask.to(residual.dtype).reshape(-1, 1, 1)
    )
    grad_w2 = torch.einsum("bnh,bnd->bhd", hidden, grad_prediction)
    grad_b2 = grad_prediction.sum(dim=1)
    grad_hidden = torch.einsum("bnd,bhd->bnh", grad_prediction, w2)
    grad_pre_activation = grad_hidden * _gelu_exact_derivative(pre_activation)
    grad_w1 = torch.einsum("bnd,bnh->bdh", key, grad_pre_activation)
    grad_b1 = grad_pre_activation.sum(dim=1)
    gradients = (grad_w1, grad_b1, grad_w2, grad_b2)
    if not create_graph:
        gradients = tuple(gradient.detach() for gradient in gradients)

    previous = (w1, b1, w2, b2)
    candidates = tuple(
        parameter - step_size * gradient
        for parameter, gradient in zip(previous, gradients)
    )
    updated = []
    for candidate, old_value in zip(candidates, previous):
        broadcast_mask = update_mask.reshape(
            update_mask.shape[0], *([1] * (candidate.ndim - 1))
        )
        updated.append(torch.where(broadcast_mask, candidate, old_value))

    updated_w1, updated_b1, updated_w2, updated_b2 = updated
    adapted_hidden = F.gelu(
        torch.einsum("bnd,bdh->bnh", query, updated_w1) + updated_b1[:, None]
    )
    adapted = (
        torch.einsum("bnh,bhd->bnd", adapted_hidden, updated_w2)
        + updated_b2[:, None]
    )
    return adapted, tuple(updated), per_example_loss


_compiled_analytic_fast_mlp_step = torch.compile(
    _analytic_fast_mlp_step,
    fullgraph=True,
    dynamic=False,
)


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
        analytic_inner_update: bool = False,
        compile_inner_update: bool = False,
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
        if compile_inner_update and not analytic_inner_update:
            raise ValueError("compile_inner_update requires analytic_inner_update=True")

        self.dim = dim
        self.inner_dim = inner_dim
        self.inner_lr = inner_lr
        self.rope_theta = rope_theta
        self.gate_init = gate_init
        self.analytic_inner_update = analytic_inner_update
        self.compile_inner_update = compile_inner_update
        self._compile_failed = False
        self.w0_w1 = nn.Parameter(torch.empty(dim, inner_dim))
        self.w0_b1 = nn.Parameter(torch.zeros(inner_dim))
        self.w0_w2 = nn.Parameter(torch.empty(inner_dim, dim))
        self.w0_b2 = nn.Parameter(torch.zeros(dim))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.inner_lr_log_multiplier = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.gate = nn.Parameter(torch.full((dim,), math.atanh(gate_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the learned initial state after meta-device checkpoint loading."""
        nn.init.xavier_uniform_(self.w0_w1)
        nn.init.zeros_(self.w0_b1)
        nn.init.xavier_uniform_(self.w0_w2)
        nn.init.zeros_(self.w0_b2)
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        nn.init.constant_(self.inner_lr_log_multiplier, math.log(math.expm1(1.0)))
        nn.init.constant_(self.gate, math.atanh(self.gate_init))

    @property
    def inner_update_backend(self) -> str:
        if not self.analytic_inner_update:
            return "autograd"
        if self.compile_inner_update and not self._compile_failed:
            return "compiled"
        if self.compile_inner_update:
            return "analytic-fallback"
        return "analytic"

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

        normalized_tokens = F.layer_norm(tokens, (self.dim,))
        query = apply_temporal_rope(self.q_proj(normalized_tokens), positions, self.rope_theta)
        key = apply_temporal_rope(self.k_proj(normalized_tokens), positions, self.rope_theta)
        value = self.v_proj(normalized_tokens)

        with torch.enable_grad():
            if self.analytic_inner_update:
                step_size = self.inner_lr * F.softplus(self.inner_lr_log_multiplier)
                arguments = (
                    query,
                    key,
                    value,
                    *state.tensors(),
                    update_mask,
                    step_size,
                )
                if self.compile_inner_update and not self._compile_failed:
                    try:
                        adapted, updated_tensors, per_example_loss = (
                            _compiled_analytic_fast_mlp_step(
                                *arguments,
                                create_graph=self.training,
                            )
                        )
                    except Exception as error:
                        self._compile_failed = True
                        warnings.warn(
                            "RoboTTT compiled inner update failed; using eager analytic "
                            f"fallback: {error}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        adapted, updated_tensors, per_example_loss = _analytic_fast_mlp_step(
                            *arguments,
                            create_graph=self.training,
                        )
                else:
                    adapted, updated_tensors, per_example_loss = _analytic_fast_mlp_step(
                        *arguments,
                        create_graph=self.training,
                    )
                updated = FastMLPState(*updated_tensors)
            else:
                prediction = self._fast_forward(key, state)
                per_example_loss = (prediction - value).square().mean(dim=(1, 2))
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

            finite_inner_loss = torch.isfinite(per_example_loss).all()
            if self.inner_update_backend == "compiled" and hasattr(torch, "_assert_async"):
                torch._assert_async(finite_inner_loss, "RoboTTT inner loss contains NaN or Inf")
            elif not finite_inner_loss:
                raise FloatingPointError("RoboTTT inner loss contains NaN or Inf")

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
