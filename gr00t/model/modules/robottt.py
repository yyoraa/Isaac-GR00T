# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fast-weight state used by RoboTTT sequence adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


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


class RoboTTTLayer(nn.Module):
    """One RoboTTT layer with a learned initial fast model."""

    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if inner_dim <= 0:
            raise ValueError(f"inner_dim must be positive, got {inner_dim}")

        self.dim = dim
        self.inner_dim = inner_dim
        self.w0_w1 = nn.Parameter(torch.empty(dim, inner_dim))
        self.w0_b1 = nn.Parameter(torch.zeros(inner_dim))
        self.w0_w2 = nn.Parameter(torch.empty(inner_dim, dim))
        self.w0_b2 = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.w0_w1)
        nn.init.xavier_uniform_(self.w0_w2)

    def initial_state(self, batch_size: int) -> FastMLPState:
        """Expand learned W0 into independent fast weights for each example."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        def expand(value: torch.Tensor) -> torch.Tensor:
            return value.unsqueeze(0).expand(batch_size, *value.shape).clone()

        return FastMLPState(
            w1=expand(self.w0_w1),
            b1=expand(self.w0_b1),
            w2=expand(self.w0_w2),
            b2=expand(self.w0_b2),
        )
