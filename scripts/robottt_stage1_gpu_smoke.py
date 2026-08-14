#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Full-dimension RoboTTT action-head GPU train/checkpoint/resume smoke."""

from dataclasses import dataclass
from pathlib import Path
import time

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.configs.robottt_training import build_wsd_scheduler, set_robottt_stage_trainability
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import torch
from transformers.feature_extraction_utils import BatchFeature
import tyro


@dataclass
class SmokeConfig:
    context_length: int = 128
    tbptt_steps: int = 128
    checkpoint: str = "/tmp/robottt-stage1-gpu-smoke.pt"
    resume_step: bool = True


def _model() -> Gr00tN1d7ActionHead:
    config = Gr00tN1d7Config(
        robottt_enabled=True,
        robottt_num_register_tokens=16,
        robottt_inner_dim=3072,
        robottt_inner_lr=0.1,
        robottt_rope_theta=10_000.0,
        robottt_gate_init=0.001,
        robottt_tbptt_steps=128,
        state_dropout_prob=0.0,
    )
    model = Gr00tN1d7ActionHead(config).to(device="cuda", dtype=torch.bfloat16)
    set_robottt_stage_trainability(model, "stage1")
    return model.train()


def _batch(config: Gr00tN1d7Config, length: int) -> tuple[BatchFeature, BatchFeature]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    backbone = BatchFeature(
        data={
            "backbone_features": torch.randn(
                1, length, 1, config.backbone_embedding_dim, device=device, dtype=dtype
            ),
            "backbone_attention_mask": torch.ones(1, length, 1, device=device, dtype=torch.bool),
            "image_mask": torch.ones(1, length, 1, device=device, dtype=torch.bool),
        }
    )
    action = BatchFeature(
        data={
            "state": torch.randn(
                1,
                length,
                config.state_history_length,
                config.max_state_dim,
                device=device,
                dtype=dtype,
            ),
            "action": torch.randn(
                1, length, config.action_horizon, config.max_action_dim, device=device, dtype=dtype
            ),
            "action_mask": torch.ones(
                1, length, config.action_horizon, config.max_action_dim, device=device, dtype=dtype
            ),
            "embodiment_id": torch.zeros(1, length, device=device, dtype=torch.long),
            "valid_mask": torch.ones(1, length, device=device, dtype=torch.bool),
            "action_loss_mask": torch.ones(1, length, device=device, dtype=torch.bool),
            "temporal_positions": torch.arange(length, device=device).view(1, length),
        }
    )
    return backbone, action


def _train_step(model, optimizer, scheduler, length: int, tbptt_steps: int) -> float:
    optimizer.zero_grad(set_to_none=True)
    backbone, action = _batch(model.config, length)
    fast_state = None
    weighted_loss = 0.0
    for start in range(0, length, tbptt_steps):
        end = min(start + tbptt_steps, length)
        segment_backbone = BatchFeature(
            data={key: value[:, start:end] for key, value in backbone.items()}
        )
        segment_action = BatchFeature(
            data={key: value[:, start:end] for key, value in action.items()}
        )
        output = model.forward_sequence(
            segment_backbone,
            segment_action,
            robottt_state=fast_state,
            tbptt_steps=tbptt_steps,
        )
        loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite outer loss: {loss.item()}")
        weight = (end - start) / length
        (loss * weight).backward()
        weighted_loss += float(loss.detach()) * weight
        fast_state = output.robottt_state.detach()
    optimizer.step()
    scheduler.step()
    return weighted_loss


def main(args: SmokeConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = _model()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-5, weight_decay=1e-5)
    scheduler = build_wsd_scheduler(
        optimizer, total_steps=30_000, warmup_steps=1_500, decay_steps=1_000
    )
    loss = _train_step(model, optimizer, scheduler, args.context_length, args.tbptt_steps)

    checkpoint = Path(args.checkpoint)
    torch.save(
        {
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        checkpoint,
    )

    resumed_loss = None
    if args.resume_step:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        resumed = _model()
        resumed.load_state_dict(payload["model"], strict=True)
        resumed_optimizer = torch.optim.AdamW(
            [parameter for parameter in resumed.parameters() if parameter.requires_grad],
            lr=2e-5,
            weight_decay=1e-5,
        )
        resumed_optimizer.load_state_dict(payload["optimizer"])
        resumed_scheduler = build_wsd_scheduler(
            resumed_optimizer,
            total_steps=30_000,
            warmup_steps=1_500,
            decay_steps=1_000,
        )
        resumed_scheduler.load_state_dict(payload["scheduler"])
        torch.set_rng_state(payload["cpu_rng"])
        torch.cuda.set_rng_state(payload["cuda_rng"])
        resumed_loss = _train_step(
            resumed,
            resumed_optimizer,
            resumed_scheduler,
            args.context_length,
            args.tbptt_steps,
        )

    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    print(
        {
            "context_length": args.context_length,
            "tbptt_steps": args.tbptt_steps,
            "loss": loss,
            "resumed_loss": resumed_loss,
            "peak_gpu_bytes": peak,
            "elapsed_seconds": elapsed,
            "checkpoint": str(checkpoint),
        }
    )


if __name__ == "__main__":
    main(tyro.cli(SmokeConfig, description=__doc__))
