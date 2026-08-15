# RoboTTT Trajectory Feature Caching and Analytic Inner Update Design

**Date:** 2026-08-15  
**Status:** Approved in chat; awaiting written-spec review  
**Scope:** GR00T N1.7 RoboTTT Stage 1 training on RoboCasa365, with a reusable analytic inner-update path for both training stages

## Goal

Reduce the dominant per-step overhead in the current RoboTTT implementation without changing the Stage 1 optimization objective, timestep ordering, fast-state semantics, register tokens, or checkpointed model parameters.

The change has two deliverables:

1. Compute every frozen vision-language backbone feature in a trajectory exactly once, in bounded GPU micro-batches, and then consume those features in the existing sequential TTT scan.
2. Replace per-layer `torch.autograd.grad` calls for the two-layer fast MLP with explicit, differentiable gradients and fuse the tensor kernel with `torch.compile`.

## Current Behavior and Bottlenecks

`RoboTTTTrainer.training_step` slices a trajectory according to `robottt_tbptt_steps`, which is currently one. Every slice calls the complete `Gr00tN1d7.forward` entry point. As a result, a context-128 optimizer step invokes the frozen Cosmos/Qwen vision-language backbone 128 times, even though Stage 1 trains only register tokens and RoboTTT parameters.

Each of the 32 RoboTTT layers also invokes `torch.autograd.grad` once per robot timestep. At context 128 this creates 4,096 small inner-gradient graphs and Python dispatch points per optimizer step. The `update_mask.any()` and finite-loss Python conditions can additionally synchronize the host with the GPU.

## Non-Goals

- This change does not alter the RoboTTT paper architecture, register-token count, layer placement, RoPE, loss, curriculum, optimizer, or TBPTT setting.
- It does not cache features across trajectories, optimizer steps, datasets, or checkpoints.
- It does not detach a trainable backbone. The frozen-feature fast path is Stage 1 only.
- It does not introduce a Triton or custom CUDA extension.
- It does not restart formal training until numerical equivalence and a real-GPU benchmark pass.

## Architecture

### 1. Bounded frozen-backbone feature producer

The trainer remains responsible for trajectory ordering and backward boundaries. It will slice the CPU-side collated trajectory into feature chunks before calling Hugging Face's device preparation. This avoids moving the full 8,192-timestep image trajectory to the 4090 at once.

The configured `robottt_backbone_micro_batch_size` is the maximum number of flattened robot timesteps sent through the backbone per call. For batch size `B`, the temporal chunk length is `max(1, micro_batch_size // B)`. The Stage 1 launcher will use 8 initially; the model-config default remains disabled for compatibility with existing non-RoboTTT callers.

For each chunk, the trainer will:

1. Slice raw temporal tensors and flattened VLM tensors with the existing multiplicity-aware alignment rule.
2. Move only that chunk to the model device.
3. Call `Gr00tN1d7.prepare_input` once for the chunk.
4. Verify that every backbone parameter is frozen.
5. Run `model.backbone` once under the normal autocast context and `torch.no_grad()`.
6. Reshape backbone outputs from `[B*C, ...]` to `[B, C, ...]`.
7. Consume each cached timestep in order through the action head, backpropagate its trajectory-weighted loss, detach the fast state at the configured TBPTT boundary, and then release the chunk.

The action-head call receives already-computed backbone outputs and prepared action inputs through a dedicated model method. It must not call `prepare_input` or the backbone again. Random action noise, flow time, and state dropout remain sampled inside the per-timestep action-head call, preserving the current random-number ordering.

The public full-model `forward` remains available for inference, tests, Stage 2, and compatibility. When the feature-cache option is disabled, the current trainer path remains the reference path.

### 2. Frozen-backbone safety boundary

Before entering the cached path, the trainer checks all backbone parameters. If any backbone parameter has `requires_grad=True`, it raises a targeted error rather than silently truncating gradients. This makes the optimization valid for Stage 1 and explicitly invalid for Stage 2 full-parameter tuning.

Stage 2 continues to use the reference full-model path unless a future design provides an activation-preserving backbone strategy. The analytic inner-update optimization remains valid in both stages because it preserves the outer computation graph.

### 3. Analytic fast-MLP inner gradient

For key tokens `K`, target values `V`, and per-example fast state `(W1, b1, W2, b2)`:

```text
Z = K W1 + b1
H = GELU(Z)
P = H W2 + b2
L_b = mean((P - V)^2)
```

With update mask `m_b`, token count `N`, and output width `D`:

```text
G_P  = m_b * 2(P - V) / (N D)
G_W2 = H^T G_P
G_b2 = sum_tokens(G_P)
G_H  = G_P W2^T
G_Z  = G_H * GELU'(Z)
G_W1 = K^T G_Z
G_b1 = sum_tokens(G_Z)
```

`GELU'` will match PyTorch's default exact GELU:

```text
GELU'(x) = 0.5 * (1 + erf(x / sqrt(2)))
           + x * exp(-x^2 / 2) / sqrt(2*pi)
```

The implementation uses ordinary PyTorch tensor operations without `no_grad`, so Stage 1 outer gradients still flow through the inner update to `W0`, Q/K/V projections, the learned inner learning-rate multiplier, and the gate. In evaluation mode, the analytic gradients are detached before the parameter subtraction to reproduce the current `create_graph=False` behavior.

Masked examples receive zero analytic gradients and are still blended with the previous state, guaranteeing that an inactive state is bitwise preserved. The all-masked case follows the same tensor path and removes the Python `update_mask.any()` branch.

### 4. Compiled fusion boundary

A pure tensor function will compute both fast-MLP passes, the masked inner loss, all four analytic gradients, the fast-state update, and the adapted query output. It accepts and returns tensors rather than `FastMLPState`, which keeps the compile boundary independent of Python dataclasses.

The model configuration gains three backward-compatible controls:

- `robottt_backbone_micro_batch_size: int | None = None`
- `robottt_analytic_inner_update: bool = False`
- `robottt_compile_inner_update: bool = False`

The RoboTTT Stage 1 launcher sets the backbone micro-batch size to 8, and both stages enable the analytic and compiled inner update. Compilation requires the analytic flag and uses one lazily compiled module-level callable shared by all identical layers. The current autograd implementation remains as the disabled-by-default reference path and as an explicit comparison oracle in tests.

On the configured PyTorch 2.9/CUDA 12.8 environment, the compiled path uses a device-side finite assertion so it does not synchronize Python once per layer and timestep. If compilation is unavailable or compilation fails, training emits one warning and uses the eager analytic implementation; the benchmark must report whether the compiled path was actually active so a silent fallback cannot be mistaken for a successful optimization.

## Data and Gradient Flow

```text
CPU trajectory
  -> raw temporal chunk
  -> device preparation
  -> frozen VLM backbone under no_grad (once per chunk)
  -> cached [B, chunk, sequence, hidden] features
  -> timestep 0 action head + 32-layer analytic TTT scan -> backward -> detach
  -> timestep 1 action head + 32-layer analytic TTT scan -> backward -> detach
  -> ...
  -> release feature chunk
  -> next raw chunk
```

The loss from timestep `t` retains the existing weight `(segment_length / trajectory_length)` and gradient-accumulation scaling. Fast state is carried across both timestep and feature-chunk boundaries. Chunk boundaries therefore control memory only and have no optimization meaning.

## Error Handling

- Reject non-positive backbone micro-batch sizes.
- Reject `robottt_compile_inner_update=True` when analytic updates are disabled.
- Reject the frozen-feature path when any backbone parameter is trainable.
- Preserve the existing shape validation for tokens, temporal positions, masks, and trajectory alignment.
- Preserve NaN/Inf detection without a per-step host synchronization on the compiled CUDA path.
- Do not catch data-alignment, shape, or non-finite errors. They remain fatal and visible in the training log.

## Files and Responsibilities

- `gr00t/model/modules/robottt.py`: eager reference update, analytic tensor kernel, optional compiled wrapper, and fast-state integration.
- `gr00t/model/modules/dit.py`: propagate the two inner-update configuration flags to all RoboTTT layers.
- `gr00t/configs/model/gr00t_n1d7.py`: backward-compatible model configuration fields.
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`: action-head-only sequence entry point using prepared inputs and cached backbone outputs.
- `gr00t/experiment/robottt_trainer.py`: CPU-side feature chunking, frozen-backbone validation, sequential loss/backward loop, and state carry.
- `gr00t/experiment/launch_robottt.py`: enable the Stage 1 micro-batch and analytic compiled update.
- Existing RoboTTT model and trainer test modules: numerical, gradient, alignment, and call-count regressions.

## Verification

### Unit and integration tests

1. Compare eager analytic updated weights, adapted output, inner loss, and update count against an autograd reference in float64 and float32.
2. Compare outer gradients for initial fast weights, Q/K/V projections, gate, and learned inner learning rate.
3. Cover mixed masks and the all-masked case, including bitwise preservation of inactive states.
4. Compare compiled and eager analytic paths on CUDA in float32 and bfloat16 with dtype-appropriate tolerances.
5. Use a small real model boundary with a counting backbone to prove a trajectory of length `T` invokes the backbone `ceil(B*T/micro_batch_size)` times while invoking the action scan in temporal order `T` times.
6. With a fixed random seed, compare reference and cached paths for reported loss, final fast state, and trainable-parameter gradients.
7. Verify the cached path rejects a trainable backbone and that legacy/default configurations retain the reference path.

### Real-GPU benchmark

After tests pass, run the current context-128 Stage 1 workload on the 4090 with identical data, seed, dtype, and checkpoint:

- one compile/warm-up iteration excluded from timing;
- at least five measured optimizer steps;
- report mean and median seconds per step, peak allocated/reserved memory, backbone calls per step, and whether compiled execution was active;
- compare against the existing 21.5-22 second-per-step baseline.

Formal curriculum training resumes only if loss and gradient equivalence pass, no non-finite values appear, and the optimized path demonstrates a material measured speedup.
