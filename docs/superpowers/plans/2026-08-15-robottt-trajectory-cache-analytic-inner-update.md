# RoboTTT Trajectory Cache and Analytic Inner Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage 1 RoboTTT training compute each frozen VLM feature once and replace per-layer autograd inner updates with a numerically equivalent compiled analytic kernel.

**Architecture:** The trainer slices the CPU trajectory into bounded feature chunks, runs the frozen backbone once per chunk, and feeds cached timestep features through the existing sequential action/TTT path while preserving per-timestep backward and TBPTT state detach. `RoboTTTLayer` keeps its current autograd implementation as a reference, adds an eager differentiable analytic update, and optionally routes the pure tensor update through one shared lazy `torch.compile` callable.

**Tech Stack:** Python 3.12, PyTorch 2.9.0/CUDA 12.8, Hugging Face Transformers Trainer, pytest, NVIDIA RTX 4090

**Spec:** `docs/superpowers/specs/2026-08-15-robottt-trajectory-cache-analytic-inner-update-design.md`

## Global Constraints

- Preserve 16 register tokens, RoboTTT insertion in every action-DiT layer selected by the loaded checkpoint/configuration, TTT-KVB, temporal RoPE, sequence action forcing, and `robottt_tbptt_steps=1`.
- The cached-backbone path is legal only when every backbone parameter has `requires_grad=False`; fail closed otherwise.
- Feature chunks bound GPU memory and have no loss, optimizer, RNG, or state-detach semantics.
- The analytic training path must preserve outer gradients through the inner update; evaluation must match `create_graph=False`.
- Existing checkpoints and non-RoboTTT callers retain the reference path unless new flags are explicitly enabled.
- Do not restart the 30,000-step run until unit, integration, CUDA parity, and real-step benchmark checks pass.

---

### Task 1: Differentiable Analytic Fast-MLP Update

**Files:**
- Modify: `tests/gr00t/model/test_robottt.py`
- Modify: `gr00t/model/modules/robottt.py`

**Interfaces:**
- Consumes: `FastMLPState`, `RoboTTTLayer._fast_forward`, and the current autograd update in `RoboTTTLayer.step`.
- Produces: `RoboTTTLayer(..., analytic_inner_update: bool = False)`, `_gelu_exact_derivative(x)`, and `_analytic_fast_mlp_step(query, key, value, state_tensors, update_mask, step_size)` returning `(adapted, updated_w1, updated_b1, updated_w2, updated_b2, per_example_loss)`.

- [ ] **Step 1: Add a failing numerical-equivalence test**

  Add a test that deep-copies one float64 layer, enables the analytic path only on the copy, uses a mixed `[True, False]` update mask, and compares output, all four updated state tensors, inner loss, and update count against the autograd reference.

  ```python
  def test_analytic_inner_update_matches_autograd_reference():
      torch.manual_seed(19)
      reference = RoboTTTLayer(dim=4, inner_dim=7, gate_init=0.2).double().train()
      analytic = RoboTTTLayer(
          dim=4, inner_dim=7, gate_init=0.2, analytic_inner_update=True
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
          analytic_result[2]["inner_loss"], reference_result[2]["inner_loss"], rtol=1e-10, atol=1e-10
      )
      assert analytic_result[2]["num_updates"].item() == 1
  ```

- [ ] **Step 2: Run the test and verify RED**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/gr00t/model/test_robottt.py::test_analytic_inner_update_matches_autograd_reference -q
  ```

  Expected: FAIL because `RoboTTTLayer` has no analytic backend.

- [ ] **Step 3: Add a failing outer-gradient equivalence test**

  Backpropagate `output.square().mean() + inner_loss` through both copies and compare gradients for `w0_w1`, `w0_b1`, `w0_w2`, `w0_b2`, Q/K/V projections, `gate`, and `inner_lr_log_multiplier` in float64. This test catches accidental detach, a transposed einsum, a wrong loss denominator, or the approximate GELU derivative.

- [ ] **Step 4: Run the gradient test and verify RED**

  Run the single new test and confirm it fails only because the analytic option is absent.

- [ ] **Step 5: Implement the eager analytic update**

  In `robottt.py`, implement the exact GELU derivative and the pure tensor equations from the spec. Multiply `2 * (prediction - value) / (num_tokens * dim)` by `update_mask[:, None, None]` before computing parameter gradients. Keep `_blend_state` after subtraction so inactive examples preserve their previous tensors exactly. Route `step` through this helper only when `analytic_inner_update` is true; leave the existing `torch.autograd.grad` branch unchanged as the oracle.

  In training, do not detach analytic gradients. When `self.training` is false, detach the four analytic gradient tensors before subtraction to reproduce `create_graph=False`.

- [ ] **Step 6: Run analytic model tests and verify GREEN**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/gr00t/model/test_robottt.py -q
  ```

  Expected: all tests pass with no warnings.

- [ ] **Step 7: Commit the analytic eager implementation**

  ```bash
  git add gr00t/model/modules/robottt.py tests/gr00t/model/test_robottt.py
  git commit -m "feat: add analytic RoboTTT inner update"
  ```

---

### Task 2: Lazy Compiled Inner-Update Kernel

**Files:**
- Modify: `tests/gr00t/model/test_robottt.py`
- Modify: `gr00t/model/modules/robottt.py`

**Interfaces:**
- Consumes: `_analytic_fast_mlp_step` from Task 1.
- Produces: `RoboTTTLayer(..., compile_inner_update: bool = False)`, a shared lazy compiled callable, and `RoboTTTLayer.inner_update_backend` reporting `autograd`, `analytic`, `compiled`, or `analytic-fallback`.

- [ ] **Step 1: Add failing option-validation and backend tests**

  Add a CPU test asserting that compile without analytic raises `ValueError`, and that analytic without compile reports `analytic` after one step.

  ```python
  def test_compile_inner_update_requires_analytic_backend():
      with pytest.raises(ValueError, match="requires analytic_inner_update"):
          RoboTTTLayer(dim=4, inner_dim=8, compile_inner_update=True)
  ```

- [ ] **Step 2: Run the option test and verify RED**

  Expected: FAIL because the compile constructor option does not exist.

- [ ] **Step 3: Add a CUDA compiled-versus-eager parity test**

  Mark the test skipped when CUDA is unavailable. On CUDA, create matching analytic-eager and analytic-compiled layers, run one warm-up call and one measured call in float32, then compare output, state, loss, and outer gradients with `rtol=2e-5, atol=2e-6`. Repeat forward/state parity under bfloat16 with `rtol=2e-2, atol=2e-2`. Assert that the compiled layer reports `compiled`, so a fallback cannot satisfy the test.

- [ ] **Step 4: Run the CUDA test and verify RED**

  Run on mll3:

  ```bash
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/gr00t/model/test_robottt.py -k compiled -q
  ```

  Expected: FAIL because no compiled backend exists.

- [ ] **Step 5: Implement lazy compilation and safe fallback**

  Wrap the pure tensor helper once at module scope with `torch.compile(fullgraph=True, dynamic=False)`. Invoke it lazily only when the flag is enabled. Keep Python dataclasses and mask metrics outside the compiled boundary. Use a CUDA device-side asynchronous finite assertion for the compiled output; keep the direct finite check on eager paths. Catch only compile/backend exceptions, emit one warning per process, switch that layer to eager analytic execution, and expose `analytic-fallback` through `inner_update_backend`.

- [ ] **Step 6: Run CPU and CUDA parity tests and verify GREEN**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/gr00t/model/test_robottt.py -q
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/gr00t/model/test_robottt.py -k compiled -q
  ```

- [ ] **Step 7: Commit the compiled backend**

  ```bash
  git add gr00t/model/modules/robottt.py tests/gr00t/model/test_robottt.py
  git commit -m "perf: compile RoboTTT analytic update"
  ```

---

### Task 3: Propagate Backward-Compatible Optimization Configuration

**Files:**
- Modify: `gr00t/configs/model/gr00t_n1d7.py`
- Modify: `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
- Modify: `gr00t/model/modules/dit.py`
- Modify: `gr00t/model/gr00t_n1d7/setup.py`
- Modify: `gr00t/experiment/launch_robottt.py`
- Modify: `tests/gr00t/model/test_robottt_dit.py`
- Modify: `tests/gr00t/experiment/test_robottt_checkpoint_loading.py`
- Modify: `tests/gr00t/experiment/test_robottt_training.py`

**Interfaces:**
- Consumes: the two `RoboTTTLayer` flags from Tasks 1-2.
- Produces: model config fields `robottt_backbone_micro_batch_size`, `robottt_analytic_inner_update`, and `robottt_compile_inner_update`, passed through checkpoint loading, action-head construction, DiT blocks, and the launcher.

- [ ] **Step 1: Add failing propagation tests**

  Extend the existing checkpoint-loading capture to assert all three keyword values. Extend the DiT construction test to assert every layer has analytic and compile enabled. Extend the launcher test to assert Stage 1 uses micro-batch 8 and both stages enable analytic compilation, while Stage 2 leaves frozen-backbone caching disabled.

- [ ] **Step 2: Run the focused tests and verify RED**

  ```bash
  .venv/bin/python -m pytest \
    tests/gr00t/model/test_robottt_dit.py \
    tests/gr00t/experiment/test_robottt_checkpoint_loading.py \
    tests/gr00t/experiment/test_robottt_training.py -q
  ```

  Expected: FAIL on missing configuration fields/keywords.

- [ ] **Step 3: Add config fields and thread them through constructors**

  Add backward-compatible defaults:

  ```python
  robottt_backbone_micro_batch_size: int | None = None
  robottt_analytic_inner_update: bool = False
  robottt_compile_inner_update: bool = False
  ```

  Pass the two inner-update flags from `Gr00tN1d7ActionHead` to `AlternateVLDiT`, then to every `BasicTransformerBlock`, then to `RoboTTTLayer`. Pass all three fields into `AutoModel.from_pretrained` in setup. Add launcher option `backbone_micro_batch_size: int = 8`, validate it as positive, enable it only for Stage 1, and enable both inner-update flags for both stages.

- [ ] **Step 4: Run propagation tests and verify GREEN**

  Run the same focused command and confirm all pass.

- [ ] **Step 5: Commit configuration propagation**

  ```bash
  git add gr00t/configs/model/gr00t_n1d7.py gr00t/model/gr00t_n1d7/gr00t_n1d7.py \
    gr00t/model/modules/dit.py gr00t/model/gr00t_n1d7/setup.py \
    gr00t/experiment/launch_robottt.py tests/gr00t/model/test_robottt_dit.py \
    tests/gr00t/experiment/test_robottt_checkpoint_loading.py \
    tests/gr00t/experiment/test_robottt_training.py
  git commit -m "feat: configure RoboTTT training fast paths"
  ```

---

### Task 4: Cached-Backbone Model Entry Point

**Files:**
- Modify: `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
- Create: `tests/gr00t/model/test_robottt_cached_forward.py`

**Interfaces:**
- Consumes: prepared `BatchFeature` objects from `Gr00tN1d7.prepare_input` and the current `Gr00tN1d7ActionHead.forward_sequence`.
- Produces: `Gr00tN1d7.forward(..., cached_backbone_output=None, cached_action_input=None)` with mutually exclusive raw-input and cached-input modes, plus `_reshape_sequence_backbone_output(output, batch_size, trajectory_length)`.

- [ ] **Step 1: Add a failing cached-forward behavior test**

  Build a small `Gr00tN1d7` object with a backbone whose `forward` raises if called and a recording action head returning a real `BatchFeature`. Call the public model `forward` with cached backbone/action features and assert the action result and carried fast state. The test fails if the cached branch accidentally calls the backbone.

- [ ] **Step 2: Run the cached-forward test and verify RED**

  ```bash
  .venv/bin/python -m pytest tests/gr00t/model/test_robottt_cached_forward.py -q
  ```

  Expected: FAIL because cached arguments are not accepted.

- [ ] **Step 3: Add failing validation and reshape tests**

  Test that raw `inputs` cannot be combined with cached inputs, that both cached objects are required together, and that a flat `[B*T, S, D]` backbone feature plus masks reshapes to `[B, T, S, D]` without changing values.

- [ ] **Step 4: Implement the cached public forward branch**

  Keep the current raw-input branch intact. In cached mode, validate arguments and call only `action_head.forward_sequence` with the supplied fast state and configured TBPTT steps. Extract the existing flat-to-sequence backbone reshape loop into the static helper and use it from both raw and cached training preparation.

- [ ] **Step 5: Run model tests and verify GREEN**

  ```bash
  .venv/bin/python -m pytest tests/gr00t/model/test_robottt_cached_forward.py \
    tests/gr00t/model/test_robottt_action_head.py -q
  ```

- [ ] **Step 6: Commit the cached model boundary**

  ```bash
  git add gr00t/model/gr00t_n1d7/gr00t_n1d7.py \
    tests/gr00t/model/test_robottt_cached_forward.py
  git commit -m "feat: add cached RoboTTT model forward"
  ```

---

### Task 5: Micro-Batched Frozen-Feature Trainer

**Files:**
- Modify: `gr00t/experiment/robottt_trainer.py`
- Modify: `tests/gr00t/experiment/test_robottt_training.py`

**Interfaces:**
- Consumes: raw collated sequence inputs, `_slice_trajectory_inputs`, model config `robottt_backbone_micro_batch_size`, `Gr00tN1d7.prepare_input`, the frozen backbone, and cached public forward from Task 4.
- Produces: `_training_step_with_cached_backbone(model, inputs, num_items_in_batch)` and helpers that carry fast state across feature chunks while retaining the reference `training_step` when caching is disabled.

- [ ] **Step 1: Add a failing backbone-call-count and ordering test**

  Use a real `RoboTTTTrainer` object with only the Trainer services needed by `training_step`, a counting frozen backbone, and a recording cached model branch. For `B=1`, `T=7`, and micro-batch 3, assert three backbone calls with temporal spans `[0,1,2]`, `[3,4,5]`, and `[6]`, seven ordered action calls, and state generations carried across chunk boundaries.

  The fake backbone is allowed only at the slow model boundary; assertions target trainer behavior and returned loss, not the fake itself.

- [ ] **Step 2: Run the call-count test and verify RED**

  ```bash
  .venv/bin/python -m pytest tests/gr00t/experiment/test_robottt_training.py -k cached_backbone -q
  ```

  Expected: FAIL because the trainer still enters the full model once per timestep.

- [ ] **Step 3: Add failing safety and equivalence tests**

  Add one test that sets a backbone parameter trainable and expects a targeted `RuntimeError`. Add one fixed-seed test comparing reference and cached paths for reported loss, final fast-state tensors, and trainable parameter gradients on a two-timestep tiny real RoboTTT action head. Add a test that caching disabled continues to use the existing path.

- [ ] **Step 4: Run the new tests and verify RED**

  Confirm failures are due to the missing cached trainer path and missing frozen-backbone guard.

- [ ] **Step 5: Implement CPU-first chunking and sequential consumption**

  In `training_step`, inspect trajectory metadata before `_prepare_inputs`. If caching is disabled, execute the existing implementation unchanged. If enabled:

  1. validate positive micro-batch size and a fully frozen unwrapped backbone;
  2. derive temporal chunk length as `max(1, micro_batch_size // batch_size)`;
  3. slice the raw CPU payload for each chunk, then call `_prepare_inputs` only on that chunk;
  4. call the unwrapped model's `prepare_input`, then its frozen backbone once under `torch.no_grad()` and `compute_loss_context_manager()`;
  5. reshape backbone outputs to sequence form;
  6. slice cached backbone/action features one timestep at a time and invoke the wrapped public model cached branch;
  7. apply the existing trajectory and gradient-accumulation loss scaling, backward behavior, and deepspeed keyword;
  8. carry and detach fast state exactly at each `robottt_tbptt_steps` boundary;
  9. release local chunk tensors before advancing.

  Do not move the complete raw trajectory to GPU and do not sample action noise during feature production.

- [ ] **Step 6: Run trainer and model integration tests and verify GREEN**

  ```bash
  .venv/bin/python -m pytest tests/gr00t/experiment/test_robottt_training.py \
    tests/gr00t/model/test_robottt_cached_forward.py \
    tests/gr00t/model/test_robottt_action_head.py -q
  ```

- [ ] **Step 7: Commit the cached trainer path**

  ```bash
  git add gr00t/experiment/robottt_trainer.py \
    tests/gr00t/experiment/test_robottt_training.py
  git commit -m "perf: cache frozen RoboTTT trajectory features"
  ```

---

### Task 6: Full Regression, Numerical Probe, and 4090 Benchmark

**Files:**
- Modify only if a regression test exposes a production defect: the file responsible for that defect and its existing test module.
- Runtime artifacts: `/home/yiqi/yiyun/robottt-runs/benchmark-trajectory-cache-20260815/`

**Interfaces:**
- Consumes: all optimized paths from Tasks 1-5 and the stopped Stage 1 checkpoint/data configuration.
- Produces: verified tests, benchmark log, measured speed/memory comparison, and a go/no-go decision for resuming formal training.

- [ ] **Step 1: Run formatting and focused CPU regressions**

  ```bash
  .venv/bin/ruff format --check gr00t/model/modules/robottt.py \
    gr00t/model/modules/dit.py gr00t/model/gr00t_n1d7/gr00t_n1d7.py \
    gr00t/experiment/robottt_trainer.py gr00t/experiment/launch_robottt.py tests/gr00t
  .venv/bin/ruff check gr00t/model/modules/robottt.py gr00t/model/modules/dit.py \
    gr00t/model/gr00t_n1d7/gr00t_n1d7.py gr00t/experiment/robottt_trainer.py \
    gr00t/experiment/launch_robottt.py tests/gr00t/model/test_robottt.py \
    tests/gr00t/model/test_robottt_cached_forward.py \
    tests/gr00t/experiment/test_robottt_training.py
  .venv/bin/python -m pytest tests/gr00t/model/test_robottt.py \
    tests/gr00t/model/test_robottt_dit.py \
    tests/gr00t/model/test_robottt_action_head.py \
    tests/gr00t/model/test_robottt_cached_forward.py \
    tests/gr00t/experiment/test_robottt_training.py \
    tests/gr00t/experiment/test_robottt_checkpoint_loading.py -q
  ```

- [ ] **Step 2: Run CUDA compiled parity and a context-128 numerical probe**

  Run the compiled tests on GPU, then run one reference and one optimized context-128 training step from the same checkpoint/data batch with identical seeds. Record loss, each trainable parameter's gradient norm, final fast-state checksums, and peak memory. Require finite values and dtype-appropriate closeness; investigate any mismatch before benchmarking.

- [ ] **Step 3: Benchmark optimized context-128 steps**

  Start a dedicated tmux session using the same base checkpoint, RoboCasa365 mixture, batch size, accumulation, BF16, and seed as the stopped run. Write logs under the runtime artifact directory. Exclude the first compile/warm-up optimizer step and measure at least five subsequent steps. Capture:

  ```text
  mean_seconds_per_step
  median_seconds_per_step
  max_memory_allocated
  max_memory_reserved
  backbone_calls_per_optimizer_step
  inner_update_backend
  ```

  Compare against the recorded 21.5-22.0 seconds/step baseline. The benchmark passes only if every backbone frame is encoded once, the backend reports `compiled`, no non-finite value occurs, and measured time shows a material improvement.

- [ ] **Step 4: Run `git diff --check` and inspect the complete diff**

  ```bash
  git diff --check
  git status --short
  git diff --stat HEAD~4..HEAD
  ```

  Confirm that pre-existing unrelated edits remain intact and no runtime data, token, cache, or benchmark artifact is staged.

- [ ] **Step 5: Commit only any verification-driven fixes**

  If verification required a fix, first add a failing regression test, apply the minimal production correction, rerun the affected and full focused suites, and commit only those files with:

  ```bash
  git commit -m "fix: preserve RoboTTT fast-path equivalence"
  ```

- [ ] **Step 6: Report benchmark outcome before resuming formal training**

  Report exact test counts, loss/gradient comparison, old and new seconds per step, speedup ratio, peak memory, backend status, and the checkpoint from which formal training can resume. Leave formal training stopped until that evidence is presented.
