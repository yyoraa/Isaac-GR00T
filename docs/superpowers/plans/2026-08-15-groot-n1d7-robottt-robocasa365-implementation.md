# GR00T N1.7 RoboTTT RoboCasa365 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the publicly specified RoboTTT algorithm to all 16 action-transformer layers of public GR00T N1.7 GA, then train and evaluate it on genuine long RoboCasa365 episodes.

**Architecture:** A functional fast-weight MLP is owned by every DiT block and scans robot timesteps after attention but before feed-forward. The N1.7 action head prepends 16 learned register tokens, performs independent per-timestep flow noising, carries per-episode fast state through TBPTT, and exposes an explicit inference lifecycle that updates once per observation. A separate sequence dataset preserves episode boundaries and supplies valid/loss masks to two exact training stages.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Diffusers, LeRobot/RoboCasa365, pytest, Ruff, DeepSpeed ZeRO-3.

**Spec:** `docs/superpowers/specs/2026-08-15-groot-n1d7-robottt-robocasa365-design.md`

## Global Constraints

- Use public `nvidia/GR00T-N1.7-3B`; document its Cosmos/Qwen3-VL backbone mismatch with the paper's Eagle description.
- Use exactly 16 learned register tokens of width 1536.
- Insert one RoboTTT module after attention and before feed-forward in every one of the 16 DiT blocks.
- Each fast MLP is 1536 -> 3072 -> 1536 with GeLU; do not reduce width or layer coverage for memory.
- Use TTT-KVB update-then-apply, learned `W0`, base inner learning rate 0.1, learned multiplier, vector tanh gate initialized to 0.001, and temporal RoPE theta 10000.
- Sample independent flow time/noise per robot timestep; all 40 tokens in one action chunk share that timestep's flow time.
- Reset fast state at episode boundaries and update exactly once per environment observation during four-step flow inference.
- Never concatenate separate episodes to manufacture context; 8K is allowed only for genuine episodes of that length.
- Stage 1 is 30K steps with only new sequence parameters trainable, AdamW weight decay 1e-5, WSD peak LR 2e-5.
- Stage 2 is 20K steps at 1K context with all parameters trainable, AdamW weight decay 1e-5, cosine peak LR 5e-5.
- Stage 2 remains full-parameter training using ZeRO-3 CPU offload; do not silently substitute LoRA.
- Follow strict red-green-refactor TDD and commit after each independently passing task.

---

### Task 1: RoboTTT configuration and fast-state types

**Files:**
- Modify: `gr00t/configs/model/gr00t_n1d7.py`
- Create: `gr00t/model/modules/robottt.py`
- Create: `tests/gr00t/model/test_robottt.py`

**Interfaces:**
- Produces: `RoboTTTConfig`, `FastMLPState`, `RoboTTTState`, `RoboTTTLayer.initial_state()`.
- `RoboTTTConfig` fields: `enabled`, `num_register_tokens`, `hidden_dim`, `inner_dim`, `inner_lr`, `rope_theta`, `gate_init`.
- `FastMLPState` stores batched `w1`, `b1`, `w2`, `b2`; `detach()` returns the same values detached from the old graph.

- [ ] **Step 1: Write failing configuration and state tests**

```python
def test_robottt_defaults_match_paper():
    config = Gr00tN1d7Config(robottt_enabled=True)
    assert config.robottt_num_register_tokens == 16
    assert config.robottt_inner_dim == 3072
    assert config.robottt_inner_lr == 0.1
    assert config.robottt_rope_theta == 10000.0

def test_initial_fast_state_is_batched_and_meta_learnable():
    layer = RoboTTTLayer(dim=8, inner_dim=16)
    state = layer.initial_state(batch_size=2)
    assert state.fast.w1.shape == (2, 8, 16)
    state.fast.w1.sum().backward()
    assert layer.w0_w1.grad is not None
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/gr00t/model/test_robottt.py -v`
Expected: collection fails because `gr00t.model.modules.robottt` does not exist.

- [ ] **Step 3: Add exact config fields and typed state containers**

Implement frozen dataclasses with `tree_map`, `detach`, `to`, and batch-shape validation. Add the seven `robottt_*` fields to `Gr00tN1d7Config`; disabled is the default so base checkpoints retain behavior.

- [ ] **Step 4: Add learned W0 and initial-state expansion**

Initialize `w0_w1 [D,H]`, `w0_b1 [H]`, `w0_w2 [H,D]`, and `w0_b2 [D]` with Xavier weights and zero biases. Expand and clone them to a leading batch axis without severing the gradient path to W0.

- [ ] **Step 5: Verify GREEN and lint**

Run: `python -m pytest tests/gr00t/model/test_robottt.py -v`
Run: `ruff check gr00t/model/modules/robottt.py gr00t/configs/model/gr00t_n1d7.py tests/gr00t/model/test_robottt.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add gr00t/configs/model/gr00t_n1d7.py gr00t/model/modules/robottt.py tests/gr00t/model/test_robottt.py
git commit -m "feat: add RoboTTT fast state"
```

### Task 2: TTT-KVB inner update, temporal RoPE, and residual gate

**Files:**
- Modify: `gr00t/model/modules/robottt.py`
- Modify: `tests/gr00t/model/test_robottt.py`

**Interfaces:**
- Produces: `apply_temporal_rope(x, positions, theta)`, `RoboTTTLayer.step(tokens, state, positions, update_mask) -> tuple[tokens, state, metrics]`, and `RoboTTTLayer.scan(tokens, state, positions, valid_mask, update_mask)`.
- `tokens` for `step` is `[B,N,D]`; for `scan` it is `[B,T,N,D]`.

- [ ] **Step 1: Write failing hand-derived RoPE and update-order tests**

Use a four-dimensional literal tensor at positions 0 and 1 and compare against hand-computed sine/cosine rotation. Configure a one-dimensional deterministic fast MLP and assert that `step()` output uses the post-gradient state, not W0.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/gr00t/model/test_robottt.py -k 'rope or update_then_apply' -v`
Expected: FAIL because RoPE and `step` are absent.

- [ ] **Step 3: Implement fast MLP and TTT-KVB**

Compute `q=Wq(x)`, `k=Wk(x)`, `v=Wv(x)`, rotate q/k by temporal position, and evaluate `mean((fast_mlp(k)-v)^2)`. Obtain gradients for all four fast tensors with `torch.autograd.grad(create_graph=self.training)`. Apply `fast - 0.1 * softplus(lr_multiplier) * grad`, query the updated model, and add `tanh(gate) * output` to the incoming tokens.

- [ ] **Step 4: Implement masked temporal scan**

Iterate time in order. A false valid mask returns the input unchanged and preserves state. A false update mask applies the current state without an inner update. Return mean finite inner-loss metrics over valid updates.

- [ ] **Step 5: Add meta-gradient, masking, detach, and NaN tests**

Assert outer loss reaches W0 and Q/K/V; padding cannot change state; detached state has no `grad_fn`; non-finite inner loss raises `FloatingPointError` with layer/timestep context.

- [ ] **Step 6: Verify GREEN**

Run: `python -m pytest tests/gr00t/model/test_robottt.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add gr00t/model/modules/robottt.py tests/gr00t/model/test_robottt.py
git commit -m "feat: implement TTT-KVB temporal scan"
```

### Task 3: Insert RoboTTT into all 16 DiT blocks

**Files:**
- Modify: `gr00t/model/modules/dit.py`
- Create: `tests/gr00t/model/test_robottt_dit.py`

**Interfaces:**
- `BasicTransformerBlock.forward_attention(...)` returns the attention residual.
- `BasicTransformerBlock.forward_feed_forward(hidden_states)` returns the block output.
- `AlternateVLDiT.initial_robottt_state(batch_size)` returns 16 layer states.
- `AlternateVLDiT.forward(..., robottt_state=None, temporal_shape=None, temporal_positions=None, valid_mask=None, update_mask=None)` returns the existing model output plus the next state when enabled.

- [ ] **Step 1: Write a failing placement test**

Replace attention, RoboTTT, and FF components with deterministic arithmetic modules and assert a block computes `FF(TTT(AttentionResidual(x)))`, distinguishing it from both alternative orders.

- [ ] **Step 2: Write a failing all-layer/state-carry test**

Construct a four-layer miniature `AlternateVLDiT`, enable RoboTTT, run `[B=1,T=3,N=2,D=8]`, and assert each block's update counter is three. In the production config assert there are exactly 16 RoboTTT layers.

- [ ] **Step 3: Verify RED**

Run: `python -m pytest tests/gr00t/model/test_robottt_dit.py -v`
Expected: FAIL because block phases and temporal arguments are absent.

- [ ] **Step 4: Split block phases without changing the disabled path**

Move existing attention code verbatim into `forward_attention` and FF code into `forward_feed_forward`; make legacy `forward` compose them. Keep all mask and AdaNorm behavior unchanged.

- [ ] **Step 5: Wire temporal reshape and per-layer state**

For each block, flatten `[B,T,N,D]` to `[B*T,N,D]` for existing attention, reshape back, scan only `[R,q,A]` with that block's RoboTTT layer, flatten for FF, then restore `[B,T,N,D]`. Do not pass VLM tokens to the scan.

- [ ] **Step 6: Verify base parity and GREEN**

Seed both old-style and disabled-RoboTTT paths with identical weights and inputs; assert exact output equality. Run the new tests plus `tests/gr00t/model/test_action_head.py`.

- [ ] **Step 7: Commit**

```bash
git add gr00t/model/modules/dit.py tests/gr00t/model/test_robottt_dit.py
git commit -m "feat: insert RoboTTT in every DiT block"
```

### Task 4: Register tokens and sequence action forcing in the N1.7 action head

**Files:**
- Modify: `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
- Create: `tests/gr00t/model/test_robottt_action_head.py`

**Interfaces:**
- `Gr00tN1d7ActionHead.register_tokens` has shape `[16,1536]` when enabled.
- `forward_sequence(backbone_output, action_input, robottt_state=None, tbptt_steps=None)` accepts state `[B,T,H,D]`, action `[B,T,40,A]`, masks `[B,T,...]`, and returns loss plus final fast state.
- `sample_sequence_time(batch_size, trajectory_length, device, dtype)` returns `[B,T,1,1]`.

- [ ] **Step 1: Write failing token-layout and independent-time tests**

Assert the DiT receives 57 tokens in the order 16 registers, one state, 40 actions; assert the decoder receives only the last 40. With a fixed RNG seed, assert at least two temporal flow samples differ and the shape is `[B,T,1,1]`.

- [ ] **Step 2: Write failing loss-mask and episode-boundary tests**

Set one timestep's `action_loss_mask` to zero and change its target by a large amount; total loss must be unchanged. Provide an episode-reset mask and assert the next fast state equals a fresh scan rather than the preceding episode state.

- [ ] **Step 3: Verify RED**

Run: `python -m pytest tests/gr00t/model/test_robottt_action_head.py -v`
Expected: FAIL because registers and `forward_sequence` are absent.

- [ ] **Step 4: Implement register construction and explicit action slicing**

Create one trainable register bank, expand to `[B,T,16,D]`, concatenate before state/action features, and assert the action slice begins at `16 + 1`. Keep the old single-step forward unchanged when disabled.

- [ ] **Step 5: Implement sequence action forcing and TBPTT**

Sample noise and beta time at `[B,T,...]`, flatten only VLM/within-step computations, and scan segments in time order. Carry the 16 fast states between segments and call `.detach()` only at segment boundaries. Apply valid and action-loss masks to the numerator and denominator.

- [ ] **Step 6: Verify GREEN and backward**

Run the new tests and a miniature backward pass; assert register, W0, Q/K/V, gate, and learned LR gradients are finite.

- [ ] **Step 7: Commit**

```bash
git add gr00t/model/gr00t_n1d7/gr00t_n1d7.py tests/gr00t/model/test_robottt_action_head.py
git commit -m "feat: add RoboTTT sequence action head"
```

### Task 5: Online inference lifecycle and one update per observation

**Files:**
- Modify: `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
- Modify: `gr00t/policy/gr00t_policy.py`
- Create: `tests/gr00t/model/test_robottt_inference.py`
- Modify: `tests/gr00t/policy/test_gr00t_policy.py`

**Interfaces:**
- `reset_robottt_state(batch_size=1)` clears all 16 layer states and observation counter.
- `begin_robottt_observation(episode_ids)` initializes/reset state and returns a transaction.
- Four denoising calls use `update_mask=[True,False,False,False]`; state commits once after action generation.

- [ ] **Step 1: Write a failing four-denoise counter test**

Use a miniature head with `num_inference_timesteps=4`; assert each layer records one update and four applications for one observation, two updates after two observations, and zero carried updates after reset.

- [ ] **Step 2: Write a failing policy reset test**

Exercise policy reset and episode-ID change through the real policy boundary. Assert timeout, explicit reset, and changed episode ID each clear fast state.

- [ ] **Step 3: Verify RED**

Run: `python -m pytest tests/gr00t/model/test_robottt_inference.py tests/gr00t/policy/test_gr00t_policy.py -v`
Expected: FAIL because lifecycle methods are absent.

- [ ] **Step 4: Implement transactional inference state**

Enable gradients only around the inner TTT update even though action inference is otherwise no-grad. Hold the first-call candidate state locally, reuse it for remaining flow evaluations without more updates, and commit only when an action chunk completes successfully. Abort discards the candidate.

- [ ] **Step 5: Verify GREEN**

Run the two targeted test files and existing action-head tests.

- [ ] **Step 6: Commit**

```bash
git add gr00t/model/gr00t_n1d7/gr00t_n1d7.py gr00t/policy/gr00t_policy.py tests/gr00t/model/test_robottt_inference.py tests/gr00t/policy/test_gr00t_policy.py
git commit -m "feat: persist RoboTTT state across observations"
```

### Task 6: Genuine episode sequence dataset and collator

**Files:**
- Create: `gr00t/data/dataset/trajectory_sequence_dataset.py`
- Modify: `gr00t/data/dataset/factory.py`
- Create: `gr00t/data/collator/trajectory_collator.py`
- Modify: `gr00t/configs/data/data_config.py`
- Create: `tests/gr00t/data/test_trajectory_sequence_dataset.py`

**Interfaces:**
- `TrajectorySequenceDataset` wraps `LeRobotEpisodeLoader` and yields one ordered window from one episode.
- `TrajectoryWindow` contains `episode_id`, `start`, `length`, `valid_mask`, `episode_reset_mask`, `action_loss_mask`, and per-timestep modalities.
- `TrajectoryCollator` pads only time and stacks as `[B,T,...]`.

- [ ] **Step 1: Write failing window-boundary tests**

Build two synthetic episodes with unique numeric sentinels. Assert every window contains one sentinel only, start indices are monotonic, and no window crosses the episode boundary.

- [ ] **Step 2: Write failing padding/curriculum tests**

Collate lengths three and five, assert `[B,5]` valid masks and zero action loss on padding. Set buckets `[2,4,8]`; assert the chosen length never exceeds either curriculum limit or episode length.

- [ ] **Step 3: Verify RED**

Run: `python -m pytest tests/gr00t/data/test_trajectory_sequence_dataset.py -v`
Expected: FAIL because sequence dataset/collator modules are absent.

- [ ] **Step 4: Implement ordered windows and deterministic sampling**

Reuse `LeRobotEpisodeLoader` for parquet/video/language decoding. Sample `(episode_id,start)` from a seeded index; materialize consecutive timesteps with existing modality processors; produce explicit masks and never wrap or concatenate.

- [ ] **Step 5: Implement collator and factory selection**

Add `sequence_mode`, `context_length`, `context_buckets`, and `tbptt_steps` fields to data config. Select the new dataset/collator only when sequence mode is enabled; preserve the existing sharded single-step default.

- [ ] **Step 6: Verify GREEN and existing data tests**

Run the new tests plus `tests/gr00t/data/test_dataset_factory.py` and `tests/gr00t/data/test_sharded_datasets.py`.

- [ ] **Step 7: Commit**

```bash
git add gr00t/data gr00t/configs/data/data_config.py tests/gr00t/data/test_trajectory_sequence_dataset.py
git commit -m "feat: load episode-safe trajectory windows"
```

### Task 7: Two-stage training, freezing, WSD, and resumable curriculum

**Files:**
- Create: `gr00t/configs/robottt_training.py`
- Create: `gr00t/experiment/robottt_trainer.py`
- Create: `gr00t/experiment/launch_robottt.py`
- Create: `gr00t/configs/deepspeed/robottt_zero3_offload.json`
- Create: `tests/gr00t/experiment/test_robottt_training.py`

**Interfaces:**
- `RoboTTTTrainingConfig.stage` is `stage1` or `stage2` and validates all paper hyperparameters.
- `set_robottt_stage_trainability(model, stage)` freezes exactly the intended parameter set.
- `build_wsd_scheduler(optimizer, total_steps, warmup_steps, decay_steps)` implements linear warmup, stable plateau, and linear decay.
- Checkpoint state includes curriculum bucket, sampler state, manifest hash, and RNG state.

- [ ] **Step 1: Write failing trainability and schedule tests**

For Stage 1 assert only names containing registers or RoboTTT are trainable. For Stage 2 assert all parameters are trainable. Check literal WSD factors at warmup end, stable midpoint, and final step.

- [ ] **Step 2: Write failing deterministic-resume test**

Train a tiny synthetic model for two steps, save, resume, and assert the next episode/start/context bucket and scheduler LR match an uninterrupted three-step run.

- [ ] **Step 3: Verify RED**

Run: `python -m pytest tests/gr00t/experiment/test_robottt_training.py -v`
Expected: FAIL because training modules are absent.

- [ ] **Step 4: Implement exact stage presets**

Stage 1: 30K, WSD 2e-5, weight decay 1e-5. Stage 2: 20K, cosine 5e-5, weight decay 1e-5, context 1024. Validate conflicting CLI overrides rather than silently changing these defaults.

- [ ] **Step 5: Implement ZeRO-3 CPU offload config**

Set parameter and optimizer offload to CPU with pinned memory, BF16 enabled, activation checkpointing, microbatch one, automatic gradient accumulation, and non-lossy checkpoint saving. Emit an error if Stage 2 requests LoRA or partial trainability.

- [ ] **Step 6: Verify GREEN**

Run the new training tests and existing resume-compatibility tests.

- [ ] **Step 7: Commit**

```bash
git add gr00t/configs/robottt_training.py gr00t/configs/deepspeed/robottt_zero3_offload.json gr00t/experiment/robottt_trainer.py gr00t/experiment/launch_robottt.py tests/gr00t/experiment/test_robottt_training.py
git commit -m "feat: add exact RoboTTT training stages"
```

### Task 8: RoboCasa365 subset manifest and disk guard

**Files:**
- Create: `scripts/robocasa365/select_robottt_subset.py`
- Create: `gr00t/data/robocasa365_manifest.py`
- Create: `tests/gr00t/data/test_robocasa365_manifest.py`

**Interfaces:**
- `build_manifest(registry, budget_bytes, reserve_bytes, seed)` returns deterministic selected episodes and byte total.
- Ranking prioritizes human composite tasks, subtask count, frame count, three-camera availability, then stable task/episode ID.
- The CLI writes JSONL before downloading data and exits nonzero when the budget cannot preserve the configured reserve.

- [ ] **Step 1: Write failing deterministic-ranking and budget tests**

Use a literal registry fixture spanning atomic/composite, seen/unseen, missing camera, and varying sizes. Assert exact selected IDs, split preservation, and rejection one byte above the budget.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/gr00t/data/test_robocasa365_manifest.py -v`
Expected: FAIL because manifest module is absent.

- [ ] **Step 3: Implement metadata-only selection and CLI**

Read registry/episode metadata without video payloads, compute real bytes and length buckets, reserve model/checkpoint space, and write dataset revision, task, episode, split, duration, subtasks, frames, cameras, bytes, and checksums.

- [ ] **Step 4: Verify GREEN**

Run the manifest tests and invoke `--help`; neither command may start a download.

- [ ] **Step 5: Commit**

```bash
git add scripts/robocasa365/select_robottt_subset.py gr00t/data/robocasa365_manifest.py tests/gr00t/data/test_robocasa365_manifest.py
git commit -m "feat: select disk-safe RoboCasa365 data"
```

### Task 9: RoboCasa365 evaluation and scientific ablations

**Files:**
- Create: `gr00t/eval/robottt_metrics.py`
- Modify: `gr00t/eval/sim/robocasa365/rollout.py`
- Create: `scripts/eval/run_robottt_robocasa365.py`
- Create: `tests/gr00t/eval/test_robottt_metrics.py`
- Modify: `tests/gr00t/eval/test_robocasa365_rollout_policy.py`

**Interfaces:**
- Metrics aggregate success, completed stages, context length, subtask count, latency, GPU peak bytes, and CPU peak bytes by Seen/Unseen split.
- Evaluation modes are `base`, `history`, `gdn`, `robottt_update_off`, and `robottt_full`.
- Each rollout calls `reset_robottt_state()` before the first observation and on every reset/exception.

- [ ] **Step 1: Write failing aggregation and reset tests**

Use literal two-episode records and assert exact split averages and length-bucket counts. Force a simulator exception and assert policy fast state is cleared in `finally`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/gr00t/eval/test_robottt_metrics.py tests/gr00t/eval/test_robocasa365_rollout_policy.py -v`
Expected: FAIL because metrics/modes/reset integration are absent.

- [ ] **Step 3: Implement modes, metrics, and context sweep**

Add the five explicit modes and contexts 128, 512, 1024, 2048, and real maximum. Record model/data revisions and checkpoint hashes in every result JSON. Skip unsupported real context lengths with a recorded reason rather than stitching episodes.

- [ ] **Step 4: Verify GREEN**

Run targeted eval tests and existing RoboCasa365 simulator unit tests.

- [ ] **Step 5: Commit**

```bash
git add gr00t/eval scripts/eval/run_robottt_robocasa365.py tests/gr00t/eval
git commit -m "feat: evaluate RoboTTT on RoboCasa365"
```

### Task 10: Full verification and mll3 smoke runs

**Files:**
- Modify: `README.md`
- Create: `docs/robottt-robocasa365.md`

- [ ] **Step 1: Document exact commands and limitations**

Document HF gated-model authentication, metadata selection, Stage 1, Stage 2 ZeRO-3 offload, resume, eval modes, expected checkpoint contents, Eagle/Cosmos mismatch, and the rule against cross-episode stitching.

- [ ] **Step 2: Run static and CPU verification**

Run: `ruff format --check gr00t tests scripts`
Run: `ruff check gr00t tests scripts`
Run: `python -m pytest tests/ -m "not gpu" -v --timeout=300`
Expected: PASS.

- [ ] **Step 3: Run targeted 4090 smoke tests**

Run a Stage 1 optimizer step with `B=1,T=128`, backward, optimizer step, checkpoint save, reload, and one resumed step. Record GPU peak memory and wall time. Then run a short simulator rollout and verify one update per observation.

- [ ] **Step 4: Gate full data and model runs**

Before download/load, verify at least 38 GB remain after the selected dataset and verify access to both `nvidia/GR00T-N1.7-3B` and `nvidia/Cosmos-Reason2-2B`. If access is absent, retain passing synthetic tests and print the exact authentication gate without claiming training completion.

- [ ] **Step 5: Commit documentation and verification record**

```bash
git add README.md docs/robottt-robocasa365.md
git commit -m "docs: add RoboTTT reproduction workflow"
```

- [ ] **Step 6: Final branch review**

Inspect every commit and the complete diff against the design commit. Confirm no Fast-WAM files, credentials, downloaded datasets, model weights, generated checkpoints, or transient fast states are tracked.
