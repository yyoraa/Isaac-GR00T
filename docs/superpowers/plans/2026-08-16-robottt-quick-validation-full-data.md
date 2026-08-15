# RoboTTT Quick Validation and Full Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download complete RoboCasa365 composite data on Quest and produce a comparable base/update-off/full quick result on mll3.

**Architecture:** Keep full RoboTTT model semantics while adding narrow launcher overrides for a 100-step profile. Reuse the existing open-loop evaluator, adding explicit online-update control, per-trajectory reset, and JSON output. Keep bulk data on Quest scratch and the small subset on mll3.

**Tech Stack:** Python 3.12, PyTorch, Transformers Trainer, DeepSpeed, Tyro, pytest, Slurm, Hugging Face Hub.

**Spec:** `docs/superpowers/specs/2026-08-16-robottt-quick-validation-full-data-design.md`

## Global Constraints

- Preserve 16 register tokens and all 16 RoboTTT layers.
- Never concatenate episodes or train on the held-out Composite-Unseen evaluation episodes.
- Do not cancel Quest job `9378749`.
- Do not delete model weights, datasets, checkpoints, or prior logs.
- Every behavior change follows RED/GREEN TDD.

---

### Task 1: Correct stage scheduling and add quick overrides

**Files:**
- Modify: `gr00t/experiment/launch_robottt.py`
- Modify: `tests/gr00t/experiment/test_robottt_training.py`

**Interfaces:**
- Consumes: `RoboTTTTrainingConfig.for_stage(stage)`.
- Produces: `RoboTTTLaunchConfig.max_steps: int | None`, `context_length: int | None`, and a `Config` whose scheduler follows `preset.scheduler`.

- [ ] **Step 1: Write failing launcher tests**

```python
def test_stage1_launcher_uses_wsd_and_honors_quick_overrides(tmp_path):
    launch = _launch(tmp_path, stage="stage1", max_steps=100, context_length=128)
    config = build_robottt_config(launch)
    assert config.training.lr_scheduler_type == "wsd"
    assert config.training.max_steps == 100
    assert config.data.context_length == 128
    assert config.data.sequence_stride == 128

def test_launcher_rejects_nonpositive_quick_overrides(tmp_path):
    with pytest.raises(ValueError, match="max_steps"):
        build_robottt_config(_launch(tmp_path, max_steps=0))
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/gr00t/experiment/test_robottt_training.py -q`

Expected: failure because the new dataclass arguments do not exist and Stage 1 still maps to cosine.

- [ ] **Step 3: Implement minimal launcher behavior**

Add optional fields, validate positive values, map `lr_scheduler_type = preset.scheduler`,
and apply overrides only when non-`None`.

- [ ] **Step 4: Run GREEN**

Run: `.venv/bin/python -m pytest tests/gr00t/experiment/test_robottt_training.py -q`

Expected: all tests pass.

### Task 2: Make open-loop evaluation mode-aware and machine-readable

**Files:**
- Modify: `gr00t/eval/open_loop_eval.py`
- Create: `tests/gr00t/eval/test_robottt_open_loop_eval.py`

**Interfaces:**
- Produces: `EvaluationMode = Literal["base", "robottt_update_off", "robottt_full"]`.
- Produces: `configure_robottt_evaluation_mode(policy, mode) -> None`.
- Produces: `write_evaluation_json(path, config, per_trajectory) -> Path`.

- [ ] **Step 1: Write failing behavior tests**

Use a real lightweight stateful policy double whose `reset()` and
`set_robottt_online_updates()` mutate observable state. Assert update-off disables updates,
full enables them, every trajectory begins after reset, and JSON aggregate values equal
hand-computed literals.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/gr00t/eval/test_robottt_open_loop_eval.py -q`

Expected: import failure for the new functions.

- [ ] **Step 3: Implement mode and JSON helpers**

Expose `--mode`, `--output-json`, and `--seed`; reset before each trajectory; return
per-trajectory records from the evaluator; write deterministic JSON with means computed
from those records.

- [ ] **Step 4: Run GREEN and existing evaluation tests**

Run: `.venv/bin/python -m pytest tests/gr00t/eval/test_robottt_open_loop_eval.py tests/gr00t/eval/test_robottt_metrics.py tests/gr00t/eval/test_robocasa365_rollout_policy.py -q`

Expected: all tests pass.

### Task 3: Deploy and run the quick 4090 experiment

**Files:**
- Runtime output: `/home/yiqi/yiyun/robottt-runs/quick-stage1-100`
- Runtime results: `/home/yiqi/yiyun/robottt-runs/quick-eval`

**Interfaces:**
- Consumes: corrected launcher and evaluation CLI.
- Produces: checkpoints 25/50/75/100 and three comparable JSON result files.

- [ ] **Step 1: Push the tested branch and fast-forward mll3 into an isolated worktree**
- [ ] **Step 2: Run a one-step real-model smoke with pretrain-only data**
- [ ] **Step 3: Preserve the old output and stop PID 200045 after the smoke passes**
- [ ] **Step 4: Launch 100 steps with context 128, accumulation 1, TBPTT 1, save every 25 steps**
- [ ] **Step 5: Validate checkpoint loadability and run fixed held-out trajectories in all three modes**
- [ ] **Step 6: Compare literal MSE/MAE records and verify full-mode fast state advancement**

### Task 4: Verify the full Quest download and prepare formal data metadata

**Files:**
- Runtime data: `/scratch/qge0476/robottt-quest/data/full`
- Runtime logs: `/scratch/qge0476/robottt-quest/logs/full-data-9404435.*`

**Interfaces:**
- Consumes: revision-pinned Hugging Face datasets.
- Produces: complete LeRobot trees and a verified manifest for later 8xH100 training.

- [ ] **Step 1: Monitor Slurm job 9404435 to terminal state**
- [ ] **Step 2: Verify the three revisions and required metadata/parquet/video directories**
- [ ] **Step 3: Compute episode/frame/size statistics without hashing every video twice**
- [ ] **Step 4: Generate and validate a full-data manifest with pretrain/train/eval roles**
- [ ] **Step 5: Keep the queued H100 job unchanged until its code/data compatibility is verified**

## Self-review

- Spec coverage: launcher correctness, quick training, comparable evaluation, full Quest download, checkpoint verification, and queue preservation are each mapped to a task.
- Placeholder scan: no deferred implementation placeholders remain.
- Type consistency: launcher override names and evaluation helper names are used consistently.

