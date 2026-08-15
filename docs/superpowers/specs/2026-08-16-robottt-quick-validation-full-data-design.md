# RoboTTT Quick Validation and Full RoboCasa365 Data Design

Date: 2026-08-16

## Goal

Produce a fast, scientifically interpretable RoboTTT result on the mll3 RTX 4090 while
downloading the complete public RoboCasa365 composite datasets to Quest scratch for the
queued 8xH100 run.

## Boundaries

- Keep the GR00T N1.7 3B base model, 16 register tokens, all 16 RoboTTT layers,
  TTT-KVB, temporal RoPE, analytic inner updates, and per-episode fast state unchanged.
- The mll3 run is a quick directional experiment, not a paper-scale result.
- The complete datasets live only under Quest scratch. The 4090 keeps the existing
  disk-safe file-000 subset.
- Never train on the held-out Composite-Unseen episodes used for evaluation.
- Do not cancel Quest job 9378749; preserving its scheduler age is more important than
  changing its already-spooled shell script.

## Quest data layout

Download the three revision-pinned public datasets to:

```text
/scratch/qge0476/robottt-quest/data/full/pretrain
/scratch/qge0476/robottt-quest/data/full/composite_seen
/scratch/qge0476/robottt-quest/data/full/composite_unseen
```

The transfer runs as a `data-transfer` Slurm job without forwarding user tokens. After
download, verify each repository revision, disk size, required LeRobot metadata, parquet
files, and three camera-video directories. Generate a manifest from the downloaded tree
before any formal training consumes it.

## Quick training profile

The launcher gains explicit optional `max_steps` and `context_length` overrides. Defaults
remain the public reproduction schedule. Scheduler selection comes from the stage preset:
Stage 1 uses WSD and Stage 2 uses cosine.

The quick mll3 run uses:

```text
stage                 stage1
training data         pretrain file-000 subset only
max_steps             100
context_length        128
gradient accumulation 1
TBPTT steps           1
save_steps            25
workers               0 initially, increased only after measurement
```

The existing long run is stopped only after the corrected launcher passes tests. Its
output is retained as a clearly labeled legacy misconfigured run rather than deleted.

## Offline evaluation

Extend `open_loop_eval.py` with an explicit mode:

- `base`: load the public base checkpoint.
- `robottt_update_off`: load the quick checkpoint and disable online fast-weight updates.
- `robottt_full`: load the same quick checkpoint with online updates enabled.

Reset policy state before every trajectory. Emit deterministic JSON containing per-
trajectory MSE/MAE, aggregate MSE/MAE, mode, checkpoint, dataset, trajectory IDs, step
limit, execution horizon, denoising steps, and seed. Use the same held-out trajectory IDs
and seed for all modes.

The first gate evaluates a small fixed set from Composite-Seen and Composite-Unseen. The
quick experiment is considered informative only if training completes without NaN/OOM,
the full mode advances fast state, and the three modes produce directly comparable
records. A lower error for `robottt_full` is evidence of a positive direction, not a
paper-level success claim.

## Verification

- TDD regression tests prove Stage 1 selects WSD and explicit quick overrides are honored.
- Evaluation tests prove trajectory reset, update-off mode, and literal JSON metrics.
- Existing RoboTTT unit and training tests remain green.
- A 1-step real-model smoke precedes the 100-step run.
- Checkpoints 25/50/75/100 must exist and be loadable before evaluation.
- Quest download completion is verified from Slurm accounting and the on-disk tree.

