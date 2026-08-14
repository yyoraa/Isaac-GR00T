# GR00T N1.7 + RoboTTT + RoboCasa365 reproduction

This branch implements the publicly described RoboTTT algorithm in the public GR00T N1.7 GA codebase. It adds 16 learned register tokens and a TTT-KVB fast-weight MLP to every one of the 16 action DiT blocks, supports temporal sequence action forcing and truncated backpropagation through time (TBPTT), and provides the paper-style two-stage training schedule.

## Reproduction boundary

This is a public reproduction, not a claim that NVIDIA's private RoboTTT checkpoint or private pretraining mixture is available. The paper implementation uses an Eagle-based GR00T variant; public GR00T N1.7 GA uses the Cosmos-Reason2-2B/Qwen3-VL backbone. RoboCasa365 replaces the paper's unavailable paired human-video/correction data. The RoboTTT mechanism, register count, placement in all action-transformer blocks, TTT-KVB update, RoPE, sequence action forcing, TBPTT, Stage 1 freeze policy, and Stage 2 full tuning are retained.

Never concatenate samples across episode boundaries. `TrajectorySequenceDataset` creates windows within a single episode and emits an explicit reset mask.

## Prerequisites

Authenticate to Hugging Face and accept the licenses for both gated repositories before launching a full run:

```bash
hf auth login
hf download nvidia/GR00T-N1.7-3B --dry-run
hf download nvidia/Cosmos-Reason2-2B --dry-run
```

Keep at least 38 GB free after selecting the dataset. A full copy of the three RoboCasa365 repositories is about 117 GB. The disk-safe materialized set used on `mll3` contains complete episodes from file 000 of each official EMBER mirror:

```text
/home/yiqi/yiyun/robottt-data/subset/pretrain          211 episodes / 169,045 frames
/home/yiqi/yiyun/robottt-data/subset/composite_seen    214 episodes / 185,549 frames
/home/yiqi/yiyun/robottt-data/subset/composite_unseen  177 episodes / 139,597 frames
```

The combined manifest is `/home/yiqi/yiyun/robottt-data/subset/manifest.json`. It pins source repository revisions and SHA-256 hashes for the parquet and all three camera videos. Run `scripts/robocasa365/materialize_file0_subset.py --help` to reproduce the materialization from downloaded source payloads. Regenerate `meta/stats.json` for every materialized subset with `gr00t.data.stats.generate_stats`; do not reuse statistics from the full upstream repository.

## Stage 1

Stage 1 freezes the public base model and trains only the register tokens and RoboTTT parameters for 30,000 steps at `2e-5`, weight decay `1e-5`, with WSD scheduling. The context curriculum is 128, 512, 1024, 2048, 4096, then 8192. On a 24 GB RTX 4090, use `--tbptt-steps 1`; this changes the outer-gradient truncation interval, not the forward temporal state, the 16-layer insertion, or the register/MLP dimensions.

```bash
DATA_ROOT=/home/yiqi/yiyun/robottt-data/subset
/home/yiqi/.local/bin/uv run python gr00t/experiment/launch_robottt.py \
  --stage stage1 \
  --base-model-path /path/to/GR00T-N1.7-3B \
  --dataset-path "$DATA_ROOT/pretrain:$DATA_ROOT/composite_seen:$DATA_ROOT/composite_unseen" \
  --manifest-path "$DATA_ROOT/manifest.json" \
  --output-dir /home/yiqi/yiyun/robottt-runs/stage1 \
  --gradient-accumulation-steps 8 \
  --tbptt-steps 1
```

## Stage 2

Stage 2 fully tunes all parameters for 20,000 steps at `5e-5`, weight decay `1e-5`, cosine scheduling, and context length 1024. The launcher enables single-GPU DeepSpeed ZeRO-3 parameter and optimizer CPU offload. Start from the selected Stage 1 checkpoint:

```bash
DATA_ROOT=/home/yiqi/yiyun/robottt-data/subset
/home/yiqi/.local/bin/uv run python gr00t/experiment/launch_robottt.py \
  --stage stage2 \
  --base-model-path /home/yiqi/yiyun/robottt-runs/stage1/checkpoint-30000 \
  --dataset-path "$DATA_ROOT/pretrain:$DATA_ROOT/composite_seen:$DATA_ROOT/composite_unseen" \
  --manifest-path "$DATA_ROOT/manifest.json" \
  --output-dir /home/yiqi/yiyun/robottt-runs/stage2 \
  --gradient-accumulation-steps 8 \
  --tbptt-steps 1
```

To resume an interrupted stage, repeat the same command with `--resume-from-checkpoint`. Resume is fail-closed: `robottt_state.json` must agree on stage and manifest hash, while the normal trainer checkpoint restores model, optimizer, scheduler, RNG, and sampler state.

## Evaluation

Run simulator rollouts with the checkpoint and a registered RoboCasa365 environment. Policy reset clears every layer's fast state at episode boundaries. The action head advances the fast state once per observation even though flow matching invokes it multiple times for the same observation.

```bash
/home/yiqi/.local/bin/uv run python gr00t/eval/rollout_policy.py \
  --model-path /home/yiqi/yiyun/robottt-runs/stage2/checkpoint-20000 \
  --env-name robocasa365_panda_omron/CloseFridge_PandaOmron_Env \
  --robocasa-split target-composite-unseen \
  --n-episodes 10 --n-envs 1 --n-action-steps 8
```

Report the same episode set in all five modes: `base`, `history`, `gdn`, `robottt_update_off`, and `robottt_full`. Sweep context lengths 128, 512, 1024, 2048, and the real episode maximum. After saving one JSON object per episode to JSONL, create a hash-pinned aggregate record with:

```bash
/home/yiqi/.local/bin/uv run python scripts/eval/run_robottt_robocasa365.py \
  --checkpoint /path/to/checkpoint \
  --manifest /home/yiqi/yiyun/robottt-data/subset/manifest.json \
  --episode-records /path/to/episodes.jsonl \
  --output /path/to/result.json \
  --mode robottt_full --context-length 1024
```

## Verification record

The full-dimension action head has 814,726,160 DiT parameters. On the RTX 4090, a `B=1, T=128`, BF16 Stage 1 optimizer/checkpoint/reload/resume smoke with all 16 RoboTTT layers and 16 register tokens completed with TBPTT 1: first loss `2.03326416015625`, resumed loss `2.03070068359375`, peak allocated CUDA memory `9,391,261,184` bytes, and wall time `26.63 s`. The smoke checkpoint is `/tmp/robottt-stage1-gpu-smoke.pt` and is intentionally not tracked.

The synthetic smoke does not load the gated VLM backbone and is not a quality metric. A full GR00T train or simulator success-rate claim requires successful access to both gated NVIDIA repositories, a trained Stage 2 checkpoint, and actual rollouts.
