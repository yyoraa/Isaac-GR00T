# GR00T N1.7 + RoboTTT + RoboCasa365 Design

Date: 2026-08-15

## 1. Goal and reproduction boundary

Implement the public RoboTTT algorithm in the official NVIDIA Isaac-GR00T repository,
using the public GR00T N1.7 GA model and RoboCasa365 trajectories. The implementation
must include the paper's 16 learned register tokens, a TTT layer in every one of the 16
action-model transformer blocks, sequence action forcing, learned initial fast weights,
TTT-KVB, temporal RoPE, per-episode online state, and truncated backpropagation through
time (TBPTT).

"Complete reproduction" here means reproducing the architecture, objectives, update
semantics, two training stages, evaluation protocol, and relevant ablations that are
specified publicly. It does not mean claiming numerical reproduction of the paper's
private checkpoint or private pretraining data. RoboCasa365 replaces those unavailable
training sources.

There is one unavoidable model-version difference. The RoboTTT paper describes its
GR00T N1.7 instance as using an Eagle vision-language backbone. The current public N1.7
GA repository uses `nvidia/Cosmos-Reason2-2B` (Qwen3-VL). This project targets the public
GA model without pretending that the backbone is identical to the paper's internal or
pre-GA checkpoint. The action-model dimensions and 16-layer insertion scheme are taken
from the public GA code.

Primary references:

- RoboTTT paper: <https://arxiv.org/html/2607.15275>
- Isaac-GR00T: <https://github.com/NVIDIA/Isaac-GR00T>
- RoboCasa365: <https://robocasa.ai/robocasa365>

## 2. Baseline and compatibility

The base model is `nvidia/GR00T-N1.7-3B`. Existing N1.7 checkpoints must still load.
RoboTTT parameters are additive and are initialized so that enabling the module begins
near the original model behavior. A configuration switch must support a strict baseline
path with RoboTTT and register tokens disabled.

The existing RoboCasa Panda Omron embodiment configuration is reused. Its three cameras,
state fields, action fields, action horizon, normalizers, and simulator integration remain
authoritative. RoboTTT changes temporal modeling; it does not redefine the embodiment.

## 3. Per-timestep token construction

For robot timestep `t`, build the action-model token sequence

`X_t = [R_t(16), q_t(1), A_t(40)]`,

where:

- `R_t` is a bank of 16 learned register embeddings copied across batch and trajectory
  time. These are real trainable tokens, not action tokens selected as a substitute.
- `q_t` is the existing encoded robot-state token.
- `A_t` is the existing noised action-chunk token sequence.
- `Phi_t` contains the language and image tokens produced by the public N1.7 VLM.

Within a robot timestep, the original alternating self-attention and VLM cross-attention
behavior remains intact. The transformer receives `[R_t, q_t, A_t]`; its cross-attention
uses `Phi_t` where the original block calls for it. Raw VLM tokens are never placed in
the temporal TTT stream.

The action decoder consumes only the final 40 action-token outputs. Register and state
outputs are excluded by an explicit slice, with shape assertions to prevent silent token
offset errors.

## 4. RoboTTT layer

Each of the 16 `BasicTransformerBlock` instances receives a RoboTTT module. It is called
after the block's attention residual is formed and before the feed-forward sublayer, in
every layer rather than only at selected depths.

For each layer, the fast model is an independent two-layer MLP:

- input width: 1536
- hidden width: 3072
- output width: 1536
- activation: GeLU

This is approximately 9.44 million fast parameters per transformer layer, matching the
paper's roughly 10M description. Width, depth, or layer coverage must not be reduced to
make the 4090 run; memory is addressed by segmentation and offload instead.

Each RoboTTT layer also owns trainable slow parameters:

- query, key, and value projections
- learned initial fast weights `W0`
- a base inner-loop learning rate of 0.1 multiplied by a learned positive multiplier
- a vector residual gate `alpha`, initialized through `tanh(alpha) = 0.001`
- temporal RoPE with `theta = 10000`

For timestep `t`, apply temporal RoPE to the projected query and key streams. Compute the
TTT-KVB mean-squared reconstruction loss between the fast model's key-conditioned output
and projected values. Perform one differentiable gradient-descent update of the fast MLP,
then evaluate the updated MLP on the query stream. Add the gated result to the transformer
residual. This is update-then-apply, not apply-then-update.

Fast weights initialize from learned `W0` exactly once at an episode boundary. They carry
across robot timesteps and reset between episodes or independent simulator rollouts.

## 5. Diffusion and online-update semantics

Training uses sequence action forcing. Each trajectory timestep receives its own flow time
and noise sample, independent of other timesteps in the same trajectory:

`tau_t = 0.999 * (1 - u_t), u_t ~ Beta(1.5, 1)`.

The same sampled `tau_t` applies to the 40 action tokens within that robot timestep. It
must not be broadcast across the entire temporal sequence.

At inference, the flow solver evaluates the action model multiple times for one robot
observation. RoboTTT fast state advances exactly once per observation, not once per flow
function evaluation. The first solver evaluation computes the update and produces the
next fast state; later solver evaluations for that observation use the resulting state
without further updates. The state is committed after the action chunk is produced. A
counter-based test enforces one update per environment observation. This is an explicit
implementation choice because the paper does not spell out repeated denoising-call
handling.

## 6. Long-trajectory training interface

A trajectory batch contains `[B, T, ...]` values for all three images, state, language,
action chunks, episode identity, valid-timestep mask, and action-loss mask. The VLM and
within-timestep action computation may flatten `B*T` for efficiency, but the TTT scan must
restore temporal order and never cross an episode boundary.

Variable-length episodes are padded only within a batch. Padding cannot update fast state
or contribute to TTT, flow-matching, or action losses. Sequence windows preserve absolute
trajectory ordering and never concatenate different episodes to manufacture longer
contexts.

TBPTT divides a long sequence into ordered segments. Fast weights carry from one segment
to the next, while their computation graph is detached at each segment boundary. The
segment length is a memory-control parameter and does not change the model definition.

The data API also accepts a context-only mask. Context frames may update the fast model
while being excluded from the outer action loss. This supports the paper's human-video
context and DAgger-style protocols when compatible paired data later becomes available.
RoboCasa365 alone does not provide the same paired human-video/correction supervision, so
those numerical results will not be claimed.

## 7. RoboCasa365 data plan

Use metadata-first selection because mll3 currently has only about 83 GB free. The data
preparation command must inspect the registry and episode metadata before downloading
video shards, calculate the expected storage requirement, and refuse to exceed a
configurable budget. The initial budget is 45 GB so that model weights, environments,
checkpoints, and working space remain available.

The main long-context corpus prioritizes human demonstrations from composite tasks:

- pretraining composite tasks for Stage 1 diversity
- target Composite-Seen tasks for validation and adaptation analysis
- target Composite-Unseen tasks held out for generalization evaluation
- episodes with more subtasks, longer duration, and reliable three-camera coverage

Task selection is deterministic and saved as a manifest with dataset revision, task IDs,
episode IDs, split, duration, subtask count, frame count, camera availability, and checksum.
MimicGen atomic trajectories are excluded from the main long-context result because they
do not test the central long-horizon claim; they may be used as a separately reported
short-context ablation.

The loader reports the real length distribution and supported context buckets. Training
at 8K context is enabled only if individual episodes genuinely contain at least 8K robot
timesteps after the chosen sampling rate. Otherwise the maximum real episode length is
used and the limitation is reported. Episodes are never stitched together.

## 8. Two-stage optimization

### Stage 1: sequence-layer pretraining

- 30,000 optimizer steps
- only registers, RoboTTT modules, `W0`, Q/K/V projections, learning-rate multipliers,
  and gates are trainable
- all public GR00T N1.7 base parameters are frozen
- AdamW, weight decay `1e-5`
- WSD schedule with peak learning rate `2e-5`
- context curriculum increases through the real supported buckets toward 8K when possible
- sequence action forcing and TBPTT enabled

Stage 1 is expected to produce an independently testable checkpoint. Its effect is
measured before Stage 2 rather than assumed.

### Stage 2: full-model post-training

- 20,000 optimizer steps
- all model parameters trainable
- 1K temporal context, capped to the longest real supported episode where necessary
- AdamW, weight decay `1e-5`
- cosine schedule with peak learning rate `5e-5`
- sequence action forcing and TBPTT enabled

The exact full-parameter Stage 2 path is retained on the single RTX 4090. It uses
DeepSpeed ZeRO-3 with CPU parameter and optimizer offload, activation checkpointing,
mixed precision, microbatch size one, and gradient accumulation. The host has sufficient
RAM for the intended offload path, but throughput will be far below the paper's 8-GPU
setup. The implementation must fail clearly if memory is still insufficient; it must not
silently replace full fine-tuning with LoRA.

## 9. Checkpoints and resumption

Training checkpoints include all slow model weights, registers, learned `W0`, optimizer,
scheduler, scaler, sampler/curriculum state, global step, dataset manifest hash, and RNG
state. Fast weights created while scanning a particular episode are transient and are not
part of normal training checkpoints. An optional debug artifact may serialize them only
when explicitly requested.

Resuming must reproduce the next sampled batch and curriculum bucket. Loading a base N1.7
checkpoint without RoboTTT keys uses documented initialization; loading a RoboTTT
checkpoint with incompatible register count, hidden width, or layer count is an error.

## 10. Evaluation

Use the repository's existing RoboCasa365 simulator integration. Report at least:

- task success rate and stage/progress completion
- Composite-Seen and Composite-Unseen separately
- results bucketed by subtask count and trajectory/context length
- adaptation scaling at 128, 512, 1K, 2K, and the maximum genuine supported context
- inference latency and peak GPU/CPU memory

Required baselines and ablations are:

- public GR00T N1.7 single-step baseline
- history-window baseline without fast-weight updates
- gradient-descent-on-input or equivalent GDN baseline where feasible
- RoboTTT architecture with online updates disabled
- full RoboTTT
- Stage 1 checkpoint before Stage 2
- Stage 2 checkpoint

No result is labeled a paper reproduction unless its model version, data, context length,
checkpoint, and evaluation episodes are recorded.

## 11. Tests and acceptance criteria

Unit tests must cover:

- exactly 16 register tokens with correct batch/time expansion and gradients
- strict base path when RoboTTT/registers are disabled
- insertion after attention and before feed-forward in all 16 layers
- TTT-KVB update-then-apply ordering
- meta-gradients reaching learned `W0`
- temporal RoPE and `theta = 10000`
- independent flow times/noise across trajectory timesteps
- TBPTT state carry with graph detachment
- padding and episode-boundary reset with no state leakage
- exactly one online update per observation across all denoising evaluations
- context-only/action-loss masking
- checkpoint save/load and deterministic resume

Data tests must cover schema conversion for the Panda Omron embodiment, three-camera
alignment, episode boundaries, manifest reproducibility, context bucketing, and storage
budget rejection.

Integration acceptance requires:

1. a tiny CPU synthetic-sequence training test;
2. unchanged base-model open-loop output shapes with RoboTTT disabled;
3. one Stage 1 optimizer step at context 128 on the mll3 RTX 4090;
4. checkpoint round-trip and resume on that GPU;
5. a short RoboCasa365 simulator rollout proving reset and one-update-per-observation;
6. ruff, repository CPU tests, new RoboTTT tests, and the targeted GPU smoke tests pass.

## 12. Failure handling

- If Hugging Face access to the gated Cosmos backbone is absent, stop at the model-load
  gate with the exact required repository and authentication action; code and synthetic
  tests continue independently.
- If a planned dataset subset exceeds the free-space guard, select fewer tasks using the
  recorded deterministic ranking rather than partially downloading an untracked subset.
- If GPU memory is insufficient, reduce TBPTT segment length and increase accumulation;
  do not remove registers, reduce the fast MLP, or skip transformer layers.
- Any simulator reset, timeout, exception, or episode-ID change clears all fast state.
- NaN/Inf checks cover inner loss, fast gradients, updated fast weights, and outer loss,
  and produce a diagnostic checkpoint before aborting.

## 13. Expected implementation surfaces

The implementation will add a dedicated RoboTTT module under `gr00t/model/modules/`, wire
it into `gr00t/model/modules/dit.py` and the N1.7 action head, extend N1.7 configuration,
add long-trajectory RoboCasa365 data/collation support, add two-stage training launch
configs, and add unit/integration tests under `tests/`. Exact file boundaries are fixed in
the implementation plan after validating the current training and data abstractions.

The original Fast-WAM experiment remains separate and untouched. This branch's source of
truth is the official Isaac-GR00T repository at commit `376ba89` plus the changes described
here.
