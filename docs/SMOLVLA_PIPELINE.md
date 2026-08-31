# SmolVLA Offline Pipeline

Status: **Steps 4-6B completed; formal five-epoch Step 6B run and held-out evaluation passed on 2026-08-31**

This document defines the design and execution record for the VLA-only portion of Stage 7. It starts
from the validated local `LeRobotDataset` produced by steps 1-3 and ends with reviewed offline
base-model and PEFT artifacts. It does not launch Isaac Sim or Isaac Lab, import
`embodied_ai.sim`, or alter the simulator environment.

## Fixed inputs and constraints

- Dataset root:
  `/root/autodl-tmp/EmbodiedAI/datasets/stage7-franka-pick-place-batch-v1`
- LeRobot repository identity: `embodiedai/franka-pick-place-stage7-batch-v1`
- Dataset profile: `franka-pick-place-smolvla-v1`
- Dataset contents: 20 episodes, 2,138 frames, five exact instructions, and 20 Hz control.
- Expanded Step 6B candidate:
  `/root/autodl-tmp/EmbodiedAI/datasets/stage7-franka-pick-place-batch-v2-100`, repo ID
  `embodiedai/franka-pick-place-stage7-batch-v2-100`, with 100 episodes, 10,707 frames, five exact
  instructions, and 20 Hz control. It passed the same mapping and independent validation gates on
  2026-08-30; it does not alter the fixed 20-episode Step 4/5/6A execution record above.
- Policy inputs: 9D Franka joint position, one front RGB image with shape `(3, 224, 224)`, and
  the exact instruction string. Joint velocity remains excluded and cube position remains
  privileged simulator state.
- Policy output: the canonical normalized 7D action in the order defined by the action contract.
- VLA runtime: the locked Python 3.12 environment under `$EMBODIEDAI_ENVS/vla`.
- Base model: the locally pinned SmolVLA checkpoint revision
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`, with its recorded model SHA-256 and pinned local
  SmolVLM tokenizer/config dependency.
- Large datasets, model files, processor artifacts, predictions, checkpoints, and logs stay under
  `/root/autodl-tmp/EmbodiedAI`; only small source, configuration, tests, and documentation belong
  in Git.

The base checkpoint metadata describes 6D state, 6D action, and three 256 x 256 cameras. The
project dataset is 9D state, 7D action, and one 224 x 224 camera. Loading the checkpoint's saved
processor pipeline unchanged would therefore be incorrect. Step 4 must explicitly bind a project
feature configuration and project statistics while retaining compatible pretrained weights.

## Runtime boundary

All three steps run on the VLA side:

```text
validated LeRobotDataset
  -> project feature binding + selected dataset statistics
  -> preprocessing
  -> SmolVLA base or PEFT adapter
  -> postprocessing to the canonical 7D action
  -> offline predictions, metrics, and provenance
```

Dependency-light action/observation contracts may be imported. Isaac/Isaac Lab modules, assets,
environment creation, and simulator stepping are outside this pipeline. Step 8 will later own the
online process boundary, action scheduling, and closed-loop Isaac interaction.

The accepted execution used the restarted GPU-mode allocation after a fresh resource preflight.
No package or lock change was required. Future full dataset media processing, base inference, and
fine-tuning runs still require a fresh GPU-mode resource preflight; any proposed package or lock
change receives a separate review before execution.

## Step 4 — SmolVLA preprocessing and postprocessing

### Goal

Build one reusable, configuration-driven processor boundary shared by offline base inference,
fine-tuning, and the later online adapter. The module converts a raw LeRobot sample into the
inputs expected by SmolVLA and converts model output back into the existing canonical action
schema without simulator dependencies.

### 4.1 Dataset and model binding

1. Require the expected dataset repository identity, mapping profile, conversion provenance,
   validation result, feature names, dimensions, image metadata, control rate, and statistics.
2. Require the pinned base model revision, checksum, tokenizer/config revision, and locked VLA
   environment metadata. Loading is local/offline after the assets are present.
3. Derive a project policy configuration with:
   - `observation.state`: 9 values;
   - `observation.images.front`: one RGB tensor;
   - `action`: 7 values;
   - one observation step;
   - a 50-step action chunk at 20 Hz;
   - the model's existing 32-value state/action padding capacity.
4. Explicitly replace the checkpoint's 6D/three-camera input feature declaration and 6D output
   declaration. Missing checkpoint cameras must not be synthesized as project observations.
5. Keep the project's normalized end-effector delta/gripper action semantics. Do not enable
   ALOHA-specific joint transforms or silently convert actions to absolute poses.

### 4.2 Preprocessing sequence

The planned sequence is:

1. **Raw-sample validation.** Check required keys, shapes, dtypes, finite values, task text, image
   range, and component order before changing the sample.
2. **Temporal assembly.** For inference, use only the current observation and instruction. For
   training, ask `LeRobotDataset` for action offsets `0..49` at 20 Hz and preserve
   `action_is_pad`; a window may not cross an episode boundary.
3. **Batch formation.** Add the batch dimension consistently to state, image, language, masks,
   and training actions.
4. **Language processing.** Preserve the exact dataset `task` string, apply SmolVLA's required
   newline convention, tokenize with the pinned local tokenizer, enforce the configured maximum
   token length, and retain the attention mask.
5. **Image processing.** Accept LeRobot's decoded CHW float image in `[0, 1]`; validate it before
   the model performs its configured padded resize and `[0, 1]` to `[-1, 1]` conversion. The
   project processor must not normalize the image a second time.
6. **State/action normalization.** Normalize state and training actions with an explicit
   statistics bundle. Step 5 may use the validated corpus statistics for a base compatibility
   baseline. Step 6 must use statistics computed from the training split only; validation/test
   frames must not influence training normalization.
7. **Device and dtype placement.** Move tensors and masks to the selected device in one processor
   step. The first accepted implementation uses the reviewed SmolVLA dtype/AMP setting rather
   than introducing an unreviewed precision change.
8. **Processed-batch validation.** Confirm batch alignment, language masks, action padding masks,
   model feature keys, and finite tensors. SmolVLA owns internal zero-padding from 9/7 values to
   its maximum dimension of 32.

The processor factory accepts the statistics source explicitly and records its scope and hash.
It must not silently fall back from project statistics to the base checkpoint's 6D statistics.

### 4.3 Postprocessing sequence

1. Accept either one action from `select_action` or a 50-step chunk from the chunk-prediction
   path, with the form stated explicitly by the caller.
2. Remove only model-internal padding and require the project action dimension to be exactly 7.
3. Undo model normalization with the same explicit project statistics bundle used by the
   preprocessor.
4. Move the result to CPU and validate finite values, shape, component order, and canonical
   normalized bounds.
5. Preserve the action in the contract's normalized end-effector-delta/gripper representation.
   Postprocessing does not turn it into a metric pose or simulator command.
6. Report out-of-range output; do not silently clamp it during offline evaluation. Online safety
   clipping and action scheduling belong to the Stage 8 adapter.

### 4.4 Proposed implementation units

The exact filenames remain subject to implementation review, but responsibilities should remain
separate:

- a project SmolVLA profile/config binding under `src/embodied_ai/policies/smolvla/`;
- a processor factory and raw/processed/postprocessed validators in the same package;
- small configuration under `configs/policy/` for model, feature, and processor choices;
- processor unit tests under `tests/policies/`, including failure cases;
- a bounded VLA-side processor validation entry point under `scripts/train/` or
  `scripts/evaluate/`.

### 4.5 Validation and exit gate

- Static tests reject wrong feature names, the checkpoint's original 6D/three-camera statistics,
  incorrect image layouts, non-finite values, cross-episode action windows, and action-dimension
  drift.
- A real-dataset sample passes preprocessing without privileged fields.
- Action normalize/unnormalize round trips meet a stated numerical tolerance for valid frames.
- Serialized processor configuration and statistics reload deterministically and retain hashes.
- CPU structural tests and GPU processor parity are distinguished in the report; a CPU-only pass
  is not presented as base-model inference.

Step 4 exits only when one reviewed processor boundary is suitable for both steps 5 and 6.

## Step 5 — SmolVLA base offline inference

### Goal

Deploy the already pinned base SmolVLA checkpoint through the Step 4 project processors and run
offline inference on the existing dataset. This is a compatibility and baseline measurement, not
a claim that an unfine-tuned policy can complete the Isaac task.

### 5.1 Loader and reproducibility

1. Perform the GPU/resource and data-disk preflight, enter only the VLA environment, and force
   model/tokenizer loading from the pinned local revisions.
2. Verify model, tokenizer/config, dataset, mapping profile, validation report, processor config,
   and statistics hashes before inference.
3. Load the base weights into the project-bound 9D/one-camera/7D configuration and fail closed on
   missing or unexpected compatibility assumptions.
4. Fix run seed and inference-noise generation. Because SmolVLA action sampling is stochastic,
   deterministic comparison requires recording or regenerating the same noise tensor.
5. Reset policy action queues at every episode boundary and before independent-frame probes.

### 5.2 Two offline inference passes

1. **Bounded compatibility probe.** Select a committed set of episode/frame anchors covering all
   instruction variants and representative cube/goal settings. Verify raw-to-model-to-action
   execution, finite 7D output, repeatability, latency after warm-up, and peak GPU memory.
2. **Dataset baseline pass.** Run causal observations through the base policy without supplying
   ground-truth actions to the inference input. Compare the predicted first action with the expert
   action at each evaluated frame. Where a full 50-step chunk is evaluated, compare only valid
   horizon elements and apply `action_is_pad` at episode tails.

The evaluation should aggregate normalized-action MAE/RMSE by component, gripper agreement, bound
violation rate, finite-output rate, and results by episode/instruction. Chunk metrics and
single-step metrics remain separate. A deterministic episode-level validation split should be
frozen before fine-tuning so this base result can be reused as the unchanged baseline.

Offline action error is diagnostic: it cannot measure pick-and-place success, closed-loop
recovery, or stability. Those results belong to Stage 8.

### 5.3 Artifacts and exit gate

Store external run artifacts below `$EMBODIEDAI_RUNS` or `$EMBODIEDAI_ARTIFACTS`, including:

- a run manifest containing Git revision, lock hash, hardware, seeds, all data/model/processor
  identities, and command/config identity;
- predictions with episode/frame identity and the corresponding valid expert targets;
- aggregate and per-group metrics;
- warm-up-separated latency and peak allocated/reserved GPU memory;
- captured warnings and known compatibility limitations.

Step 5 exits when the pinned base model reloads offline, the selected real observations produce
repeatable finite canonical 7D actions, prediction/metric artifacts are reproducible, and the run
fits the reviewed GPU resource envelope. Poor imitation accuracy is an expected reportable base
result, not a reason to alter the data contract or hide the baseline.

## Step 6 — SmolVLA fine-tuning

Step 6 is split into a small technical-feasibility run and a later formal dataset-driven run.
Neither substep uses Isaac. Closed-loop simulator evaluation starts only in Stage 8.

### Step 6A — Lightweight PEFT feasibility on the 20 episodes

#### Scope

- Freeze a deterministic episode-level train/validation manifest before optimization. The initial
  candidate is 15 training and 5 validation episodes so that validation can include one episode
  for each instruction variant while covering cube/goal settings as evenly as possible. The exact
  episode IDs are reviewed and committed;
  frames from one episode may not appear in both splits.
- Compute and persist normalization statistics from the training episodes only, then rebuild both
  train and validation processors from that same statistics bundle.
- Start with LoRA/PEFT rather than full-parameter fine-tuning. Use the already validated local
  base revision and a conservative rank such as 2 or 4; record target modules and trainable
  parameter counts explicitly.
- First overfit one or a few batches to verify action-horizon formation, padding masks, finite
  loss/gradients, optimizer updates, save/reload behavior, and loss movement.
- Then run a bounded feasibility job with batch size 1, reviewed gradient accumulation, fixed
  seeds, periodic validation, and an initial ceiling of 200 optimizer steps. Resource preflight
  may lower that ceiling; increasing it requires a reviewed configuration change.

The 20-episode run proves that the module works end to end. It is too small to support a strong
generalization claim or to select a production policy.

#### Artifacts and exit gate

- Versioned split manifest, training config, seed, and train-only statistics.
- Base revision/checksum plus mapping, dataset, processor, and lock provenance.
- Loss, learning rate, gradient/parameter-update evidence, validation metrics, timing, and peak
  memory.
- PEFT adapter/checkpoint and processor artifacts outside Git.
- A clean-process reload that reproduces offline prediction and records base-versus-adapter
  metrics on the same held-out frames.

Step 6A exits when the bounded job is finite and reproducible, the adapter alone can be saved and
reloaded over the pinned base, output remains contract-compatible, and limitations from the tiny
corpus are stated. A lower training loss alone is not sufficient.

### Step 6B — Formal fine-tuning on an expanded dataset

#### Prerequisites and scope

1. Stage 6 publishes a larger, more diverse immutable expert corpus and Stage 7 steps 1-3 convert
   and independently validate a new versioned `LeRobotDataset`. **Completed 2026-08-30:** 100
   successful episodes, 10,707 frames, 10 cube resets, 10 goals, and five balanced instructions
   were converted to `stage7-franka-pick-place-batch-v2-100`. Full provenance/table/statistics
   validation and 200 deterministic boundary-image decodes passed.
2. Freeze episode- or scenario-level train/validation/test splits that cover instruction, cube,
   goal, and later task diversity without frame leakage. The test split remains untouched until
   configuration selection is complete.
3. Recompute training-only statistics and repeat the Step 5 base baseline on the frozen
   validation/test protocol.
4. Scale the Step 6A PEFT pipeline with reviewed batch/accumulation, training length, validation
   cadence, checkpoint retention, and early-stopping/model-selection rules.
5. Compare a small, explicit set of PEFT configurations. Full-parameter fine-tuning is deferred
   unless dataset scale, compute budget, and a separate review justify it.
6. Select by validation results, evaluate the chosen checkpoint once on the locked test split,
   and compare base, Step 6A, and formal adapters with the same Step 5 offline metrics.

#### Exit gate

Step 6B exits with a selected, reloadable adapter/checkpoint; complete data/split/model/processor
provenance; reproducible held-out offline metrics; and a documented configuration suitable for
Stage 8 closed-loop evaluation. Formal offline improvement does not itself establish simulator
task success.

## Step 6B implementation record — 2026-08-31

`configs/policy/smolvla_stage7_split_v2_100.json` freezes an 80/10/10 whole-episode split. Each
validation and test partition contains all 10 cube resets, all 10 goals, and exactly two examples
of each instruction; training contains eight examples of every reset/goal and 16 of every
instruction. Source episode identities and conversion provenance are hash-checked, and the formal
balance is enforced in code.

Training-only processors were published under
`$EMBODIEDAI_ARTIFACTS/stage7/processors/franka-pick-place-smolvla-v2-100-train80-v1`. The formal
rank-8/alpha-16 LoRA run used batch size 16, five epochs, 536 optimizer steps per epoch, and 2,680
steps total. It trained 371,328 of 450,417,504 parameters in 2,037.8 seconds. Fixed validation
flow-matching loss decreased monotonically from 3.6908 before training to 0.6298, 0.4873, 0.4011,
0.3580, and 0.3259 after epochs 1-5. Peak allocated/reserved CUDA memory was about 3.13/3.25 GiB.
All epoch adapters are retained; epoch 5 is selected and published at
`$EMBODIEDAI_CHECKPOINTS/stage7-step6b/smolvla-lora-r8-100ep-5epochs-v1`.

Clean-process offline reload was exactly repeatable. On validation anchors, base versus Step 6B
MAE/RMSE was 0.5103/0.7100 versus 0.2691/0.5168. On the untouched test anchors it was
0.5212/0.7158 versus 0.2974/0.5498, and gripper-sign agreement improved from 0.3333 to 0.6667.
The formal adapter still produced an 8.10% test bound-violation rate and therefore remained subject
to the unchanged Stage 8 hard-action gate.

## Implementation record — 2026-08-30

The implementation is configuration-driven and VLA-only:

- `src/embodied_ai/policies/smolvla/profile.py` defines the 9D/one-camera/7D project binding and
  rejects incompatible dataset/model metadata;
- `split.py` and `config.py` validate the committed whole-episode split and run configuration;
- `processing.py` owns train/inference preprocessing, postprocessing, diagnostics, statistics,
  serialization, and reload;
- `runtime.py` verifies and loads only the pinned local base/VLM assets;
- `dataset.py` opens the reviewed `LeRobotDataset` with policy-specific action horizons;
- `offline.py` owns deterministic inference and action-space metrics;
- `training.py` owns the bounded LoRA wrapper, optimizer, fixed-input micro-overfit check, and
  held-out flow-matching loss.

The accepted split is 15 training episodes (1,599 frames) and five validation episodes. Validation
contains exactly one episode per instruction, all four goal settings, and four of five cube reset
positions. Statistics are computed from training episodes only and serialized with the processor.

Step 4 published
`/root/autodl-tmp/EmbodiedAI/artifacts/stage7/processors/franka-pick-place-smolvla-v1-train-split-v1`.
The real 50-step action horizon, padding masks, tokens, image/state/action shapes, save/reload
parity, and action round trip passed; maximum round-trip absolute error was `5.96e-08`.

Step 5 published predictions and a report under
`/root/autodl-tmp/EmbodiedAI/runs/stage7-step5/smolvla-base-offline-v1`. It evaluated the
first/middle/last frame of each validation episode (15 anchors). All outputs were finite and an
identical rerun had zero maximum absolute difference. Base normalized-action MAE/RMSE were
`0.498620`/`0.678683`; gripper-sign agreement was `0.3333`, bound-violation rate was `0.07619`, and
mean warmed inference latency was about `0.142 s`. These are offline diagnostics, not task success.

Step 6A published the adapter/checkpoint under
`/root/autodl-tmp/EmbodiedAI/checkpoints/stage7-step6a/smolvla-lora-r2-steps50-v1` and reports under
`/root/autodl-tmp/EmbodiedAI/runs/stage7-step6a`. Rank-2 LoRA made 92,832 of 450,139,008 parameters
trainable (74 tensors). The fixed-input three-step micro check decreased loss from `14.849166` to
`14.727476`. The 50-step bounded job changed all trainable tensors and reduced held-out
flow-matching loss from `5.807343` to `3.894992`; peak allocated/reserved CUDA memory was about
1.26/1.33 GiB. A clean process reloaded the adapter and its processors and reproduced the first
probe exactly. Adapter MAE/RMSE were `0.505373`/`0.694298`, slightly worse than the unchanged base
baseline, so the checkpoint proves training/save/reload mechanics only and is not promoted as an
improved policy.

## Handoff

Steps 4-6B have passed their approved execution. Stage 8 reused the selected Step 6B adapter through
the loopback Robot Client + Policy Server boundary with receding-horizon `execute_horizon = 5`.
Offline improvement did not yield closed-loop success: all Step 6B rollouts were rejected by the
unchanged hard-action gate. Any action-distribution or safety-policy change requires a new reviewed
experiment rather than mutation of these frozen reports.

No step implicitly authorizes package installation, lock modification, model download, dataset
mutation, or Isaac execution.
