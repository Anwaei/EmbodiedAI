# Roadmap

## Stage status

| Stage | Scope | Status |
|---|---|---|
| 0 | Review environment and architecture plan | Approved |
| 1 | Hardware audit | Completed; GPU-mode allocation revalidated 2026-08-20 |
| 2 | Repository skeleton and storage policy | Completed 2026-08-20 |
| 3 | uv tooling, independent definitions, and lock review | Completed 2026-08-20; see `DEPENDENCY_LOCKS.md` |
| 4 | Install and validate Isaac environment | Completed 2026-08-20; see `ENVIRONMENT.md` |
| 5 | Install and validate VLA environment | Completed 2026-08-21; see `ENVIRONMENT.md` |
| 6 | Isaac Demonstration Pipeline | Completed; 20-episode baseline and 100-episode expanded corpus validated |
| 7 | LeRobot + VLA Training Pipeline | In progress; steps 1-5 and 6A complete; expanded 6B dataset converted/validated, split review pending |
| 8 | Closed-loop Policy Evaluation | Approved 2026-08-31; Robot Client + Policy Server implementation pending |
| 9 | RL Policy Refinement | Not approved |
| 10 | ROS 2 deployment boundary | Not approved |
| 11 | Reproducibility and robustness gate | Not approved |

## MVP milestones

1. Reproducible Isaac and VLA environments from reviewed locks.
2. Franka pick-and-place scene and deterministic expert demonstration generation.
3. Versioned dataset export and validation in LeRobot format.
4. SmolVLA inference, then bounded LoRA/PEFT fine-tuning.
5. Closed-loop policy evaluation in simulation.
6. Optional PPO/SAC residual action refinement after the imitation baseline works.
7. ROS 2 Humble deployment interface through the Isaac Sim bridge.
8. Domain-randomization robustness report covering visual, camera, object, physics, and
   action/sensor perturbations.

## Stage 6 — Isaac Demonstration Pipeline

Scope boundary: Stage 6 owns Isaac Sim / Isaac Lab task execution, expert action generation,
contract-compliant raw episode publication, and derived preview artifacts. It does not import
LeRobot or train policies.

1. Define dependency-light observation, action, and episode contracts. **Completed.**
2. Create the Franka, table, cube, goal, and RGB-camera Isaac Lab task skeleton. **Completed.**
3. Keep LeRobot and VLA imports out of the Isaac runtime. **Completed.**
4. Add deterministic reset and vectorized current-state evaluation interfaces. **Completed.**
5. Add immutable NPY episode recording and the bounded structural rollout path. **Completed.**
6. Add expert demonstration generation. **Completed.** The initial implementation uses a
   deterministic state-machine expert for
   `franka-pick-place` with the canonical instruction `Pick up the cube and place it in the
   goal.` The collector writes one episode per environment and records explicit task, instruction,
   and expert provenance. The interfaces must allow instruction paraphrases, additional task
   definitions, and state-machine, learned-policy, or teleoperation experts without changing
   the recorder's observation/action boundary.
7. Generate multiple expert episodes. **Completed.** A versioned collection plan fixes the task
   and state-machine expert while varying five instruction paraphrases, five cube reset positions,
   and four goal positions across 20 unique seeds/episode IDs. The batch launcher uses one fresh
   Isaac process per episode, validates every immutable manifest/payload, and atomically maintains
   a collection summary. All 20 episodes succeeded on 2026-08-27 with 2,138 total frames and no
   partial directories. Raw episodes remain under the external dataset root; preview videos remain
   derived artifacts.
8. Expand the expert corpus for Stage 7 Step 6B. **Completed 2026-08-30.** A second versioned plan
   covers the complete product of 10 conservative cube resets and 10 goal positions while balancing
   five instruction paraphrases across both spatial axes. All 100 episodes succeeded, producing
   10,707 validated frames with 100 unique manifest hashes and no partial directories. The immutable
   Contract corpus was then converted and independently validated by the repeated Stage 7
   steps 1-3 cycle without changing the immutable raw source.

Exit gate: a reviewed, validated multi-episode Isaac demonstration corpus is ready for conversion,
without LeRobot or VLA dependencies in the Isaac runtime.

## Stage 7 — LeRobot + VLA Training Pipeline

Stage 5 proved that the locked environment, synthetic LeRobot round trip, offline base checkpoint,
and bounded PEFT mechanics work; those environment smokes do not replace this project-data
pipeline.

1. Define and implement the versioned Contract → LeRobot feature/task mapping. **Completed.** The
   `franka-pick-place-smolvla-v1` profile maps joint positions, front RGB, normalized 7D actions,
   and exact episode instructions while explicitly excluding joint velocity and privileged cube
   position from the initial policy input.
2. Build a `LeRobotDataset` from validated Stage 6 episodes without mutating the raw source data.
   **Completed.** The converter validates source manifests/payloads, preserves
   source/expert provenance in a conversion sidecar, keeps one-to-one episode boundaries, and
   atomically publishes a finalized/reopened local dataset. The 20-episode baseline remains at
   `stage7-franka-pick-place-batch-v1`. The expanded corpus is separately published as a
   100-episode, 10,707-frame, five-instruction video-backed dataset at
   `/root/autodl-tmp/EmbodiedAI/datasets/stage7-franka-pick-place-batch-v2-100`; its repo ID is
   `embodiedai/franka-pick-place-stage7-batch-v2-100`.
3. Validate episode counts, frames, timestamps, image/action shapes, task/instruction mapping,
   normalization inputs, and deterministic reload behavior. **Completed.** The independent
   validator matched every state/action/table value and source provenance record, recomputed the
   stored normalization statistics, decoded both boundaries of every episode through two fresh
   LeRobot instances, and atomically wrote a passed report to
   `/root/autodl-tmp/EmbodiedAI/runs/stage7-validation/stage7-franka-pick-place-batch-v2-100.json`.
   The expanded dataset passed for 100 episodes, 10,707 frames, five tasks, and 200 decoded image
   samples. Its 224 x 224 AV1 stream independently reports 10,707 frames at 20 Hz over 535.35
   seconds; no state/action dimension is constant.
4. Develop and validate the reusable SmolVLA preprocessing/postprocessing boundary. **Completed.**
   The project-owned processor explicitly replaces the base checkpoint's 6D state, 6D action, and three-camera
   processor metadata with the project dataset's 9D joint state, canonical 7D action, one front
   RGB camera, exact instruction text, and explicit project statistics. Cover action-horizon and
   padding-mask assembly, batching, tokenization, image handling, normalization, device placement,
   action unnormalization, and contract validation without importing Isaac. Train-only statistics
   and the reviewed 15/5 whole-episode split are serialized and hash-checked. The detailed design
   and exit gate are in `docs/SMOLVLA_PIPELINE.md`.
5. Develop the SmolVLA base offline-inference chain. **Completed.** Load the already
   pinned base model and tokenizer/config locally in the VLA environment, run deterministic
   project-data compatibility probes and a causal dataset baseline, and record canonical 7D
   predictions, finite/bounds checks, action metrics, latency, memory, and complete provenance.
   The accepted 15-anchor held-out baseline is finite and exactly repeatable; its normalized-action
   MAE/RMSE are 0.4986/0.6787. This step does not run Isaac and cannot claim closed-loop task success.
6. Develop fine-tuning in two reviewed substeps. **Step 6A completed; Step 6B not started.** Step 6A uses a
   deterministic episode-level split of the existing 20 episodes and a bounded LoRA/PEFT job to
   prove train/validate/save/reload mechanics. The accepted rank-2 run trains 92,832 parameters for
   50 optimizer steps, lowers held-out flow-matching loss from 5.8073 to 3.8950, saves an adapter,
   and reloads it in a clean process. Its offline MAE of 0.5054 is slightly worse than the base, so
   it establishes technical feasibility only. Step 6B begins only after a larger corpus has passed
   steps 1-3 again, then performs formal PEFT development with frozen train/validation/test splits,
   training-only statistics, checkpoint selection, and held-out offline evaluation. Full-parameter
   fine-tuning remains separately gated. The larger corpus has now passed steps 1-3; Step 6B next
   requires review of its frozen episode-level split, training-only statistics, and formal run
   configuration before training begins.

Exit gate: a reproducible LeRobot dataset and reviewed SmolVLA adapter/checkpoint pass offline
validation in the isolated VLA environment.

## Stage 8 — Closed-loop Policy Evaluation

Stage 8 was approved on 2026-08-31 for the existing small-corpus base/Step 6A artifacts. It uses a
loopback HTTP/JSON Robot Client + Policy Server boundary, not ROS 2. The accepted action scheduler
uses a 50-step SmolVLA prediction with receding-horizon `execute_horizon = 5`.

1. Define versioned RPC, policy identity, scenario, and rollout contracts.
2. Implement a VLA-only Policy Server that reuses the Stage 7 processor and loads one pinned base
   or PEFT adapter per process.
3. Implement a one-environment Isaac Robot Client, five-action receding-horizon scheduler, online
   action safety, timeout handling, and public task evaluation.
4. Validate protocol, malformed responses, timeout, reset/queue semantics, and dependency
   boundaries with a fake server before GPU integration.
5. Run a single real two-process base-policy closed-loop smoke and atomically record its artifacts.
6. Run the five held-out small-corpus scenarios with three fixed seeds per learned policy and
   record task/action/latency/rollout metrics.
7. Compare the scripted state-machine reference, unfine-tuned SmolVLA base, and Step 6A adapter on
   the same 45-rollout matrix.
8. Publish a reproducible JSON/human-readable report and handoff. Step 6B adapters can later be
   added without changing the protocol or Robot Client.

Detailed design and gates are in `docs/STAGE8_EVALUATION.md`.

Exit gate: bounded closed-loop Isaac rollouts produce reproducible metrics and a baseline
comparison report.

## Stage 9 — RL Policy Refinement

1. Define reviewed PPO / SAC training interfaces, observations, actions, rewards, termination
   semantics, and reproducible simulation evaluation.
2. Establish standalone PPO/SAC baselines before combining learned controllers.
3. Implement an optional residual policy of the form
   `bounded_action = clamp(vla_action + rl_residual)` with explicit scale and safety limits.
4. Compare residual refinement against the Stage 8 VLA and scripted baselines; retain it only if
   it improves reviewed metrics without violating the action contract.

Exit gate: PPO/SAC and residual-policy results are reproducible, contract-compatible, and compared
against the closed-loop imitation baselines.

## Stage 10 — ROS 2 Deployment Boundary

1. Install ROS 2 Humble using the Ubuntu 22.04 binary path and build the overlay workspace.
2. Start with standard messages and separate external policy/control nodes.
3. Validate the Isaac bridge, DDS discovery, timestamps, QoS, action rate, and
   emergency-stop/timeout behavior.
4. Add custom interfaces only if standard messages are insufficient, with explicit Python 3.10
   and Python 3.11 build validation.

Exit gate: simulated sensor/state/language input reaches the policy node and safe actions reach
the simulated controller.

## Stage 11 — Reproducibility and Robustness Gate

1. Recreate each environment from locks in a clean location.
2. Run unit, integration, GPU smoke, dataset, closed-loop policy, RL, ROS, and
   domain-randomization tests.
3. Capture versions, hardware, timings, VRAM/RAM peaks, perturbation results, and known
   limitations.

Exit gate: a documented clean-room rebuild and MVP robustness report succeed.

## Deferred work

- Multi-step manipulation follows the single-step baseline.
- OpenVLA/OpenVLA-OFT experiments follow SmolVLA.
- World models and learned dynamics do not block the MVP.
