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
| 6 | Isaac Demonstration Pipeline | In progress; single-episode pipeline completed 2026-08-26, multi-episode generation pending |
| 7 | LeRobot + VLA Training Pipeline | In progress; step 1 and bounded step 2 implementation completed 2026-08-27 |
| 8 | Closed-loop Policy Evaluation | Not approved |
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
7. Generate multiple expert episodes. **Pending.** Review a collection matrix covering episode
   count, unique episode IDs, seeds, instruction variants, and any controlled reset/task
   variation before generation. Publish one immutable directory per episode, validate every
   manifest and payload, and produce a dataset-level collection summary suitable for Stage 7
   ingestion. Raw episodes remain under the external dataset root; preview videos remain derived
   artifacts.

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
   **Partially completed.** The converter now validates source manifests/payloads, preserves
   source/expert provenance in a conversion sidecar, keeps one-to-one episode boundaries, and
   atomically publishes a finalized/reopened local dataset. A three-frame image-backed CPU round
   trip passed in no-GPU mode. The real 108-frame smoke was terminated by the current 2 GiB memory
   limit before publication; its private partial directory was removed. Full real/multi-episode
   conversion therefore remains pending a higher-memory allocation.
3. Validate episode counts, frames, timestamps, image/action shapes, task/instruction mapping,
   normalization inputs, and deterministic reload behavior.
4. Implement and validate SmolVLA preprocessing against the converted dataset.
5. Run reviewed SmolVLA base inference on project observations and record finite outputs,
   latency, memory use, and action compatibility.
6. Run bounded LoRA / PEFT fine-tuning, save adapter/checkpoint provenance, and verify reload plus
   offline evaluation before proposing a longer training run.

Exit gate: a reproducible LeRobot dataset and reviewed SmolVLA adapter/checkpoint pass offline
validation in the isolated VLA environment.

## Stage 8 — Closed-loop Policy Evaluation

1. Implement a SmolVLA adapter that converts contract observations/instructions into policy
   inputs and converts model outputs back to the canonical action schema.
2. Implement the Isaac inference loop across the process/environment boundary without
   cross-installing Isaac and LeRobot stacks.
3. Record task success, failure, truncation, completion time, action validity, inference latency,
   and rollout-level diagnostics.
4. Compare the trained policy with reviewed baselines, initially the deterministic state-machine
   expert and the unfine-tuned SmolVLA base policy.

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
