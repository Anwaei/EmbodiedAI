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
| 7 | LeRobot + VLA Training Pipeline | Completed through Step 6B on 2026-08-31 |
| 8 | Closed-loop Policy Evaluation | Completed for small and expanded-corpus matrices on 2026-08-31 |
| 9 | RL Policy Refinement | Steps 9.1-9.2 completed 2026-08-31; training not started |
| 10 | ROS 2 deployment boundary | Not approved |
| 11 | Reproducibility and robustness gate | Not approved |

## MVP milestones

1. Reproducible Isaac and VLA environments from reviewed locks.
2. Franka pick-and-place scene and deterministic expert demonstration generation.
3. Versioned dataset export and validation in LeRobot format.
4. SmolVLA inference, then bounded LoRA/PEFT fine-tuning.
5. Closed-loop policy evaluation in simulation.
6. State-based PPO baseline followed by optional bounded PPO residual action refinement.
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
6. Develop fine-tuning in two reviewed substeps. **Completed.** Step 6A uses a
   deterministic episode-level split of the existing 20 episodes and a bounded LoRA/PEFT job to
   prove train/validate/save/reload mechanics. The accepted rank-2 run trains 92,832 parameters for
   50 optimizer steps, lowers held-out flow-matching loss from 5.8073 to 3.8950, saves an adapter,
   and reloads it in a clean process. Its offline MAE of 0.5054 is slightly worse than the base, so
   it establishes technical feasibility only. Step 6B freezes a spatially and linguistically
   balanced 80/10/10 whole-episode split of the 100-episode corpus, recomputes statistics from the
   80 training episodes, and trains a rank-8 LoRA adapter for exactly five epochs (2,680 optimizer
   steps). The selected epoch-5 checkpoint reduced fixed validation flow-matching loss from 3.6908
   to 0.3259. On the untouched test anchors, action MAE/RMSE improved from the base
   0.5212/0.7158 to 0.2974/0.5498, while the 8.10% bound-violation rate remained a mandatory
   closed-loop safety concern. Full-parameter fine-tuning remains separately gated.

Exit gate: a reproducible LeRobot dataset and reviewed SmolVLA adapter/checkpoint pass offline
validation in the isolated VLA environment.

## Stage 8 — Closed-loop Policy Evaluation

Stage 8 was completed on 2026-08-31 for both the small-corpus baseline and the formal Step 6B
adapter. It uses a
loopback HTTP/JSON Robot Client + Policy Server boundary, not ROS 2. The accepted action scheduler
uses a 50-step SmolVLA prediction with receding-horizon `execute_horizon = 5`.

1. Define versioned RPC, policy identity, scenario, and rollout contracts. **Completed.**
2. Implement a VLA-only Policy Server that reuses the Stage 7 processor and loads one pinned base
   or PEFT adapter per process. **Completed.**
3. Implement a one-environment Isaac Robot Client, five-action receding-horizon scheduler, online
   action safety, timeout handling, and public task evaluation. **Completed.**
4. Validate protocol, malformed responses, timeout, reset/queue semantics, and dependency
   boundaries with a fake server before GPU integration. **Completed.**
5. Run a single real two-process base-policy closed-loop smoke and atomically record its artifacts.
   **Completed.**
6. Run held-out scenarios with three fixed seeds per learned policy and record task/action/latency/
   rollout metrics. **Completed** for both five-scenario small-corpus and ten-scenario expanded-test
   suites.
7. Compare the scripted state-machine reference, unfine-tuned SmolVLA base, and PEFT adapter on
   identical matrices. **Completed.** The expanded matrix contains 30 rollouts per policy: expert
   30/30 success, base 0/30, and Step 6B 0/30. Both learned policies failed closed on the reviewed
   hard-action limit; no unsafe chunk was executed.
8. Publish a reproducible JSON/human-readable report and handoff. **Completed.** The expanded
   report is under `$EMBODIEDAI_RUNS/stage8/stage8-expanded-corpus-step6b-v1/reports`.

Detailed design and gates are in `docs/STAGE8_EVALUATION.md`.

Exit gate: bounded closed-loop Isaac rollouts produce reproducible metrics and a baseline
comparison report.

## Stage 9 — RL Policy Refinement

The first implementation has two modes: a privileged state-based standalone PPO baseline and a
bounded 6D PPO residual around a frozen FT-VLA nominal action. SAC remains deferred until these
two PPO paths are understood. RGB and language are excluded from both PPO actors; the residual
path still uses them inside the separate frozen VLA Policy Server.

1. **Step 9.1 — RL contracts/config/backend. Completed.** Freeze versioned observation, action composer,
   reward, scenario split, checkpoint, and run-manifest definitions; use the already locked RSL-RL
   PPO backend.
2. **Step 9.2 — Shared Isaac RL task layer. Completed.** Add vectorized low-dimensional state extraction,
   phase estimation, staged rewards, training resets, and compatible termination diagnostics
   without changing Stage 6 or Stage 8 behavior.
3. **Step 9.3 — Standalone PPO integration smoke.** Run one-environment and small-vectorized
   optimize/save/reload/resume paths with the normalized 7D end-effector/gripper action.
4. **Step 9.4 — Standalone PPO baseline training.** Train on a committed geometry distribution
   and select checkpoints on held-out success and return across fixed seeds.
5. **Step 9.5 — Residual contracts/composer.** Implement a nominal-policy provider and
   `clip(clip(a_vla) + alpha * clip(a_rl))`; PPO controls only six arm dimensions and the nominal
   VLA retains the gripper channel.
6. **Step 9.6 — Batched nominal VLA path.** Extend the Stage 8 process boundary with batched
   requests and independent per-environment receding-horizon queues; measure latency, VRAM,
   saturation, and queue age.
7. **Step 9.7 — Residual PPO integration smoke.** Prove the framework against a deterministic
   fake nominal policy, then a bounded real FT-VLA single-/small-vectorized pilot.
8. **Step 9.8 — FT-VLA + PPO training.** Proceed only after reviewed nominal-action, gripper,
   throughput, memory, and reset-isolation gates.
9. **Step 9.9 — Five-policy comparison.** Compare expert, base SmolVLA, standalone PPO, FT-VLA,
   and FT-VLA+PPO on identical frozen scenarios/seeds and one Stage 9 safety profile.
10. **Step 9.10 — Reproducibility handoff.** Publish configurations, checkpoint/model identities,
    curves, resource measurements, paired metrics, failure examples, and limitations.

Detailed observations, reward definitions, residual algebra, feasibility risks, and per-step exit
gates are in `docs/STAGE9_RL.md`.

Exit gate: standalone PPO and the bounded residual framework run reproducibly end to end and remain
contract-compatible. Residual effectiveness is a separate result and may be claimed only if the
paired evaluation improves the reviewed primary metric without worse safety.

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
