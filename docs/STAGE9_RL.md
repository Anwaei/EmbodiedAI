# Stage 9 RL Policy Refinement

Status: **Steps 9.1 and 9.2 implemented and GPU-validated on 2026-08-31; training has not started**

Stage 9 adds two reinforcement-learning modes for Franka pick-and-place:

1. a privileged, state-based standalone PPO baseline; and
2. a bounded 6D PPO residual around a frozen SmolVLA nominal action.

The first mode is an intentionally privileged simulation baseline. The second preserves the
multimodal VLA as the nominal policy and learns only a small end-effector correction. SAC and
full-model VLA reinforcement learning are outside the first implementation.

## 1. Feasibility assessment

The proposal is feasible with the current project boundaries, with two different levels of risk.

- **Standalone PPO is directly feasible.** The installed Isaac environment already locks
  `isaaclab-rl[rsl-rl]` and passed the Stage 4 RSL-RL smoke. The existing task already uses a
  `ManagerBasedRLEnv`, normalized 7D end-effector-delta action, vectorized task evaluation, and
  parameterized cube/goal geometry. Stage 9 must replace the placeholder reward and demonstration
  observation group with RL-specific managers; it does not require a new Python environment.
- **Bounded residual PPO is architecturally feasible but has two hard prerequisites.** Stage 8's
  synchronous one-observation HTTP path is not a sufficient throughput path for vectorized
  on-policy training, and the formal Step 6B FT-VLA failed closed because its raw chunks exceeded
  the reviewed action limit. Stage 9 therefore needs a batched/cached nominal-action provider and
  a reviewed nominal-action quality gate before formal residual training.
- A six-dimensional residual deliberately cannot change the nominal gripper command. If FT-VLA's
  gripper timing is not usable after reviewed postprocessing, residual PPO cannot repair it. That
  is a promotion gate, not an implementation bug in the residual controller.
- The 32 GiB RTX 5090 has already run Isaac and SmolVLA together, so a single-environment
  integration smoke is realistic. The maximum vectorized environment count and VLA batch size
  must be measured rather than assumed because cameras, Isaac rendering, SmolVLA, and PPO share
  the GPU.

The Stage 8 fail-closed results remain immutable historical evidence. Stage 9 may introduce a new,
explicitly versioned nominal-action composition profile, but it must re-evaluate the non-residual
VLA policies with the same profile for a fair Stage 9 comparison.

### Implemented foundation — 2026-08-31

Steps 9.1 and 9.2 added the standalone PPO contracts/configuration and shared Isaac RL task layer:

- `franka-pick-place-ppo-state-v1`: fixed 52D state-only actor/critic profile;
- `franka-pick-place-ppo-ee-delta-v1`: unchanged normalized 7D action profile;
- `franka-pick-place-staged-reward-v1`: reach, grasp, lift, place, success, action-magnitude,
  action-rate, and gripper-toggle terms;
- `EmbodiedAI-Franka-PickPlace-State-PPO-v0`: separate Gym task with no RGB camera;
- exact `rsl-rl-lib 3.1.2` / `isaaclab-rl 0.4.7` backend identity checked against the lock;
- independently sampled per-environment cube/goal resets within committed safe ranges;
- monotonic reach → grasp → lift → place → release phase state with one transition per step;
- separate cube-filtered left/right finger contact sensors; and
- public success/failure, invalid-state, and timeout termination diagnostics.

A four-environment, two-step RTX 5090 smoke passed with observation shape `(4, 52)`, action shape
`(4, 7)`, finite rewards, non-terminal reviewed resets, independent cube/goal samples, and no
camera or cross-runtime import. This is task-layer validation only: no PPO optimization or
checkpoint was produced.

## 2. Runtime and dependency boundaries

### 2.1 Standalone PPO

```text
Isaac Python 3.11 process
  vectorized Isaac Lab environment
    -> state-only RL observation
  RSL-RL PPO actor/critic
    -> normalized 7D action
  task reward + termination + checkpoint/metrics
```

The standalone trainer lives entirely in the existing Isaac environment. RGB rendering and
language tokenization are disabled for training. LeRobot and SmolVLA must not be imported.

### 2.2 Residual PPO

```text
VLA Python 3.12 Policy Server
  frozen base or FT SmolVLA
  RGB + joint state + instruction
    -> batched canonical nominal action chunks

Isaac Python 3.11 process
  vectorized Isaac Lab environment
  per-environment nominal action queues
  state-only RSL-RL PPO residual actor
    -> bounded 6D residual
  action composer
    -> final normalized 7D action
```

The process boundary remains the Stage 8 Robot Client + Policy Server boundary. Stage 9 extends it
with a versioned batch request and independent per-environment queues; it does not import LeRobot
or SmolVLA into Isaac. The VLA is frozen during residual PPO and receives no RL gradients.

## 3. Common task semantics

Both modes reuse the same 20 Hz public action contract, parameterized cube/goal resets, success
definition, failure workspace, maximum episode length, and held-out evaluation scenarios. Training
and evaluation scenario manifests must be disjoint and must record their cube position, goal,
instruction, seed, and task revision.

The RL task adds a deterministic phase estimator with hysteresis:

```text
reach -> grasp -> lift -> place -> release/success
```

The phase estimator is task state used for reward gating and diagnostics. It is not inferred from
language, and it must not oscillate backward because of one noisy contact frame. Public success
remains the existing cube-at-goal, low-speed, gripper-open condition rather than a phase flag.

## 4. Mode A: state-based standalone PPO

### 4.1 Observation

The proposed `franka-pick-place-ppo-state-v1` actor observation is a fixed-order, normalized 52D
vector:

| Component | Dimension | Notes |
|---|---:|---|
| robot joint position | 9 | seven arm joints plus two fingers |
| robot joint velocity | 9 | matching joint order |
| tool-center position | 3 | environment/robot-base frame |
| tool-center quaternion | 4 | normalized and sign-canonicalized |
| cube position | 3 | privileged simulator state |
| cube linear velocity | 3 | privileged simulator state |
| goal position | 3 | per-environment task parameter |
| tool-to-cube relative position | 3 | derived, same frame |
| cube-to-goal relative position | 3 | derived, same frame |
| previous executed action | 7 | makes action-rate cost observable |
| phase one-hot | 5 | reach, grasp, lift, place, release |

This is a privileged state-policy baseline, not a deployable visual policy. The actor and critic
use the same vector in the first version so that the baseline is easy to interpret. A later
asymmetric critic may add contact or rigid-body state only behind a new observation profile and an
ablation; it must not silently change the baseline.

Observation ordering, normalization ranges, quaternion convention, and task revision are written
to every checkpoint manifest. RGB and language are forbidden from this profile.

### 4.2 Action

PPO emits seven normalized values:

```text
[delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
```

The six arm values are clipped to `[-1, 1]` and use the existing Isaac adapter scales of 0.05 m
translation and 0.15 rad rotation per control step. The seventh PPO value is retained as a
continuous sampled action for optimization, then deterministically thresholded at zero into the
existing binary open/close command. Raw, clipped, thresholded, and executed actions are logged.

### 4.3 Reward

The intended term is `r_lift` (not `r_list`). The first reward contract is:

```text
r_t = w_r * r_reach
    + w_g * r_grasp
    + w_l * r_lift
    + w_p * r_place
    + w_s * 1_success
    - w_a * ||a_arm||^2
    - w_delta * ||a_t - a_(t-1)||^2
    - w_toggle * 1_gripper_toggle
```

- `r_reach`: dense tool-to-cube distance or distance-progress reward.
- `r_grasp`: gated bilateral-finger/contact-and-closure event with hysteresis.
- `r_lift`: normalized cube height above the table, active after grasp.
- `r_place`: cube-to-goal distance/progress, active only after a valid lift.
- `1_success`: large terminal bonus from the unchanged public evaluator.
- action terms penalize magnitude, rapid change, and unnecessary gripper toggling.

Grasp, lift, and place rewards are phase-gated to prevent collecting place reward without lifting
or repeatedly cycling between stages. Before training, each term receives an isolated unit test
and an episode-contribution audit. Numerical weights remain config values and are approved only
after no dense term can dominate the success bonus through episode length alone.

### 4.4 Termination and reset

An episode terminates on public task success, cube/workspace failure, invalid/non-finite simulator
state, or timeout. Training resets sample only from a committed training distribution derived from
the Stage 6 geometry bounds. Evaluation uses frozen held-out scenario/seed manifests. Curriculum
may move from conservative fixed geometry to the full training distribution, but it may not alter
the held-out matrix.

## 5. Mode B: bounded residual PPO

### 5.1 Residual observation

The strict first residual profile follows the current VLA input boundary after removing RGB and
language. The learned residual receives:

- the current 9D robot joint position;
- the current, clipped 7D nominal VLA action;
- the previous executed 6D residual; and
- the normalized offset within the current receding-horizon chunk.

The nominal VLA still receives RGB, joint position, and instruction in its separate process. Thus
multimodal task interpretation stays in the VLA, while PPO learns a local correction from compact
controller state. Cube/goal privileged state is not silently added to this strict residual
profile. Joint velocity or privileged task state can be evaluated later only as explicitly named
ablations, because each changes the meaning of `FT-VLA + PPO`.

### 5.2 Action composition

Let `a_vla_raw` be the current 7D action from the frozen VLA chunk and `u_rl` be the residual
actor's 6D output. The versioned composer applies:

```text
a_nom       = clip(a_vla_raw, -1, 1)
u_bounded   = clip(u_rl, -1, 1)
delta_arm   = alpha * u_bounded
a_final_arm = clip(a_nom[0:6] + delta_arm, -1, 1)
a_final_grip = binary_threshold(a_nom[6])
a_final      = concat(a_final_arm, a_final_grip)
```

`alpha` is configured separately for translation and rotation components and begins small. It is
expressed in normalized action units; the existing IK adapter performs the later conversion to
metres and radians. The residual never changes the gripper in v1.

Finite checks occur before clipping. The composer records the raw VLA action, bounded nominal,
raw residual, bounded/scaled residual, final action, and every clipping mask. A configurable
emergency raw-action rejection limit remains separate from normal contract clipping. Stage 9 must
not hide the Stage 8 FT-VLA saturation problem: raw bound-violation and saturation rates remain
promotion metrics even when a finite action is safely clipped for an engineering smoke.

### 5.3 Residual reward

Residual PPO reuses the standalone task reward and adds correction regularization:

```text
r_residual = r_task
           - w_res * ||delta_arm||^2
           - w_res_delta * ||delta_arm_t - delta_arm_(t-1)||^2
```

The penalty applies to the scaled correction actually offered to the composer, not merely the
unscaled network output. This makes the learned policy prefer the nominal VLA whenever no
correction is useful and discourages high-frequency compensation.

### 5.4 Nominal-policy and throughput gates

Formal residual training starts only when all of the following pass:

1. model, adapter, processor, statistics, and action-composer identities are frozen;
2. the nominal path is finite and its raw violation/saturation distribution has been reviewed;
3. nominal gripper behavior demonstrates that the unmodifiable gripper channel is usable;
4. batched requests, independent action queues, reset isolation, and deterministic seeds pass;
5. measured VLA batch throughput, camera cost, VRAM, and simulator throughput support the chosen
   number of environments without stale actions; and
6. fake-nominal, single-environment real-VLA, and small-vectorized pilots all publish complete
   artifacts.

If these gates fail, Stage 9 may still complete the reusable residual framework and report the
blocked policy experiment, but it must not claim an effective FT-VLA+PPO result.

## 6. Stage 9 implementation steps

### Step 9.1 — Freeze RL contracts, configurations, and backend

**Completed and validated on 2026-08-31.**

Define versioned standalone/residual observation profiles, action-composition metadata, reward
configuration, scenario split, checkpoint identity, and run manifest. Confirm the already locked
RSL-RL backend without changing packages.

Exit gate: configurations reject ambiguous dimensions, frames, scales, policy identities, and
output paths; dependency-light tests pass without launching Isaac.

### Step 9.2 — Build the shared Isaac RL task layer

**Completed and GPU-validated on 2026-08-31.**

Add vectorized state extraction, phase estimation, reward terms, reset sampling, termination
diagnostics, and RL-specific task configurations. Keep the existing demonstration and Stage 8
task behavior stable.

Exit gate: every reward/phase/termination term passes deterministic tensor tests and one Isaac
environment reproduces the public success/failure semantics.

### Step 9.3 — Run the standalone PPO environment and integration smokes

**Completed and GPU-validated on 2026-09-01.**

Connect the RL task to RSL-RL, first with one environment and zero/random actions, then with a
small vectorized headless run. Validate observation normalization, 7D action adaptation, episode
reset isolation, reward accounting, logging, and checkpoint creation/reload.

The accepted smoke used four vectorized environments and proved optimization, periodic checkpoint
creation, manifest finalization, checkpoint reload, and resume advancement. The resumed run loaded
`model_0.pt` and produced `model_1.pt`. Both runs stayed below the external Stage 9 run root and
imported no RGB, language, LeRobot, or VLA code.

Exit gate: a short PPO optimization job advances, resets, saves, reloads, and resumes without RGB,
language, LeRobot, or partial artifacts. Task success is not required for this engineering smoke.

### Step 9.4 — Train the standalone PPO baseline

**Initial baseline completed and GPU-evaluated on 2026-09-01.**

Run a bounded tuning progression: conservative fixed geometry, the training geometry distribution,
then multiple fixed seeds. Save periodic and best checkpoints using validation success first and
return second. Track reward terms, success, failure, episode length, action saturation, entropy,
KL, throughput, VRAM, and wall time.

Exit gate: the selected checkpoint is reproducible, beats zero/random controls on held-out
scenarios, and its result is reported even if it does not beat the scripted expert.

The first bounded training progression produced the following evidence:

- the v1 fixed-geometry pilot ran 200 updates over 128 environments; its first distribution
  continuation exposed reward hacking because grasp/lift return rose while task success fell to
  zero;
- reward profile v2 gates grasp and lift shaping to their active phases, adds a public workspace
  failure penalty, and raises place/success emphasis;
- the v2 fixed run resumed from the v1 `model_125.pt`, ran 300 updates over 128 environments
  (921,600 simulator steps), saved 13 checkpoints, and completed in about 10 minutes 15 seconds;
- a v2 distribution continuation ran 100 further updates (307,200 simulator steps), but none of
  its five checkpoints succeeded on the frozen scenarios;
- checkpoint selection used success count first and mean return second. The selected fixed
  `model_175.pt` achieved 1/10 success with mean return 5.687, versus 0/10 and returns 0.013 and
  -0.147 for zero and random controls. An independent rerun reproduced the same success scenario
  and aggregate metrics;
- the result is a weak privileged-state baseline, not a robust solved policy: nine scenarios
  timed out, raw action saturation was 48.1%, and the geometry-distribution continuation scored
  0/10. Later high-return checkpoints were rejected because they scored 0/10.

The engineering exit gate is therefore met narrowly: a provenance-complete checkpoint is
reproducible and beats both controls on the frozen suite. The robustness objective is not met and
must not be inferred from this result. Future tuning should use transition/progress shaping or a
curriculum before spending compute on longer runs.

Accepted artifacts are external to Git:

```text
$EMBODIEDAI_RUNS/stage9/stage9-franka-pick-place-standalone-ppo-v2/
  fixed-reward-v2-seed-20260901-from-v1-fixed125/
  distribution-reward-v2-seed-20260901-from-v2-fixed175/
  evaluations/fixed-v2-checkpoint-sweep-20260901/evaluation.json
  evaluations/distribution-v2-checkpoint-sweep-20260901/evaluation.json
  evaluations/selected-fixed175-repro-20260901/evaluation.json
```

### Step 9.5 — Implement and verify the residual contracts and composer

Add the nominal-action provider interface, residual observation profile, bounded 6D composer,
gripper pass-through, correction reward terms, and full action diagnostics. Use deterministic fake
nominal providers for unit and Isaac integration tests.

Exit gate: algebra, per-dimension scales, double clipping, gripper immutability, reset behavior,
and non-finite/emergency failures are exhaustively tested.

### Step 9.6 — Add the batched SmolVLA nominal-action path

Extend the Stage 8 process boundary with versioned batched requests/responses and independent
per-environment receding-horizon queues. Measure real base and FT-VLA latency, throughput, VRAM,
queue age, saturation, and reset isolation. Preserve a one-environment compatibility path.

Exit gate: a real frozen VLA drives a bounded single-environment residual smoke, and the reviewed
batch size completes a vectorized pilot without stale or cross-episode actions.

### Step 9.7 — Run the residual PPO integration smoke

Train the 6D residual first against a deterministic safe nominal provider, then run a bounded
real FT-VLA pilot. Verify that the VLA is frozen, PPO sees no RGB/language tensors, and gradients
do not cross the process boundary.

Exit gate: optimize/save/reload/resume and a complete real two-process rollout work end to end;
task success is not required for the engineering smoke.

### Step 9.8 — Train FT-VLA + PPO residual policy

Proceed only after the nominal-policy and throughput gates. Use the same training geometry split
and comparable seeds as standalone PPO. Tune residual scale and penalty before PPO hyperparameters;
retain checkpoints by validation success and correction/safety diagnostics.

Exit gate: the selected residual checkpoint and its frozen nominal VLA identity are inseparable in
the manifest and reproduce the reported validation result.

### Step 9.9 — Run the five-policy comparison

Evaluate on identical frozen scenarios and seeds:

| Policy | Input distinction | Role |
|---|---|---|
| state-machine expert | privileged simulator state | scripted upper reference |
| base SmolVLA | RGB + state + language | unfine-tuned imitation baseline |
| standalone PPO | privileged low-dimensional state | state-based RL baseline |
| FT-VLA | RGB + state + language | fine-tuned imitation baseline |
| FT-VLA + PPO | frozen FT-VLA plus compact residual state | residual RL candidate |

Use one Stage 9 execution-safety/composition profile for both VLA-only and residual comparisons,
while retaining the original Stage 8 fail-closed results. Report paired success, final/minimum goal
error, completion time, failure reason, reward terms, raw/final saturation, residual magnitude,
action smoothness, inference latency, simulator throughput, and training sample/compute cost.
The expert and privileged PPO are references, not like-for-like deployable visual policies.

Exit gate: the five-policy report is paired, provenance-complete, and does not conflate framework
execution with policy quality.

### Step 9.10 — Publish reproducibility and handoff artifacts

Publish exact configurations, manifests, checkpoint hashes, dependency lock/Git/hardware identity,
commands, learning curves, resource measurements, failure examples, and known limitations under
the external run root. No checkpoint, replay, video, or large metric tensor enters Git.

Exit gate: standalone PPO and the residual framework can be rebuilt and run end to end from the
reviewed repository. An effectiveness claim for residual PPO additionally requires it to improve
the reviewed primary metric without worse safety or materially worse correction/smoothness
metrics.

## 7. Artifact and comparison policy

```text
$EMBODIEDAI_RUNS/stage9/<run-id>/
  configs/
  manifests/
  checkpoints/                 # or checkpoint references/hashes
  metrics/
  rollouts/
  reports/
  videos/                      # optional derived previews
```

Checkpoints, optimizer state, normalizers, VLA queues, and temporary tensors remain under the
external data-disk roots established in `docs/ENVIRONMENT.md`. A Stage 9 manifest must bind the
task revision, observation profile, action composer, reward profile, scenario split, seeds,
RSL-RL configuration, nominal VLA identity where applicable, Git revision, and environment lock
hash.

## 8. Explicit non-goals for the first implementation

- no SAC baseline in v1;
- no RGB or language in the PPO actor;
- no RL update to SmolVLA weights;
- no residual gripper action;
- no ROS 2 transport;
- no physical-robot or Sim2Real claim; and
- no replacement or reinterpretation of Stage 8 artifacts.
