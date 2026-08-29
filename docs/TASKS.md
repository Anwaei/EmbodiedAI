# Isaac Lab Tasks

## Franka pick-and-place baseline

Stage 6 step 2 introduces the Gym task `EmbodiedAI-Franka-PickPlace-RGB-v0`.
It is a minimal `ManagerBasedRLEnv` configuration for validating the scene, deterministic
reset/evaluation behavior, and the shared observation/action boundary. The step 6 expert is a
separate controller and collector layered on this task; the task itself is not a training
environment.

### Scene

Each replicated environment contains:

- a Franka Panda using Isaac Lab's `FRANKA_PANDA_HIGH_PD_CFG`, selected because relative
  differential IK requires stable task-space tracking;
- a kinematic cuboid table whose top surface is at `z=0` in the environment frame;
- a dynamic 5 cm, 50 g cube initialized above the table;
- a fixed green 11 cm square goal marker on the table;
- a fixed 224 x 224 tiled RGB camera overlooking the workspace;
- a ground plane and dome light.

The table and cube use procedural primitives so the task does not add another remote asset
dependency. The Franka asset remains the pinned Isaac Lab asset already validated in Stage 4.
The tiled camera keeps the same batched sensor path when `num_envs` is increased later.

### Timing and action interface

Physics runs at 120 Hz with decimation 6, producing a 20 Hz control boundary. The seven
normalized action scalars are ordered as:

```text
[delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
```

The first six values drive Isaac Lab's damped-least-squares relative differential IK in the
Franka base frame. Normalized translation values are scaled by 0.05 m and normalized
axis-angle rotation values by 0.15 rad. A positive gripper value opens both fingers and a
negative value closes them. The task-specific `ActionSchema` records the normalized public
boundary; the physical scales are committed alongside it and used by the Isaac action config.

### Observation interface

The manager's policy group is deliberately not concatenated because it contains heterogeneous
state and image tensors. Isaac term names map to contract keys as follows:

| Contract key | Isaac policy term | Per-environment shape | Storage dtype/frame |
|---|---|---:|---|
| `robot.joint_position` | `joint_position` | `(9,)` | `float32`; arm radians, fingers metres |
| `robot.joint_velocity` | `joint_velocity` | `(9,)` | `float32`; arm rad/s, fingers m/s |
| `object.cube.position` | `cube_position` | `(3,)` | `float32`; environment-local frame, metres |
| `camera.front.rgb` | `camera_front_rgb` | `(3, 224, 224)` | `uint8`; camera optical frame |

The nine robot values have an explicit order: seven Panda arm joints followed by the two
finger joints. Isaac camera tensors are HWC RGB or RGBA depending on the renderer path. The
task adapter removes a possible alpha channel and converts to contiguous CHW `uint8`, so the
manager output already matches the shared observation contract.

The cube adapter subtracts each replicated environment origin from Isaac's world-frame root
position. This is required for the declared `env` frame and ensures that identical resets in
different vectorized environments produce identical contract values.

### Registration and imports

Importing `embodied_ai.sim.tasks` registers the task. All Isaac-specific imports remain under
`src/embodied_ai/sim`; the task-specific schema instance under
`src/embodied_ai/contracts/tasks` remains standard-library-only. A recursive AST test rejects
LeRobot, Transformers, Datasets, Accelerate, or PEFT imports from both the simulation package
and simulator scripts. A second check rejects those dependencies from `env/isaac/pyproject.toml`.

### Deterministic reset and evaluation

The task event restores the configured Franka and cube states without sampling random values.
Isaac Lab resets the Action Manager immediately after the event, preventing controller state
from the previous rollout from leaking across episodes. The cube always begins at
`(0.50, 0.00, 0.03)` metres in the environment frame. The goal centre is
`(0.65, -0.20, 0.03)` metres.

`evaluate_pick_place()` returns vectorized cube position, goal position, position error,
linear speed, gripper-open state, success, and failure values. Success requires centre
distance at most 5 cm, linear speed at most 0.10 m/s, and both finger joints to be at least
3 cm open. Requiring release prevents a rollout from terminating while the expert is still
holding a cube above the goal. Failure means the cube centre left the bounded workspace
`x=[0.10, 1.00]`, `y=[-0.50, 0.50]`, or fell below `z=-0.05` metres. Success and failure are
mutually exclusive and are wired into Isaac Lab termination terms; the ten-second time limit
remains a truncation. Reward remains an explicit zero placeholder because no training reward
was approved.

The following are owned by later stages:

- Stage 7 multi-episode LeRobot conversion/validation and VLA training; the versioned mapping,
  bounded converter, and one-episode real conversion are complete;
- Stage 8 learned-policy inference and closed-loop evaluation.

## Stage 6 step 6: expert demonstration generation

Status: **implemented and GPU-validated on 2026-08-25**.

### Initial scope

The first expert solves the existing `franka-pick-place` task from its deterministic reset
and emits the unchanged normalized seven-dimensional action contract. Its canonical instruction
is:

```text
Pick up the cube and place it in the goal.
```

The implementation targets one reproducible successful episode from one environment. It does
not add VLA inference, LeRobot imports, training, domain randomization, or a learned reward.
The existing fixed scene, current-state evaluator, and immutable NPY recorder remain the
authoritative task and storage boundaries.

### Expert interface

The Isaac-side `Expert` protocol separates action generation from collection. Its operations
are:

- `reset(env_ids)`: initialize per-environment expert state at an episode boundary;
- `act(env, observations)`: return a batched `float32` tensor conforming exactly
  to `FRANKA_PICK_PLACE_ACTION_SCHEMA`;
- `metadata`: return the dependency-light expert kind, identifier, revision, and configuration
  revision written to the episode manifest.

`task_context` identifies the task and selected instruction and exposes the task parameters
needed by the expert. The state-machine expert may read privileged simulator state such as the
end-effector pose, cube pose, goal pose, and gripper state. Those values guide demonstration
generation but do not automatically become VLA policy observations. The expert must use normal
actions and physics; it may not teleport the cube or write a successful terminal state directly.

The initial interface is batched even though the first collector uses one environment. Each
environment owns an independent state-machine phase, phase timer, and terminal status. This
preserves a later path to vectorized collection without changing the public action contract.

### Deterministic state machine

The controller uses a fixed downward grasp orientation, proportional clipped task-space
deltas, explicit tolerances, and a maximum duration for every phase. The implemented
phases are:

1. `move_above_cube`: open the gripper and move to a pre-grasp position above the cube.
2. `descend_to_grasp`: descend while retaining the fixed grasp orientation.
3. `close_gripper`: close the fingers and wait a fixed settling interval.
4. `lift_cube`: raise the tool and require the cube to rise above a configured lift threshold.
5. `move_above_goal`: translate the grasped cube to a pre-place position above the goal.
6. `descend_to_place`: lower the cube to the configured placement height.
7. `open_gripper`: release the cube and wait for its speed to settle.
8. `retreat`: move the tool clear of the object, then rely on the public task evaluator for the
   final success decision.
9. `done` or `failed`: stop producing rollout actions after success or a phase timeout/failure.

Transition thresholds, gains, hover heights, settle durations, and per-phase limits are in
committed TOML configuration rather than scattered constants. Actions are clipped to `[-1, 1]`;
the existing Isaac action adapter remains solely responsible for applying the 0.05 m translation
and 0.15 rad rotation scales. State-machine phase completion is internal control state and must
not replace the task's public success/failure evaluation.

### Collection lifecycle

The collector, rather than the expert, owns the environment and recorder lifecycle:

```text
select task + instruction + expert
  -> create/reset Isaac environment
  -> reset expert state
  -> record pre-action observation/action pairs at 20 Hz
  -> step Isaac Lab and evaluate the current state
  -> finalize success, failure, or truncated episode
  -> publish one immutable episode directory
```

Every collected training demonstration records the stable task ID, exact instruction text,
instruction variant ID/language, and expert provenance described in `CONTRACTS.md`. The raw
collection root may retain failed or truncated attempts for diagnosis, but the later VLA
conversion will select successful episodes by default and will not rewrite raw episodes.

The first collector preserves the currently accepted lifecycle of one explicit reset per
environment instance. A repeated-reset regression must be resolved before collecting multiple
sequential episodes in one process. Until then, repeated deterministic collection may use one
fresh process/environment lifecycle per episode. Future vectorized collection will maintain one
`NpyEpisodeRecorder` and one episode directory per environment; it will not add an environment
batch dimension to one episode.

### Extension model

The three extension axes are deliberately independent:

- **Different instruction, same task:** instruction variants such as `Move the cube to the green
  target.` retain `task=franka-pick-place`, the same reset/evaluator, and the same action schema.
  The selected exact text and variant ID are stored per episode.
- **Different task:** a command such as `Place the cube on the left side of the table.` receives a
  distinct stable task identifier and task definition because its target/evaluation semantics
  differ. A task definition binds a Gym environment/config, reset parameters, evaluator, allowed
  instruction variants, and compatible action/observation schemas.
- **Different expert:** `state_machine`, `rl_policy`, and `teleoperation` implementations share
  the same reset/act/metadata boundary. Learned experts additionally pin a checkpoint revision;
  teleoperation pins its device/control mapping and session revision. The collector does not
  branch on expert internals.

This means language paraphrases do not duplicate task logic, new tasks do not require recorder
changes, and new expert sources do not alter the episode observation/action representation.

### Repository additions

The implementation introduces these modules:

```text
src/embodied_ai/sim/experts/
├── base.py                         # expert protocol and step result
└── franka_pick_place_state_machine.py
src/embodied_ai/sim/collection/
└── expert_rollout.py               # rollout/recorder lifecycle
configs/sim/franka_pick_place/
├── state_machine_expert.toml       # gains, thresholds, heights, and phase limits
└── expert_collection_v1.toml       # reviewed 20-episode collection matrix
scripts/sim/
├── collect_franka_pick_place_expert.py
└── collect_franka_pick_place_expert_batch.py
```

All simulator control and collection code remains in the Isaac environment, while LeRobot
conversion remains a Stage 7 VLA-environment step.

## Stage 6 step 7: parameterized expert batch

Status: **implemented and GPU-validated on 2026-08-27**.

The first batch keeps `task=franka-pick-place`, the robot, schemas, camera, physics, and
state-machine configuration fixed. Its 20 rows cover five instruction paraphrases, five
conservative cube reset positions, four goal positions, and unique seeds/episode IDs. A shared
`FrankaPickPlaceEpisodeParameters` instance configures the cube default state, visible goal marker,
success termination, expert goal, and manifest fields before each environment is created.

One fresh Isaac process runs per row because repeated resets in one application remain outside the
accepted lifecycle. Each child publishes one immutable NPY episode; the parent validates it and
updates `collection_summary.json` atomically. Full child logs live under
`$EMBODIEDAI_RUNS/stage6-expert-batch/stage6-franka-pick-place-expert-v1`.

The accepted corpus is
`$EMBODIEDAI_DATASETS/stage6-expert-batch-v1-20260827`. All 20 episodes succeeded, totaling 2,138
frames with 97-114 steps per episode. The validator confirmed all payload hashes and parameter
metadata, all 20 manifest hashes are unique, and no private partial directory remains.

### Expert episode check

Run one deterministic expert rollout from an explicitly configured Isaac shell:

```bash
source scripts/bootstrap/project_env.sh
PYTHONPATH=src "$EMBODIEDAI_ENVS/isaac/bin/python" \
  scripts/sim/collect_franka_pick_place_expert.py \
  --headless --enable_cameras --device cuda:0 --steps 300 --seed 0 \
  --output_root "$EMBODIEDAI_DATASETS/stage6-expert" \
  --episode_id episode-stage6-expert-000001 \
  --video_path \
  "$EMBODIEDAI_ARTIFACTS/stage6/expert-videos/episode-stage6-expert-000001.mp4"
```

The script creates one environment, performs one explicit reset, collects synchronized
pre-action observation/action pairs, publishes one immutable episode, and reopens it through
the NPY validator. A non-success outcome is retained for diagnosis but makes the command fail.
The command accepts alternate instruction text, instruction ID/language, task ID, expert TOML,
seed, and episode ID without changing the recorder.
`--video_path` is optional and must resolve below `EMBODIEDAI_ARTIFACTS`. After a successful
episode passes NPY validation, the script derives an H.264 MP4 from the exact recorded RGB
frames. The MP4 stays outside the immutable episode and may be regenerated independently.

On 2026-08-25 the accepted RTX 5090 validation succeeded in 108 control steps. Its phase
counts were 28 move-above-cube, 13 descend-to-grasp, 10 close-gripper, 17 lift-cube,
23 move-above-goal, 15 descend-to-place, and 2 open-gripper actions. The validated episode is
`/root/autodl-tmp/EmbodiedAI/datasets/stage6-expert-validation-20260825/episode-stage6-expert-000002`.
It contains action shape `(108, 7)`, camera shape `(108, 3, 224, 224)`, the canonical
instruction, and state-machine/configuration provenance. No VLA training was run.

On 2026-08-26 the integrated `--video_path` path also passed with a fresh 108-step expert
episode. It produced a 224 x 224 H.264/yuv420p MP4 at 20 FPS with 108 readable frames and a
5.4-second duration under `EMBODIEDAI_ARTIFACTS`; neither the episode nor artifact root retained
a partial directory/file.

### Bounded skeleton check

Run from an explicitly configured project shell:

```bash
source scripts/bootstrap/project_env.sh
PYTHONPATH=src "$EMBODIEDAI_ENVS/isaac/bin/python" \
  scripts/sim/franka_pick_place_skeleton_smoke.py \
  --headless --enable_cameras --device cuda:0 --num_envs 1 --steps 2 \
  --png_path \
  "$EMBODIEDAI_ARTIFACTS/stage6/franka_pick_place_skeleton/camera_front_env0.png"
```

This check creates the registered environment, resets it, validates all scene entities and
contract-facing observation/action shapes, steps zero actions, and saves the final front-camera
observation from environment zero as a PNG before closing the simulator. If `--png_path` is
omitted, the same location under `EMBODIEDAI_ARTIFACTS` is used by default. It does not record
an episode and is not the later end-to-end Stage 6 smoke test.

On 2026-08-23 this check passed on the RTX 5090 allocation with driver 580.105.08 using one
CUDA environment and two steps. The action manager resolved dimensions 6 + 1, and the
observation manager produced `(9,)`, `(9,)`, `(3,)`, and `(3, 224, 224)` per environment.
The camera output was `uint8`, finite, and non-constant. Expected headless GLFW/display
warnings were emitted; Vulkan and offscreen rendering remained healthy.

### End-to-end dummy episode check

Run the Stage 6 structural episode smoke from an explicitly configured Isaac shell:

```bash
source scripts/bootstrap/project_env.sh
PYTHONPATH=src "$EMBODIEDAI_ENVS/isaac/bin/python" \
  scripts/sim/franka_pick_place_episode_smoke.py \
  --headless --enable_cameras --device cuda:0 --steps 2 --seed 0 \
  --output_root "$EMBODIEDAI_DATASETS/stage6-smoke" \
  --episode_id episode-stage6-smoke-000001
```

The script launches Isaac Lab and resets the environment once. It compares the resulting
joint position/velocity values with the configured Franka defaults and the cube value with the
fixed task reset position, then checks that the initial state is non-terminal. It records two
synchronized zero-action steps, publishes one immutable episode, and reopens it through the
file-level validator.

The episode contains one NPY array per observation field, synchronized observation/action
timestamps, action data, and `manifest.json`. Payloads and manifest are written under a unique
hidden partial directory. Only after hashes and metadata are complete is that directory
renamed to the final episode identifier. Existing final directories are never overwritten.
The bounded dummy rollout is deliberately marked `truncated` with reason
`smoke-test-step-limit`; it is not expert data and must not be used for VLA training.

On 2026-08-24 the check passed on the RTX 5090 allocation with Isaac Sim 5.1.0.0. The
published two-step episode covered simulation timestamps 0 and 50,000,000 ns, contained seven
hashed NPY payloads, and left no partial directory. The initial cube-to-goal distance was
0.250000 m and was 0.250050 m after the bounded zero-action rollout. The validated output is
under
`/root/autodl-tmp/EmbodiedAI/datasets/stage6-smoke-validation-20260824-a028136/`.

Runtime note: with the pinned Isaac Sim 5.1 headless RTX-camera runtime, explicitly calling
Gym `reset(seed=...)` a second time after a manual step ended the application with status zero
and no Python traceback. The accepted smoke follows the normal rollout boundary: one reset,
then bounded steps. It verifies the first reset against the configured defaults. Automatic
terminal reset behavior must receive a dedicated regression before expert collection begins.

## Stage 7 steps 1-3: LeRobot conversion and validation

Status: **implemented and validated on the 20-episode corpus on 2026-08-28**.

Run the one-time conversion from the isolated VLA environment. The output directory must not
already exist because converted datasets are immutable publications:

```bash
source scripts/bootstrap/project_env.sh
PYTHONPATH=src "$EMBODIEDAI_ENVS/vla/bin/python" \
  scripts/data/contract_episodes_to_lerobot.py \
  "$EMBODIEDAI_DATASETS/stage6-expert-batch-v1-20260827"/episode-stage6-batch-v1-* \
  --output_root "$EMBODIEDAI_DATASETS/stage7-franka-pick-place-batch-v1" \
  --repo_id embodiedai/franka-pick-place-stage7-batch-v1 \
  --storage videos
```

Validate the published dataset and regenerate the derived report without changing either source
or destination dataset:

```bash
PYTHONPATH=src "$EMBODIEDAI_ENVS/vla/bin/python" \
  scripts/data/validate_lerobot_dataset.py \
  --dataset_root "$EMBODIEDAI_DATASETS/stage7-franka-pick-place-batch-v1" \
  --source_root "$EMBODIEDAI_DATASETS/stage6-expert-batch-v1-20260827" \
  --repo_id embodiedai/franka-pick-place-stage7-batch-v1
```

The accepted dataset contains 20 episodes, 2,138 frames, and five exact instruction tasks at
20 Hz. The validator performs full provenance and table comparison, normalization-statistics
recomputation, and two-instance decoding of the first/last image in every episode. The report is
`$EMBODIEDAI_RUNS/stage7-validation/stage7-franka-pick-place-batch-v1.json`. Step 4 SmolVLA
preprocessing, inference, and training were not run.

## Stage 7 steps 4-6: reviewed implementation sequence

Status: **documentation only; no runnable command, code, processor, inference, or training job has
been approved**.

The implementation sequence is intentionally gated:

1. Implement and review the VLA-only preprocessing/postprocessing package described in
   `docs/SMOLVLA_PIPELINE.md`. Confirm project feature binding, explicit statistics, real-dataset
   processor validation, action round trips, and failure cases before loading the model.
2. After separate approval and a GPU-mode resource preflight, load the pinned base SmolVLA assets
   locally and run the bounded compatibility probe. Review finite canonical 7D output,
   deterministic noise handling, latency, memory, and provenance before the full offline baseline.
3. Freeze an episode-level evaluation protocol and produce the base offline metrics. Do not call
   this closed-loop success and do not start Isaac.
4. After separate approval, freeze the 20-episode train/validation split and training-only
   statistics, then run Step 6A's micro-overfit check and bounded PEFT feasibility job. Review
   clean-process adapter reload and base-versus-adapter metrics.
5. Do not begin Step 6B until a larger immutable corpus has repeated Stage 7 steps 1-3 and its
   train/validation/test split is approved.

All planned commands must use `$EMBODIEDAI_ENVS/vla`. Model, dataset, prediction, run, processor,
and checkpoint artifacts must use the external data-disk roots configured by
`scripts/bootstrap/project_env.sh`. Any package/lock change, new model download, dataset mutation,
or Isaac execution is outside this plan and requires review.
