# Isaac Lab Tasks

## Franka pick-and-place baseline

Stage 6 step 2 introduces the Gym task `EmbodiedAI-Franka-PickPlace-RGB-v0`.
It is a minimal `ManagerBasedRLEnv` configuration for validating the scene, deterministic
reset/evaluation behavior, and the shared observation/action boundary. It is not yet an
expert demonstrator or a training task.

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
linear speed, success, and failure values. Success requires centre distance at most 5 cm and
linear speed at most 0.10 m/s. Failure means the cube centre left the bounded workspace
`x=[0.10, 1.00]`, `y=[-0.50, 0.50]`, or fell below `z=-0.05` metres. Success and failure are
mutually exclusive and are wired into Isaac Lab termination terms; the ten-second time limit
remains a truncation. Reward remains an explicit zero placeholder because no training reward
was approved.

The following remain deferred:

- deterministic expert state machine and successful demonstration generation;
- demonstration generation and LeRobot conversion;
- learned policy inference or VLA training.

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
