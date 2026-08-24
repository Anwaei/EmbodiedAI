# Isaac Lab Tasks

## Franka pick-and-place skeleton

Stage 6 step 2 introduces the Gym task `EmbodiedAI-Franka-PickPlace-RGB-v0`.
It is a minimal `ManagerBasedRLEnv` configuration for validating the scene and the shared
observation/action boundary. It is not yet an expert demonstrator or a training task.

### Scene

Each replicated environment contains:

- a Franka Panda using Isaac Lab's `FRANKA_PANDA_HIGH_PD_CFG`, selected because relative
  differential IK requires stable task-space tracking;
- a kinematic cuboid table whose top surface is at `z=0` in the environment frame;
- a dynamic 5 cm, 50 g cube initialized above the table;
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
| `object.cube.position` | `cube_position` | `(3,)` | `float32`; environment frame, metres |
| `camera.front.rgb` | `camera_front_rgb` | `(3, 224, 224)` | `uint8`; camera optical frame |

The nine robot values have an explicit order: seven Panda arm joints followed by the two
finger joints. Isaac camera tensors are HWC RGB or RGBA depending on the renderer path. The
task adapter removes a possible alpha channel and converts to contiguous CHW `uint8`, so the
manager output already matches the shared observation contract.

### Registration and imports

Importing `embodied_ai.sim.tasks` registers the task. All Isaac-specific imports remain under
`src/embodied_ai/sim`; the task-specific schema instance under
`src/embodied_ai/contracts/tasks` remains standard-library-only. A unit test rejects LeRobot,
Transformers, Datasets, Accelerate, or PEFT imports from the simulation package.

### Current reset, reward, and termination behavior

The skeleton uses Isaac Lab's generic reset-to-default event so it can be instantiated and
stepped. Its reward is an explicit zero placeholder and its only termination is the ten-second
time limit. These are scaffolding mechanics, not the Stage 6 deterministic task/evaluation
interface.

The following remain deferred:

- seeded cube and goal placement plus a deterministic expert state machine;
- success, failure, and truncation evaluation rules;
- episode payload recording and atomic manifest publication;
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
