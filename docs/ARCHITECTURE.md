# Architecture

## System boundary

The MVP is split into processes and file contracts rather than a single Python runtime:

```text
Isaac Sim / Isaac Lab (Python 3.11)
  -> versioned episode data
LeRobot / SmolVLA (Python 3.12)
  -> checkpoint + policy metadata
Evaluation adapter (process boundary)
  -> observations/actions
ROS 2 Humble (system Python 3.10)
  -> deployment topics, services, and control safety
```

This boundary prevents Isaac, VLA, and ROS dependencies from contaminating each other.

## Roadmap pipeline ownership

The post-environment roadmap follows the runtime and artifact boundaries above:

| Stage | Pipeline owner | Primary output |
|---|---|---|
| 6 | Isaac Demonstration Pipeline | validated immutable expert episodes and derived previews |
| 7 | LeRobot + VLA Training Pipeline | validated `LeRobotDataset` plus SmolVLA base/PEFT artifacts |
| 8 | Closed-loop Policy Evaluation | Isaac rollout metrics and baseline comparisons |
| 9 | RL Policy Refinement | PPO/SAC and optional bounded residual-policy results |
| 10 | ROS 2 deployment boundary | deployment messages, nodes, and safety-checked control path |
| 11 | Reproducibility and robustness gate | clean rebuild and perturbation/robustness report |

Stage 6 ends at contract-compliant raw demonstration data. Stage 7 owns Contract → LeRobot
conversion and offline VLA training. Stage 8 owns online policy/Isaac interaction. Stage 9 is
gated on the Stage 8 imitation baseline and may refine it without changing the public action
contract. ROS 2 deployment and the final robustness gate remain separate Stages 10 and 11.

## Repository modules

- `src/embodied_ai/contracts`: dependency-light data and policy interface schemas.
- `src/embodied_ai/sim`: Isaac Lab tasks, scenes, sensors, experts, demonstration collection,
  recording, and randomization.
- `src/embodied_ai/data`: dataset validation and LeRobot conversion.
- `src/embodied_ai/policies`: policy metadata and LeRobot/SmolVLA adapters.
- `src/embodied_ai/rl`: optional residual policy and reward components.
- `src/embodied_ai/evaluation`: task metrics and robustness sweeps.
- `ros2_ws/src`: ROS 2 packages added during the approved deployment stage.

## Storage and communication

- Simulator outputs immutable episodes described by `docs/DATA_FORMAT.md`.
- The Isaac-side NPY recorder writes contract-keyed tensors and a manifest to a private
  partial directory, then atomically publishes the complete episode. It imports neither
  LeRobot nor the VLA training stack.
- Training emits checkpoints plus metadata defining observations, normalization, action
  dimensions, control frequency, and source revision.
- Online evaluation uses a narrow local RPC or ROS 2 boundary; it does not import Isaac and
  LeRobot into the same interpreter.
- Standard ROS messages are preferred for the MVP. Custom interfaces require explicit
  Python 3.10 and Python 3.11 build validation.

## Demonstration generation boundary

Stage 6 step 6 keeps task semantics, language, control, and storage as separate concerns:

```text
task definition + instruction selection
  -> Isaac-side expert
  -> canonical ActionSchema action
  -> Isaac Lab step
  -> one immutable episode per environment
  -> later LeRobot conversion in the VLA environment
```

- `task` is a stable machine-readable task identifier whose reset and success semantics are
  owned by the task definition.
- `instruction` is the exact natural-language command attached to the episode. Multiple
  instructions may describe one task without duplicating the task implementation.
- `expert` describes the source of the actions and its revision. State machines, learned RL
  policies, and teleoperation must implement the same Isaac-side action-producing interface.
- Experts may read privileged simulator state, but they emit only the public normalized action
  contract. Expert-only state is not silently added to the eventual VLA policy input.
- The collector owns rollout lifecycle and recording; an expert does not write files. With
  vectorized environments, the collector maintains one recorder and one terminal outcome per
  environment rather than storing an environment batch as one episode.

## Safety and scope

- The first task is Franka pick-and-place.
- Residual RL follows a working imitation-learning baseline.
- Domain randomization evaluation is described as Sim2Real-oriented; no physical-robot
  transfer claim is made.
- World-model work remains a non-blocking stretch goal.
