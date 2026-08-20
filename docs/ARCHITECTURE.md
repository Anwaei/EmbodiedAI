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

## Repository modules

- `src/embodied_ai/contracts`: dependency-light data and policy interface schemas.
- `src/embodied_ai/sim`: Isaac Lab tasks, scenes, sensors, recording, and randomization.
- `src/embodied_ai/data`: dataset validation and LeRobot conversion.
- `src/embodied_ai/policies`: policy metadata and LeRobot/SmolVLA adapters.
- `src/embodied_ai/rl`: optional residual policy and reward components.
- `src/embodied_ai/evaluation`: task metrics and robustness sweeps.
- `ros2_ws/src`: ROS 2 packages added during the approved deployment stage.

## Storage and communication

- Simulator outputs immutable episodes described by `docs/DATA_FORMAT.md`.
- Training emits checkpoints plus metadata defining observations, normalization, action
  dimensions, control frequency, and source revision.
- Online evaluation uses a narrow local RPC or ROS 2 boundary; it does not import Isaac and
  LeRobot into the same interpreter.
- Standard ROS messages are preferred for the MVP. Custom interfaces require explicit
  Python 3.10 and Python 3.11 build validation.

## Safety and scope

- The first task is Franka pick-and-place.
- Residual RL follows a working imitation-learning baseline.
- Domain randomization evaluation is described as Sim2Real-oriented; no physical-robot
  transfer claim is made.
- World-model work remains a non-blocking stretch goal.
