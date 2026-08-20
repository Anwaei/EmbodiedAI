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
| 6 | Demonstration-to-policy integration | Not approved |
| 7 | ROS 2 deployment boundary | Not approved |
| 8 | Reproducibility and robustness gate | Not approved |

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

## Deferred work

- Multi-step manipulation follows the single-step baseline.
- OpenVLA/OpenVLA-OFT experiments follow SmolVLA.
- World models and learned dynamics do not block the MVP.
