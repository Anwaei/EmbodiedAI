# AGENTS.md

## Project

This repository implements a simulation-based embodied AI project
for language-conditioned robotic manipulation.

Before making architectural or dependency changes, read:

- docs/PROJECT_CONTEXT.md
- docs/ARCHITECTURE.md
- docs/ENVIRONMENT.md
- docs/ROADMAP.md

## Development principles

- The primary development machine is a remote Ubuntu workstation
  with an RTX 5090 GPU.
- Do not assume package versions.
- Before changing CUDA, PyTorch, Isaac Sim, Isaac Lab, ROS2,
  LeRobot, or VLA dependencies, check compatibility first.
- Keep simulation, VLA training, and ROS2 dependencies isolated
  when appropriate.
- Prefer reproducible environment files over ad-hoc pip installs.
- Do not install system-wide Python packages unless necessary.
- Record important environment changes in docs/ENVIRONMENT.md.

## Project scope

The core project is:

Isaac Sim / Isaac Lab
→ robotic manipulation
→ demonstration generation
→ VLA fine-tuning
→ RL refinement
→ ROS2 deployment interface
→ domain-randomization evaluation

A world model is a stretch goal and should not block the MVP.

## Engineering

- Keep code modular.
- Add tests for reusable components.
- Prefer configuration files over hard-coded parameters.
- Do not commit datasets, checkpoints, or large simulation assets.
- Keep commands needed to reproduce experiments documented.