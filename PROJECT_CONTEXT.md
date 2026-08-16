# Project Context

## Goal

Build a portfolio-quality embodied AI project for applying to
robotics / embodied intelligence / robot learning positions.

The project should be achievable within a relatively short
development cycle while demonstrating a broad modern technical stack.

## Hardware

Local:
- Windows laptop
- RTX 4060 Laptop GPU, 8 GB VRAM
- Used primarily as a development terminal

Remote:
- Ubuntu GPU workstation
- RTX 5090, 32 GB VRAM
- Primary simulation and training environment

## Application

Primary scenario:
language-conditioned robotic manipulation.

Initial robot:
Franka Panda or a similarly mature manipulator.

No physical robot is currently available.
All experiments are simulation-based.

## Core architecture

Primary architecture:
Vision-Language-Action model.

Initial candidate:
SmolVLA / LeRobot ecosystem.

Possible later experiments:
OpenVLA or OpenVLA-OFT.

World models are considered an optional extension rather than
the initial core architecture.

## Desired technical coverage

The project should demonstrate:

1. Vision-Language-Action models
2. Multimodal learning
3. VLA fine-tuning / post-training
4. LoRA / PEFT where appropriate
5. Imitation learning
6. Reinforcement learning
7. Isaac Sim / Isaac Lab
8. ROS2
9. Domain randomization
10. Sim2Real-oriented training methodology
11. Reproducible ML engineering

## Proposed pipeline

Simulation
→ expert demonstrations
→ dataset
→ VLA supervised fine-tuning
→ policy evaluation
→ RL-based policy refinement
→ ROS2 inference/control interface

## RL

Do not begin with full-scale RLHF-style VLA post-training.

Initial preferred approach:

action = VLA_action + RL_residual

Candidate algorithms:
PPO or SAC.

## Sim2Real

There is currently no physical robot.

Therefore the project must not claim validated Sim2Real transfer.

Instead implement and evaluate:

- visual domain randomization
- lighting randomization
- camera randomization
- object randomization
- physics randomization
- action/sensor noise

and describe this as a:

"Sim2Real-oriented training and robustness pipeline."

## Tasks

Start simple and progressively increase difficulty:

1. Pick and place
2. Object/color-conditioned manipulation
3. Spatial-relation tasks
4. Multi-step manipulation

## ROS2

Training can remain Python-native.

ROS2 should mainly represent the deployment architecture:

camera / robot state / language
→ VLA policy node
→ robot action
→ controller
→ Isaac Sim ROS2 bridge

## Stretch goals

Only after the core pipeline is stable:

- learned latent dynamics model
- simple world model
- candidate-action prediction/evaluation
- model-based VLA action selection