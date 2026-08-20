#!/usr/bin/env python3
"""Run a bounded vectorized Isaac Lab environment smoke test."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    if args_cli.num_envs < 2:
        raise ValueError("--num_envs must be at least 2 for the vectorization smoke test")
    if args_cli.steps < 1:
        raise ValueError("--steps must be positive")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        for _ in range(args_cli.steps):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        if env.unwrapped.num_envs != args_cli.num_envs:
            raise RuntimeError(
                f"expected {args_cli.num_envs} environments, got {env.unwrapped.num_envs}"
            )

        print(
            "STAGE4_ISAACLAB_ENV_OK",
            f"task={args_cli.task}",
            f"num_envs={env.unwrapped.num_envs}",
            f"steps={args_cli.steps}",
            f"device={env.unwrapped.device}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
