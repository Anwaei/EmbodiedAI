#!/usr/bin/env python3
"""Launch and inspect the Stage 6 Franka task skeleton without recording an episode."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import embodied_ai.sim.tasks  # noqa: F401
from embodied_ai.contracts.tasks.franka_pick_place import (
    FRANKA_PICK_PLACE_ACTION_SCHEMA,
    FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
)
from embodied_ai.sim.tasks.franka_pick_place import TASK_ID
from embodied_ai.sim.tasks.franka_pick_place.env_cfg import (
    CONTRACT_OBSERVATION_TERM_MAP,
    FrankaPickPlaceEnvCfg,
)


def _validate_observations(observations: dict[str, object], num_envs: int) -> None:
    policy = observations.get("policy")
    if not isinstance(policy, dict):
        raise RuntimeError(
            f"expected an unconcatenated policy observation dict, got {type(policy)}"
        )

    for field in FRANKA_PICK_PLACE_OBSERVATION_SCHEMA.fields:
        term_name = CONTRACT_OBSERVATION_TERM_MAP[field.key]
        value = policy.get(term_name)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"missing tensor observation term {term_name!r}")
        expected_shape = (num_envs, *field.shape)
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"observation {field.key!r} has shape {tuple(value.shape)}, "
                f"expected {expected_shape}"
            )
        if field.key == "camera.front.rgb" and value.dtype is not torch.uint8:
            raise RuntimeError(f"camera dtype is {value.dtype}, expected torch.uint8")
        if field.key != "camera.front.rgb" and value.dtype is not torch.float32:
            raise RuntimeError(f"state dtype is {value.dtype}, expected torch.float32")
        if field.key == "camera.front.rgb" and value.min() == value.max():
            raise RuntimeError("camera image is constant; the scene may not be visible")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"observation {field.key!r} contains non-finite values")


def main() -> None:
    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be positive")
    if args_cli.steps < 1:
        raise ValueError("--steps must be positive")

    env_cfg = FrankaPickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = 0
    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        observations, _ = env.reset(seed=0)
        _validate_observations(observations, args_cli.num_envs)

        expected_action_shape = (args_cli.num_envs, FRANKA_PICK_PLACE_ACTION_SCHEMA.dimension)
        if tuple(env.action_space.shape) != expected_action_shape:
            raise RuntimeError(
                f"action space has shape {env.action_space.shape}, expected {expected_action_shape}"
            )

        for _ in range(args_cli.steps):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                observations, _, _, _, _ = env.step(actions)
            _validate_observations(observations, args_cli.num_envs)

        scene_keys = set(env.unwrapped.scene.keys())
        expected_scene_keys = {"robot", "table", "cube", "camera_front"}
        if not expected_scene_keys.issubset(scene_keys):
            raise RuntimeError(f"missing scene entities: {expected_scene_keys - scene_keys}")

        print(
            "STAGE6_TASK_SKELETON_OK",
            f"task={TASK_ID}",
            f"num_envs={args_cli.num_envs}",
            f"steps={args_cli.steps}",
            f"action_dim={FRANKA_PICK_PLACE_ACTION_SCHEMA.dimension}",
            "camera_shape=(3,224,224)",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
