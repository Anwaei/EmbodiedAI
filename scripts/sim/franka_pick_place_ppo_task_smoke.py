#!/usr/bin/env python3
"""Launch the Stage 9 state-only task and validate reset, reward, and termination wiring."""

from __future__ import annotations

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--seed", type=int, default=20260901)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=False, headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import embodied_ai.sim.tasks  # noqa: E402, F401
from embodied_ai.contracts.rl import (  # noqa: E402
    STANDALONE_PPO_ACTION_PROFILE,
    STANDALONE_PPO_OBSERVATION_PROFILE,
)
from embodied_ai.sim.tasks.franka_pick_place import RL_TASK_ID  # noqa: E402
from embodied_ai.sim.tasks.franka_pick_place.agents import (  # noqa: E402
    FrankaPickPlacePPORunnerCfg,
)
from embodied_ai.sim.tasks.franka_pick_place.rl_env_cfg import (  # noqa: E402
    STAGE9_STANDALONE_CONFIG,
    FrankaPickPlacePPOEnvCfg,
)
from embodied_ai.sim.tasks.franka_pick_place.rl_mdp import (  # noqa: E402
    runtime_goal_position,
    termination_diagnostics,
)


def _validate_observation(observations: dict[str, torch.Tensor], num_envs: int) -> None:
    policy = observations.get("policy")
    if not isinstance(policy, torch.Tensor):
        raise RuntimeError(f"expected one concatenated policy tensor, got {type(policy)}")
    expected = (num_envs, STANDALONE_PPO_OBSERVATION_PROFILE.dimension)
    if tuple(policy.shape) != expected:
        raise RuntimeError(f"policy observation is {tuple(policy.shape)}, expected {expected}")
    if policy.dtype is not torch.float32 or not torch.isfinite(policy).all():
        raise RuntimeError("policy observation must be finite float32")


def _inside(values: torch.Tensor, bounds: tuple[float, float]) -> bool:
    return bool(torch.all((values >= bounds[0]) & (values <= bounds[1])))


def main() -> None:
    if args_cli.num_envs < 1 or args_cli.steps < 1 or args_cli.seed < 0:
        raise ValueError("num_envs/steps must be positive and seed must be non-negative")

    env_cfg = FrankaPickPlacePPOEnvCfg()
    runner_cfg = FrankaPickPlacePPORunnerCfg()
    if runner_cfg.num_steps_per_env != STAGE9_STANDALONE_CONFIG.ppo.num_steps_per_env:
        raise RuntimeError("RSL-RL runner drifted from the reviewed Stage 9 TOML")
    backend = STAGE9_STANDALONE_CONFIG.identity.backend
    if version(backend.package) != backend.package_version:
        raise RuntimeError("installed RSL-RL version does not match the reviewed identity")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env = gym.make(RL_TASK_ID, cfg=env_cfg)
    try:
        observations, _ = env.reset(seed=args_cli.seed)
        _validate_observation(observations, args_cli.num_envs)

        expected_action_shape = (
            args_cli.num_envs,
            STANDALONE_PPO_ACTION_PROFILE.action_dimension,
        )
        if tuple(env.action_space.shape) != expected_action_shape:
            raise RuntimeError(
                f"action space is {env.action_space.shape}, expected {expected_action_shape}"
            )

        unwrapped = env.unwrapped
        scene_keys = set(unwrapped.scene.keys())
        if "camera_front" in scene_keys:
            raise RuntimeError("state-only PPO scene unexpectedly contains the RGB camera")
        expected_contacts = {"left_finger_cube_contact", "right_finger_cube_contact"}
        if not expected_contacts.issubset(scene_keys):
            raise RuntimeError(
                f"PPO scene is missing contact sensors: {expected_contacts - scene_keys}"
            )
        for sensor_name in sorted(expected_contacts):
            force_matrix = unwrapped.scene[sensor_name].data.force_matrix_w
            if force_matrix is None or tuple(force_matrix.shape[:2]) != (
                args_cli.num_envs,
                1,
            ):
                shape = None if force_matrix is None else tuple(force_matrix.shape)
                raise RuntimeError(f"contact sensor {sensor_name} has invalid shape {shape}")

        cube = unwrapped.scene["cube"].data.root_pos_w - unwrapped.scene.env_origins
        goal = runtime_goal_position(unwrapped)
        reset = STAGE9_STANDALONE_CONFIG.reset
        if not _inside(cube[:, 0], reset.cube_x_m) or not _inside(cube[:, 1], reset.cube_y_m):
            raise RuntimeError("sampled cube reset is outside reviewed training ranges")
        if not _inside(goal[:, 0], reset.goal_x_m) or not _inside(goal[:, 1], reset.goal_y_m):
            raise RuntimeError("sampled goal is outside reviewed training ranges")
        if args_cli.num_envs > 1 and torch.allclose(cube[0], cube[1]) and torch.allclose(
            goal[0], goal[1]
        ):
            raise RuntimeError("vectorized reset did not independently sample environments")

        diagnostics = termination_diagnostics(unwrapped)
        if any(bool(mask.any()) for mask in diagnostics.values()):
            raise RuntimeError("reviewed reset unexpectedly starts in a terminal state")

        reward_min = float("inf")
        reward_max = float("-inf")
        for _ in range(args_cli.steps):
            actions = torch.zeros(
                expected_action_shape,
                dtype=torch.float32,
                device=unwrapped.device,
            )
            with torch.inference_mode():
                observations, reward, terminated, truncated, _ = env.step(actions)
            _validate_observation(observations, args_cli.num_envs)
            if reward.shape != (args_cli.num_envs,) or not torch.isfinite(reward).all():
                raise RuntimeError("reward must be a finite per-environment tensor")
            if torch.any(terminated | truncated):
                raise RuntimeError("bounded zero-action task smoke terminated unexpectedly")
            reward_min = min(reward_min, float(reward.min()))
            reward_max = max(reward_max, float(reward.max()))

        print(
            "STAGE9_PPO_TASK_OK",
            f"task={RL_TASK_ID}",
            f"num_envs={args_cli.num_envs}",
            f"steps={args_cli.steps}",
            f"observation_dim={STANDALONE_PPO_OBSERVATION_PROFILE.dimension}",
            f"action_dim={STANDALONE_PPO_ACTION_PROFILE.action_dimension}",
            f"reward_range=[{reward_min:.6f},{reward_max:.6f}]",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
