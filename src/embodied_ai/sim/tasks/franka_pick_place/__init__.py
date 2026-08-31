"""Gym registration for the Franka pick-and-place task skeleton."""

import gymnasium as gym

from embodied_ai.contracts.rl import STANDALONE_PPO_TASK_ID

TASK_ID = "EmbodiedAI-Franka-PickPlace-RGB-v0"
RL_TASK_ID = STANDALONE_PPO_TASK_ID

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": (
                "embodied_ai.sim.tasks.franka_pick_place.env_cfg:FrankaPickPlaceEnvCfg"
            )
        },
        disable_env_checker=True,
    )

if RL_TASK_ID not in gym.registry:
    gym.register(
        id=RL_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": (
                "embodied_ai.sim.tasks.franka_pick_place.rl_env_cfg:"
                "FrankaPickPlacePPOEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "embodied_ai.sim.tasks.franka_pick_place.agents.rsl_rl_ppo_cfg:"
                "FrankaPickPlacePPORunnerCfg"
            ),
        },
        disable_env_checker=True,
    )

__all__ = ["RL_TASK_ID", "TASK_ID"]
