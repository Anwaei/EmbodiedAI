"""Gym registration for the Franka pick-and-place task skeleton."""

import gymnasium as gym

TASK_ID = "EmbodiedAI-Franka-PickPlace-RGB-v0"

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

__all__ = ["TASK_ID"]
