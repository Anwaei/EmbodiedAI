"""Vectorized evaluation interface for the Franka pick-and-place task."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from embodied_ai.contracts.tasks.franka_pick_place import (
    FAILURE_MINIMUM_Z_ENV_M,
    FAILURE_X_BOUNDS_ENV_M,
    FAILURE_Y_BOUNDS_ENV_M,
    GOAL_POSITION_ENV_M,
    SUCCESS_GRIPPER_OPEN_POSITION_M,
    SUCCESS_LINEAR_SPEED_TOLERANCE_M_S,
    SUCCESS_POSITION_TOLERANCE_M,
)


@dataclass(frozen=True, slots=True)
class PickPlaceEvaluation:
    """Per-environment task metrics and mutually exclusive terminal flags."""

    cube_position_env_m: torch.Tensor
    goal_position_env_m: torch.Tensor
    position_error_m: torch.Tensor
    linear_speed_m_s: torch.Tensor
    gripper_open: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor


def cube_position_env(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Return cube positions relative to each replicated environment origin."""

    cube = env.scene[asset_cfg.name]
    return cube.data.root_pos_w - env.scene.env_origins


def evaluate_cube_state(
    cube_position_env_m: torch.Tensor,
    cube_linear_velocity_w_m_s: torch.Tensor,
    gripper_open: torch.Tensor | None = None,
) -> PickPlaceEvaluation:
    """Evaluate already adapted cube state without reading simulator handles."""

    if cube_position_env_m.ndim != 2 or cube_position_env_m.shape[-1] != 3:
        raise ValueError("cube positions must have shape (num_envs, 3)")
    if cube_linear_velocity_w_m_s.shape != cube_position_env_m.shape:
        raise ValueError("cube linear velocities must match cube position shape")
    if gripper_open is None:
        gripper_open = torch.ones(
            cube_position_env_m.shape[0], dtype=torch.bool, device=cube_position_env_m.device
        )
    if gripper_open.shape != (cube_position_env_m.shape[0],):
        raise ValueError("gripper_open must have shape (num_envs,)")

    goal = cube_position_env_m.new_tensor(GOAL_POSITION_ENV_M).expand_as(
        cube_position_env_m
    )
    position_error = torch.linalg.vector_norm(cube_position_env_m - goal, dim=-1)
    linear_speed = torch.linalg.vector_norm(cube_linear_velocity_w_m_s, dim=-1)
    success = (
        (position_error <= SUCCESS_POSITION_TOLERANCE_M)
        & (linear_speed <= SUCCESS_LINEAR_SPEED_TOLERANCE_M_S)
        & gripper_open
    )

    x = cube_position_env_m[:, 0]
    y = cube_position_env_m[:, 1]
    z = cube_position_env_m[:, 2]
    failure = (
        (x < FAILURE_X_BOUNDS_ENV_M[0])
        | (x > FAILURE_X_BOUNDS_ENV_M[1])
        | (y < FAILURE_Y_BOUNDS_ENV_M[0])
        | (y > FAILURE_Y_BOUNDS_ENV_M[1])
        | (z < FAILURE_MINIMUM_Z_ENV_M)
    )
    success = success & ~failure

    return PickPlaceEvaluation(
        cube_position_env_m=cube_position_env_m,
        goal_position_env_m=goal,
        position_error_m=position_error,
        linear_speed_m_s=linear_speed,
        gripper_open=gripper_open,
        success=success,
        failure=failure,
    )


def evaluate_pick_place(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> PickPlaceEvaluation:
    """Read the current scene and return the public vectorized task evaluation."""

    cube = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    finger_joint_ids, _ = robot.find_joints("panda_finger_joint.*")
    gripper_open = torch.all(
        robot.data.joint_pos[:, finger_joint_ids] >= SUCCESS_GRIPPER_OPEN_POSITION_M,
        dim=-1,
    )
    return evaluate_cube_state(
        cube_position_env(env, asset_cfg),
        cube.data.root_lin_vel_w,
        gripper_open,
    )


def task_success(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Isaac Lab termination term for successful placement."""

    return evaluate_pick_place(env, asset_cfg).success


def task_failure(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Isaac Lab termination term for a cube outside the bounded workspace."""

    return evaluate_pick_place(env, asset_cfg).failure
