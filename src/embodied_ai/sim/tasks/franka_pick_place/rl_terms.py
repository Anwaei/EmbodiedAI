"""Pure Torch state, phase, and reward functions for the Stage 9 task."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from embodied_ai.contracts.rl import PickPlaceRlPhase


def canonicalize_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Normalize WXYZ quaternions and choose the equivalent representation with w >= 0."""

    if quaternion_wxyz.ndim != 2 or quaternion_wxyz.shape[1] != 4:
        raise ValueError("quaternion batch must have shape (num_envs, 4)")
    norm = torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True).clamp_min(1.0e-8)
    normalized = quaternion_wxyz / norm
    sign = torch.where(normalized[:, :1] < 0.0, -1.0, 1.0)
    return normalized * sign


def assemble_state_observation(
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    tool_position: torch.Tensor,
    tool_quaternion: torch.Tensor,
    cube_position: torch.Tensor,
    cube_linear_velocity: torch.Tensor,
    goal_position: torch.Tensor,
    previous_action: torch.Tensor,
    phase_one_hot: torch.Tensor,
) -> torch.Tensor:
    """Assemble the reviewed fixed-order 52D standalone PPO observation."""

    expected_dimensions = (9, 9, 3, 4, 3, 3, 3, 7, len(PickPlaceRlPhase))
    values = (
        joint_position,
        joint_velocity,
        tool_position,
        canonicalize_quaternion(tool_quaternion),
        cube_position,
        cube_linear_velocity,
        goal_position,
        previous_action,
        phase_one_hot,
    )
    batch_size = joint_position.shape[0]
    for value, expected_dimension in zip(values, expected_dimensions, strict=True):
        if value.ndim != 2 or value.shape != (batch_size, expected_dimension):
            raise ValueError(
                f"state component has shape {tuple(value.shape)}, "
                f"expected {(batch_size, expected_dimension)}"
            )

    tool_to_cube = cube_position - tool_position
    cube_to_goal = goal_position - cube_position
    observation = torch.cat(
        (
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            tool_to_cube,
            cube_to_goal,
            values[7],
            values[8],
        ),
        dim=-1,
    )
    if observation.shape != (batch_size, 52):
        raise RuntimeError(f"standalone PPO observation changed to {tuple(observation.shape)}")
    return observation


def advance_phase(
    previous_phase: torch.Tensor,
    *,
    tool_to_cube_distance_m: torch.Tensor,
    bilateral_contact: torch.Tensor,
    cube_height_m: torch.Tensor,
    cube_to_goal_distance_m: torch.Tensor,
    gripper_open: torch.Tensor,
    reach_distance_m: float,
    lift_height_m: float,
    place_distance_m: float,
) -> torch.Tensor:
    """Advance at most one monotonic pick-and-place phase per control step."""

    if previous_phase.ndim != 1:
        raise ValueError("previous_phase must have shape (num_envs,)")
    expected_shape = previous_phase.shape
    for value in (
        tool_to_cube_distance_m,
        bilateral_contact,
        cube_height_m,
        cube_to_goal_distance_m,
        gripper_open,
    ):
        if value.shape != expected_shape:
            raise ValueError("phase inputs must have matching per-environment shapes")

    phase = previous_phase.clone()
    reach = previous_phase == int(PickPlaceRlPhase.REACH)
    phase[reach & bilateral_contact & (tool_to_cube_distance_m <= reach_distance_m)] = int(
        PickPlaceRlPhase.GRASP
    )

    grasp = previous_phase == int(PickPlaceRlPhase.GRASP)
    phase[grasp & (cube_height_m >= lift_height_m)] = int(PickPlaceRlPhase.LIFT)

    lift = previous_phase == int(PickPlaceRlPhase.LIFT)
    phase[
        lift
        & (cube_height_m >= lift_height_m)
        & (cube_to_goal_distance_m <= place_distance_m)
    ] = int(PickPlaceRlPhase.PLACE)

    place = previous_phase == int(PickPlaceRlPhase.PLACE)
    phase[place & gripper_open & (cube_to_goal_distance_m <= place_distance_m)] = int(
        PickPlaceRlPhase.RELEASE
    )
    return phase


def phase_one_hot(phase: torch.Tensor) -> torch.Tensor:
    if phase.ndim != 1:
        raise ValueError("phase must have shape (num_envs,)")
    return functional.one_hot(phase.to(dtype=torch.long), num_classes=len(PickPlaceRlPhase)).to(
        dtype=torch.float32
    )


def reach_reward(
    distance_m: torch.Tensor,
    phase: torch.Tensor,
    *,
    std_m: float,
) -> torch.Tensor:
    """Dense reach reward active only before a valid grasp."""

    return (1.0 - torch.tanh(distance_m / std_m)) * (
        phase == int(PickPlaceRlPhase.REACH)
    )


def grasp_reward(bilateral_contact: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Reward a current two-finger cube contact after the grasp transition."""

    return bilateral_contact.to(dtype=torch.float32) * (
        phase >= int(PickPlaceRlPhase.GRASP)
    )


def lift_reward(
    cube_height_m: torch.Tensor,
    phase: torch.Tensor,
    *,
    resting_height_m: float,
    target_height_m: float,
) -> torch.Tensor:
    """Normalized cube-height reward gated on a previously established grasp."""

    progress = ((cube_height_m - resting_height_m) / (target_height_m - resting_height_m)).clamp(
        0.0, 1.0
    )
    return progress * (phase >= int(PickPlaceRlPhase.GRASP))


def place_reward(
    distance_m: torch.Tensor,
    phase: torch.Tensor,
    *,
    std_m: float,
) -> torch.Tensor:
    """Dense goal-distance reward active only after the cube was lifted."""

    return (1.0 - torch.tanh(distance_m / std_m)) * (
        phase >= int(PickPlaceRlPhase.LIFT)
    )


def action_magnitude_penalty(action: torch.Tensor) -> torch.Tensor:
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError("action must have shape (num_envs, 7)")
    return torch.mean(torch.square(action[:, :6]), dim=-1)


def action_rate_penalty(action: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
    if action.shape != previous_action.shape or action.ndim != 2 or action.shape[1] != 7:
        raise ValueError("action and previous_action must have shape (num_envs, 7)")
    return torch.mean(torch.square(action - previous_action), dim=-1)


def gripper_toggle_penalty(action: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
    if action.shape != previous_action.shape or action.ndim != 2 or action.shape[1] != 7:
        raise ValueError("action and previous_action must have shape (num_envs, 7)")
    current = action[:, 6]
    previous = previous_action[:, 6]
    toggled = (previous != 0.0) & ((current >= 0.0) != (previous >= 0.0))
    return toggled.to(dtype=torch.float32)


__all__ = [
    "action_magnitude_penalty",
    "action_rate_penalty",
    "advance_phase",
    "assemble_state_observation",
    "canonicalize_quaternion",
    "grasp_reward",
    "gripper_toggle_penalty",
    "lift_reward",
    "phase_one_hot",
    "place_reward",
    "reach_reward",
]
