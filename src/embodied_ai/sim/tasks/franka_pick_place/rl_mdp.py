"""Isaac adapters for the Stage 9 Franka standalone PPO task."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp as isaac_mdp

from embodied_ai.contracts.rl import PickPlaceRlPhase
from embodied_ai.contracts.tasks.franka_pick_place import (
    FAILURE_MINIMUM_Z_ENV_M,
    FAILURE_X_BOUNDS_ENV_M,
    FAILURE_Y_BOUNDS_ENV_M,
    GOAL_MARKER_SIZE_M,
    GOAL_POSITION_ENV_M,
    SUCCESS_GRIPPER_OPEN_POSITION_M,
    SUCCESS_LINEAR_SPEED_TOLERANCE_M_S,
    SUCCESS_POSITION_TOLERANCE_M,
)

from . import rl_terms


@dataclass(slots=True)
class Stage9RlRuntimeState:
    """Per-environment state that is reset independently from PPO network state."""

    goal_position_env_m: torch.Tensor
    phase: torch.Tensor
    last_phase_update_step: torch.Tensor


def _runtime_state(env: ManagerBasedRLEnv) -> Stage9RlRuntimeState:
    state = getattr(env, "_embodiedai_stage9_rl_state", None)
    if state is None:
        goal = torch.tensor(GOAL_POSITION_ENV_M, dtype=torch.float32, device=env.device).repeat(
            env.num_envs, 1
        )
        state = Stage9RlRuntimeState(
            goal_position_env_m=goal,
            phase=torch.full(
                (env.num_envs,),
                int(PickPlaceRlPhase.REACH),
                dtype=torch.long,
                device=env.device,
            ),
            last_phase_update_step=torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            ),
        )
        env._embodiedai_stage9_rl_state = state
    return state


def runtime_goal_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the per-environment semantic cube-centre goal in the environment frame."""

    return _runtime_state(env).goal_position_env_m.clone()


def _cube_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["cube"].data.root_pos_w - env.scene.env_origins


def _tool_pose(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    frame_data = env.scene["ee_frame"].data
    return frame_data.target_pos_source[:, 0, :], frame_data.target_quat_source[:, 0, :]


def _gripper_open(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    finger_joint_ids, _ = robot.find_joints("panda_finger_joint.*")
    return torch.all(
        robot.data.joint_pos[:, finger_joint_ids] >= SUCCESS_GRIPPER_OPEN_POSITION_M,
        dim=-1,
    )


def bilateral_cube_contact(
    env: ManagerBasedRLEnv,
    force_threshold_n: float = 0.5,
) -> torch.Tensor:
    """Return whether both finger links exceed the cube-filtered normal-force threshold."""

    per_finger: list[torch.Tensor] = []
    for sensor_name in ("left_finger_cube_contact", "right_finger_cube_contact"):
        force_matrix = env.scene[sensor_name].data.force_matrix_w
        if force_matrix is None:
            per_finger.append(
                torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
            )
            continue
        force_norm = torch.linalg.vector_norm(torch.nan_to_num(force_matrix), dim=-1)
        per_finger.append(force_norm.reshape(env.num_envs, -1).amax(dim=-1))
    return torch.all(torch.stack(per_finger, dim=-1) >= force_threshold_n, dim=-1)


def _phase(
    env: ManagerBasedRLEnv,
    *,
    reach_distance_m: float,
    grasp_force_n: float,
    lift_height_m: float,
    place_distance_m: float,
) -> torch.Tensor:
    """Update the monotonic phase at most once for each simulator control step."""

    state = _runtime_state(env)
    current_step = int(env.common_step_counter)
    requires_update = state.last_phase_update_step != current_step
    if torch.any(requires_update):
        tool_position, _ = _tool_pose(env)
        cube_position = _cube_position(env)
        next_phase = rl_terms.advance_phase(
            state.phase,
            tool_to_cube_distance_m=torch.linalg.vector_norm(
                cube_position - tool_position, dim=-1
            ),
            bilateral_contact=bilateral_cube_contact(
                env, force_threshold_n=grasp_force_n
            ),
            cube_height_m=cube_position[:, 2],
            cube_to_goal_distance_m=torch.linalg.vector_norm(
                cube_position - state.goal_position_env_m, dim=-1
            ),
            gripper_open=_gripper_open(env),
            reach_distance_m=reach_distance_m,
            lift_height_m=lift_height_m,
            place_distance_m=place_distance_m,
        )
        state.phase[requires_update] = next_phase[requires_update]
        state.last_phase_update_step[requires_update] = current_step
    return state.phase.clone()


def state_observation(
    env: ManagerBasedRLEnv,
    reach_distance_m: float = 0.065,
    grasp_force_n: float = 0.5,
    lift_height_m: float = 0.09,
    place_distance_m: float = 0.08,
) -> torch.Tensor:
    """Extract the fixed-order 52D state-only actor observation."""

    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(
        [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint1",
            "panda_finger_joint2",
        ],
        preserve_order=True,
    )
    tool_position, tool_quaternion = _tool_pose(env)
    cube_position = _cube_position(env)
    phase = _phase(
        env,
        reach_distance_m=reach_distance_m,
        grasp_force_n=grasp_force_n,
        lift_height_m=lift_height_m,
        place_distance_m=place_distance_m,
    )
    return rl_terms.assemble_state_observation(
        joint_position=robot.data.joint_pos[:, joint_ids],
        joint_velocity=robot.data.joint_vel[:, joint_ids],
        tool_position=tool_position,
        tool_quaternion=tool_quaternion,
        cube_position=cube_position,
        cube_linear_velocity=env.scene["cube"].data.root_lin_vel_w,
        goal_position=runtime_goal_position(env),
        previous_action=env.action_manager.action,
        phase_one_hot=rl_terms.phase_one_hot(phase),
    )


def _phase_and_distances(
    env: ManagerBasedRLEnv,
    *,
    reach_distance_m: float,
    grasp_force_n: float,
    lift_height_m: float,
    place_distance_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = _phase(
        env,
        reach_distance_m=reach_distance_m,
        grasp_force_n=grasp_force_n,
        lift_height_m=lift_height_m,
        place_distance_m=place_distance_m,
    )
    cube_position = _cube_position(env)
    tool_position, _ = _tool_pose(env)
    return (
        phase,
        torch.linalg.vector_norm(cube_position - tool_position, dim=-1),
        cube_position[:, 2],
        torch.linalg.vector_norm(cube_position - runtime_goal_position(env), dim=-1),
    )


def reach_reward(
    env: ManagerBasedRLEnv,
    *,
    std_m: float,
    reach_distance_m: float,
    grasp_force_n: float,
    lift_height_m: float,
    place_distance_m: float,
) -> torch.Tensor:
    phase, tool_distance, _, _ = _phase_and_distances(
        env,
        reach_distance_m=reach_distance_m,
        grasp_force_n=grasp_force_n,
        lift_height_m=lift_height_m,
        place_distance_m=place_distance_m,
    )
    return rl_terms.reach_reward(tool_distance, phase, std_m=std_m)


def grasp_reward(
    env: ManagerBasedRLEnv,
    *,
    reach_distance_m: float,
    grasp_force_n: float,
    lift_height_m: float,
    place_distance_m: float,
) -> torch.Tensor:
    phase, _, _, _ = _phase_and_distances(
        env,
        reach_distance_m=reach_distance_m,
        grasp_force_n=grasp_force_n,
        lift_height_m=lift_height_m,
        place_distance_m=place_distance_m,
    )
    return rl_terms.grasp_reward(
        bilateral_cube_contact(env, force_threshold_n=grasp_force_n), phase
    )


def lift_reward(
    env: ManagerBasedRLEnv,
    *,
    resting_height_m: float,
    target_height_m: float,
    reach_distance_m: float,
    grasp_force_n: float,
    lift_height_m: float,
    place_distance_m: float,
) -> torch.Tensor:
    phase, _, cube_height, _ = _phase_and_distances(
        env,
        reach_distance_m=reach_distance_m,
        grasp_force_n=grasp_force_n,
        lift_height_m=lift_height_m,
        place_distance_m=place_distance_m,
    )
    return rl_terms.lift_reward(
        cube_height,
        phase,
        resting_height_m=resting_height_m,
        target_height_m=target_height_m,
    )


def place_reward(
    env: ManagerBasedRLEnv,
    *,
    std_m: float,
    reach_distance_m: float,
    grasp_force_n: float,
    lift_height_m: float,
    place_distance_m: float,
) -> torch.Tensor:
    phase, _, _, goal_distance = _phase_and_distances(
        env,
        reach_distance_m=reach_distance_m,
        grasp_force_n=grasp_force_n,
        lift_height_m=lift_height_m,
        place_distance_m=place_distance_m,
    )
    return rl_terms.place_reward(goal_distance, phase, std_m=std_m)


def task_success_dynamic_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube_position = _cube_position(env)
    cube_speed = torch.linalg.vector_norm(env.scene["cube"].data.root_lin_vel_w, dim=-1)
    goal_error = torch.linalg.vector_norm(cube_position - runtime_goal_position(env), dim=-1)
    return (
        (goal_error <= SUCCESS_POSITION_TOLERANCE_M)
        & (cube_speed <= SUCCESS_LINEAR_SPEED_TOLERANCE_M_S)
        & _gripper_open(env)
        & ~task_failure_dynamic(env)
    )


def task_failure_dynamic(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube_position = _cube_position(env)
    x, y, z = cube_position.unbind(dim=-1)
    return (
        (x < FAILURE_X_BOUNDS_ENV_M[0])
        | (x > FAILURE_X_BOUNDS_ENV_M[1])
        | (y < FAILURE_Y_BOUNDS_ENV_M[0])
        | (y > FAILURE_Y_BOUNDS_ENV_M[1])
        | (z < FAILURE_MINIMUM_Z_ENV_M)
    )


def invalid_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    tool_position, tool_quaternion = _tool_pose(env)
    tensors = (
        robot.data.joint_pos,
        robot.data.joint_vel,
        cube.data.root_pos_w,
        cube.data.root_lin_vel_w,
        tool_position,
        tool_quaternion,
        runtime_goal_position(env),
    )
    finite = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for value in tensors:
        finite &= torch.all(torch.isfinite(value.reshape(env.num_envs, -1)), dim=-1)
    return ~finite


def success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    return task_success_dynamic_goal(env).to(dtype=torch.float32)


def failure_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the public workspace-failure mask for terminal penalty weighting."""

    return task_failure_dynamic(env).to(dtype=torch.float32)


def action_magnitude_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return rl_terms.action_magnitude_penalty(env.action_manager.action)


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return rl_terms.action_rate_penalty(
        env.action_manager.action, env.action_manager.prev_action
    )


def gripper_toggle_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return rl_terms.gripper_toggle_penalty(
        env.action_manager.action, env.action_manager.prev_action
    )


def termination_diagnostics(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Return explicit mutually auditable terminal masks for logging and smokes."""

    return {
        "success": task_success_dynamic_goal(env),
        "failure": task_failure_dynamic(env),
        "invalid_state": invalid_state(env),
        "time_out": env.episode_length_buf >= env.max_episode_length,
    }


def _sample_range(
    count: int,
    bounds: tuple[float, float],
    *,
    device: str,
) -> torch.Tensor:
    return bounds[0] + torch.rand(count, device=device) * (bounds[1] - bounds[0])


def reset_franka_pick_place_rl(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    *,
    cube_x_m: tuple[float, float],
    cube_y_m: tuple[float, float],
    goal_x_m: tuple[float, float],
    goal_y_m: tuple[float, float],
    cube_z_m: float,
    goal_z_m: float,
) -> None:
    """Reset robot state and independently sample cube/goal geometry for selected envs."""

    isaac_mdp.reset_scene_to_default(env, env_ids)
    count = int(env_ids.numel())
    cube_position = torch.stack(
        (
            _sample_range(count, cube_x_m, device=env.device),
            _sample_range(count, cube_y_m, device=env.device),
            torch.full((count,), cube_z_m, device=env.device),
        ),
        dim=-1,
    )
    goal_position = torch.stack(
        (
            _sample_range(count, goal_x_m, device=env.device),
            _sample_range(count, goal_y_m, device=env.device),
            torch.full((count,), goal_z_m, device=env.device),
        ),
        dim=-1,
    )

    set_franka_pick_place_rl_geometry(
        env,
        env_ids,
        cube_position_env_m=cube_position,
        goal_position_env_m=goal_position,
    )


def set_franka_pick_place_rl_geometry(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    *,
    cube_position_env_m: torch.Tensor,
    goal_position_env_m: torch.Tensor,
) -> None:
    """Apply exact per-environment cube/goal geometry after a reset.

    Training reset sampling and held-out evaluation share this single write path,
    which prevents the simulator marker and semantic runtime goal from diverging.
    The caller remains responsible for resetting the robot when a fresh episode is
    required.
    """

    count = int(env_ids.numel())
    expected_shape = (count, 3)
    if cube_position_env_m.shape != expected_shape:
        raise ValueError(
            f"cube_position_env_m must have shape {expected_shape}, "
            f"got {tuple(cube_position_env_m.shape)}"
        )
    if goal_position_env_m.shape != expected_shape:
        raise ValueError(
            f"goal_position_env_m must have shape {expected_shape}, "
            f"got {tuple(goal_position_env_m.shape)}"
        )
    cube_position = cube_position_env_m.to(device=env.device, dtype=torch.float32)
    goal_position = goal_position_env_m.to(device=env.device, dtype=torch.float32)
    if not torch.all(torch.isfinite(cube_position)) or not torch.all(
        torch.isfinite(goal_position)
    ):
        raise ValueError("evaluation geometry must be finite")

    cube = env.scene["cube"]
    cube_state = cube.data.default_root_state[env_ids].clone()
    cube_state[:, :3] = cube_position + env.scene.env_origins[env_ids]
    cube.write_root_pose_to_sim(cube_state[:, :7], env_ids)
    cube.write_root_velocity_to_sim(torch.zeros_like(cube_state[:, 7:]), env_ids)

    goal_marker = env.scene["goal_marker"]
    marker_state = goal_marker.data.default_root_state[env_ids].clone()
    marker_position = goal_position.clone()
    marker_position[:, 2] = GOAL_MARKER_SIZE_M[2] / 2.0
    marker_state[:, :3] = marker_position + env.scene.env_origins[env_ids]
    goal_marker.write_root_pose_to_sim(marker_state[:, :7], env_ids)
    goal_marker.write_root_velocity_to_sim(torch.zeros_like(marker_state[:, 7:]), env_ids)

    state = _runtime_state(env)
    state.goal_position_env_m[env_ids] = goal_position
    state.phase[env_ids] = int(PickPlaceRlPhase.REACH)
    state.last_phase_update_step[env_ids] = -1


__all__ = [
    "Stage9RlRuntimeState",
    "action_magnitude_penalty",
    "action_rate_penalty",
    "bilateral_cube_contact",
    "failure_penalty",
    "grasp_reward",
    "gripper_toggle_penalty",
    "invalid_state",
    "lift_reward",
    "place_reward",
    "reach_reward",
    "reset_franka_pick_place_rl",
    "runtime_goal_position",
    "set_franka_pick_place_rl_geometry",
    "state_observation",
    "success_bonus",
    "task_failure_dynamic",
    "task_success_dynamic_goal",
    "termination_diagnostics",
]
