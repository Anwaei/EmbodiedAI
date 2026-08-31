"""Isaac-side adapters for the Stage 8 policy evaluation loop."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv

from embodied_ai.contracts.policy_rpc import LivePolicyObservation
from embodied_ai.contracts.tasks.franka_pick_place import (
    FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
)
from embodied_ai.sim.collection.expert_rollout import contract_observation
from embodied_ai.sim.tasks.franka_pick_place.env_cfg import CONTRACT_OBSERVATION_TERM_MAP


def extract_live_observation(
    observations: Mapping[str, object], *, env_index: int = 0
) -> tuple[LivePolicyObservation, dict[str, np.ndarray]]:
    """Adapt Isaac tensors to both the wire payload and recorder-ready contract values."""

    values = contract_observation(
        observations,
        schema=FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
        term_map=CONTRACT_OBSERVATION_TERM_MAP,
        env_index=env_index,
    )
    image = np.asarray(values["camera.front.rgb"], dtype=np.uint8)
    wire = LivePolicyObservation.from_rgb_bytes(
        np.asarray(values["robot.joint_position"], dtype=np.float32), image.tobytes()
    )
    return wire, values


def current_task_state(
    env: ManagerBasedEnv,
    *,
    goal_position_env_m: tuple[float, float, float],
) -> dict[str, object]:
    from embodied_ai.sim.tasks.franka_pick_place.evaluation import evaluate_pick_place

    result = evaluate_pick_place(env, goal_position_env_m=goal_position_env_m)
    return {
        "cube_position_env_m": result.cube_position_env_m[0].detach().cpu().numpy(),
        "goal_error_m": float(result.position_error_m[0]),
        "cube_speed_m_s": float(result.linear_speed_m_s[0]),
        "gripper_open": bool(result.gripper_open[0]),
        "success": bool(result.success[0]),
        "failure": bool(result.failure[0]),
    }


def step_environment(env: object, action: np.ndarray) -> Mapping[str, object]:
    tensor = torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device).reshape(1, 7)
    # no_grad avoids autograd overhead without turning Isaac's reusable state buffers into
    # immutable inference tensors, which would make the next episode reset fail.
    with torch.no_grad():
        observations, _, _, _, _ = env.step(tensor)
    return observations


__all__ = ["current_task_state", "extract_live_observation", "step_environment"]
