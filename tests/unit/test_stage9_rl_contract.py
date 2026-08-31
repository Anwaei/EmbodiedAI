"""Dependency-light tests for the frozen Stage 9 PPO profiles."""

import json

from embodied_ai.contracts.rl import (
    RSL_RL_PPO_BACKEND,
    STANDALONE_PPO_ACTION_PROFILE,
    STANDALONE_PPO_OBSERVATION_PROFILE,
    STANDALONE_PPO_REWARD_PROFILE_ID,
    STANDALONE_PPO_TASK_ID,
    RlMode,
    Stage9RunIdentity,
)


def test_standalone_observation_profile_is_fixed_52d_state_only() -> None:
    profile = STANDALONE_PPO_OBSERVATION_PROFILE

    assert profile.dimension == 52
    assert tuple(component.dimension for component in profile.components) == (
        9,
        9,
        3,
        4,
        3,
        3,
        3,
        3,
        3,
        7,
        5,
    )
    assert profile.excluded_modalities == (
        "camera.front.rgb",
        "language.instruction",
    )
    assert {component.name for component in profile.components if component.privileged} == {
        "cube.position",
        "cube.linear-velocity",
        "goal.position",
        "tool-to-cube.position",
        "cube-to-goal.position",
        "task-phase.one-hot",
    }


def test_standalone_action_and_backend_match_reviewed_boundary() -> None:
    action = STANDALONE_PPO_ACTION_PROFILE

    assert (action.action_dimension, action.arm_dimension, action.gripper_dimension) == (7, 6, 1)
    assert action.control_hz == 20.0
    assert action.translation_scale_m == 0.05
    assert action.rotation_scale_rad == 0.15
    assert action.gripper_threshold == 0.0
    assert RSL_RL_PPO_BACKEND.package_version == "3.1.2"
    assert RSL_RL_PPO_BACKEND.isaaclab_rl_version == "0.4.7"


def test_run_identity_is_json_serializable_and_binds_profiles() -> None:
    identity = Stage9RunIdentity(
        run_id="stage9-contract-test",
        mode=RlMode.STANDALONE_PPO,
        task_id=STANDALONE_PPO_TASK_ID,
        observation_profile_id=STANDALONE_PPO_OBSERVATION_PROFILE.profile_id,
        action_profile_id=STANDALONE_PPO_ACTION_PROFILE.profile_id,
        reward_profile_id=STANDALONE_PPO_REWARD_PROFILE_ID,
        backend=RSL_RL_PPO_BACKEND,
        seed=7,
    )

    encoded = json.loads(json.dumps(identity.to_dict()))
    assert encoded["mode"] == "standalone_ppo"
    assert encoded["backend"]["package"] == "rsl-rl-lib"
