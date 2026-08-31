"""Stage 9 state-based standalone PPO environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from embodied_ai.contracts.rl import STANDALONE_PPO_ACTION_PROFILE
from embodied_ai.contracts.tasks.franka_pick_place import GOAL_MARKER_SIZE_M
from embodied_ai.rl.config import Stage9StandalonePpoConfig, reviewed_stage9_config_path

from . import rl_mdp
from .env_cfg import ActionsCfg, FrankaPickPlaceEnvCfg, FrankaPickPlaceSceneCfg

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_DATA_ROOT = Path(os.environ.get("EMBODIEDAI_DATA", "/root/autodl-tmp/EmbodiedAI"))
STAGE9_STANDALONE_CONFIG = Stage9StandalonePpoConfig.from_toml(
    reviewed_stage9_config_path(_REPOSITORY_ROOT),
    repository_root=_REPOSITORY_ROOT,
    data_root=_DATA_ROOT,
)

_RL_FRANKA_CFG = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_RL_FRANKA_CFG.spawn.activate_contact_sensors = True


@configclass
class FrankaPickPlacePPOSceneCfg(FrankaPickPlaceSceneCfg):
    """State-only training scene with finger contact sensing and movable goals."""

    robot = _RL_FRANKA_CFG
    camera_front = None
    goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.65, -0.20, GOAL_MARKER_SIZE_M[2] / 2.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=GOAL_MARKER_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20)),
        ),
    )
    left_finger_cube_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    right_finger_cube_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )


_PHASE_PARAMS = {
    "reach_distance_m": STAGE9_STANDALONE_CONFIG.phase.reach_distance_m,
    "grasp_force_n": STAGE9_STANDALONE_CONFIG.phase.grasp_force_n,
    "lift_height_m": STAGE9_STANDALONE_CONFIG.phase.lift_height_m,
    "place_distance_m": STAGE9_STANDALONE_CONFIG.phase.place_distance_m,
}


@configclass
class FrankaPickPlacePPOObservationsCfg:
    """One concatenated 52D state-only observation for actor and critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(func=rl_mdp.state_observation, params=dict(_PHASE_PARAMS))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaPickPlacePPOEventCfg:
    """Independent vectorized cube/goal reset sampling."""

    reset_scene = EventTerm(
        func=rl_mdp.reset_franka_pick_place_rl,
        mode="reset",
        params={
            "cube_x_m": STAGE9_STANDALONE_CONFIG.reset.cube_x_m,
            "cube_y_m": STAGE9_STANDALONE_CONFIG.reset.cube_y_m,
            "goal_x_m": STAGE9_STANDALONE_CONFIG.reset.goal_x_m,
            "goal_y_m": STAGE9_STANDALONE_CONFIG.reset.goal_y_m,
            "cube_z_m": STAGE9_STANDALONE_CONFIG.reset.cube_z_m,
            "goal_z_m": STAGE9_STANDALONE_CONFIG.reset.goal_z_m,
        },
    )


@configclass
class FrankaPickPlacePPORewardsCfg:
    """Reviewed staged task reward plus smoothness penalties."""

    reach = RewTerm(
        func=rl_mdp.reach_reward,
        weight=STAGE9_STANDALONE_CONFIG.reward.reach_weight,
        params={
            **_PHASE_PARAMS,
            "std_m": STAGE9_STANDALONE_CONFIG.reward.reach_std_m,
        },
    )
    grasp = RewTerm(
        func=rl_mdp.grasp_reward,
        weight=STAGE9_STANDALONE_CONFIG.reward.grasp_weight,
        params=dict(_PHASE_PARAMS),
    )
    lift = RewTerm(
        func=rl_mdp.lift_reward,
        weight=STAGE9_STANDALONE_CONFIG.reward.lift_weight,
        params={
            **_PHASE_PARAMS,
            "resting_height_m": STAGE9_STANDALONE_CONFIG.reset.cube_z_m,
            "target_height_m": STAGE9_STANDALONE_CONFIG.reward.lift_target_height_m,
        },
    )
    place = RewTerm(
        func=rl_mdp.place_reward,
        weight=STAGE9_STANDALONE_CONFIG.reward.place_weight,
        params={
            **_PHASE_PARAMS,
            "std_m": STAGE9_STANDALONE_CONFIG.reward.place_std_m,
        },
    )
    success = RewTerm(
        func=rl_mdp.success_bonus,
        weight=STAGE9_STANDALONE_CONFIG.reward.success_weight,
    )
    failure = RewTerm(
        func=rl_mdp.failure_penalty,
        weight=-STAGE9_STANDALONE_CONFIG.reward.failure_weight,
    )
    action_magnitude = RewTerm(
        func=rl_mdp.action_magnitude_penalty,
        weight=-STAGE9_STANDALONE_CONFIG.reward.action_magnitude_weight,
    )
    action_rate = RewTerm(
        func=rl_mdp.action_rate_penalty,
        weight=-STAGE9_STANDALONE_CONFIG.reward.action_rate_weight,
    )
    gripper_toggle = RewTerm(
        func=rl_mdp.gripper_toggle_penalty,
        weight=-STAGE9_STANDALONE_CONFIG.reward.gripper_toggle_weight,
    )


@configclass
class FrankaPickPlacePPOTerminationsCfg:
    """Public success/failure plus invalid-state and timeout diagnostics."""

    success = DoneTerm(func=rl_mdp.task_success_dynamic_goal)
    failure = DoneTerm(func=rl_mdp.task_failure_dynamic)
    invalid_state = DoneTerm(func=rl_mdp.invalid_state)
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class FrankaPickPlacePPOEnvCfg(FrankaPickPlaceEnvCfg):
    """Vectorized state-based PPO task that leaves the Stage 6 task unchanged."""

    scene: FrankaPickPlacePPOSceneCfg = FrankaPickPlacePPOSceneCfg(
        num_envs=STAGE9_STANDALONE_CONFIG.num_envs,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: FrankaPickPlacePPOObservationsCfg = FrankaPickPlacePPOObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: FrankaPickPlacePPOEventCfg = FrankaPickPlacePPOEventCfg()
    rewards: FrankaPickPlacePPORewardsCfg = FrankaPickPlacePPORewardsCfg()
    terminations: FrankaPickPlacePPOTerminationsCfg = FrankaPickPlacePPOTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = (
            STAGE9_STANDALONE_CONFIG.max_episode_steps
            / STANDALONE_PPO_ACTION_PROFILE.control_hz
        )
        self.num_rerenders_on_reset = 0


__all__ = [
    "STAGE9_STANDALONE_CONFIG",
    "FrankaPickPlacePPOEnvCfg",
    "FrankaPickPlacePPOEventCfg",
    "FrankaPickPlacePPOObservationsCfg",
    "FrankaPickPlacePPORewardsCfg",
    "FrankaPickPlacePPOSceneCfg",
    "FrankaPickPlacePPOTerminationsCfg",
]
