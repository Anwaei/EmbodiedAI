"""Isaac Lab configuration for the Franka pick-and-place task skeleton."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from embodied_ai.contracts.tasks.franka_pick_place import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONTROL_HZ,
    CUBE_RESET_POSITION_ENV_M,
    GOAL_MARKER_SIZE_M,
    GOAL_POSITION_ENV_M,
    IK_ROTATION_SCALE_RAD,
    IK_TRANSLATION_SCALE_M,
    FrankaPickPlaceEpisodeParameters,
)

from . import evaluation as task_evaluation
from . import mdp as task_mdp

ROBOT_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]

CONTRACT_OBSERVATION_TERM_MAP = {
    "robot.joint_position": "joint_position",
    "robot.joint_velocity": "joint_velocity",
    "object.cube.position": "cube_position",
    "camera.front.rgb": "camera_front_rgb",
}


@configclass
class FrankaPickPlaceSceneCfg(InteractiveSceneCfg):
    """Franka, table, cube, ground, lighting, and a fixed front RGB camera."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, -0.025)),
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 0.8, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.42, 0.30, 0.20)
            ),
        ),
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_RESET_POSITION_ENV_M),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.45, 0.80)
            ),
        ),
    )

    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                GOAL_POSITION_ENV_M[0],
                GOAL_POSITION_ENV_M[1],
                GOAL_MARKER_SIZE_M[2] / 2.0,
            )
        ),
        spawn=sim_utils.CuboidCfg(
            size=GOAL_MARKER_SIZE_M,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.65, 0.20)
            ),
        ),
    )

    camera_front = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraFront",
        update_period=1.0 / CONTROL_HZ,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.4, 0.0, 1.0),
            rot=(0.35355, -0.61237, -0.61237, 0.35355),
            convention="ros",
        ),
    )

    # This expert-only sensor exposes the same tool-center frame controlled by the IK action.
    # It is intentionally not part of the public observation contract.
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="tool_center",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.107)),
            )
        ],
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Normalized 6D relative IK command followed by one binary gripper command."""

    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint[1-7]"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=(
            IK_TRANSLATION_SCALE_M,
            IK_TRANSLATION_SCALE_M,
            IK_TRANSLATION_SCALE_M,
            IK_ROTATION_SCALE_RAD,
            IK_ROTATION_SCALE_RAD,
            IK_ROTATION_SCALE_RAD,
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.107)
        ),
    )
    gripper = isaac_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Unconcatenated terms that map one-to-one to the observation contract."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_position = ObsTerm(
            func=isaac_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ROBOT_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )
        joint_velocity = ObsTerm(
            func=isaac_mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ROBOT_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )
        cube_position = ObsTerm(
            func=task_mdp.cube_position_env,
            params={"asset_cfg": SceneEntityCfg("cube")},
        )
        camera_front_rgb = ObsTerm(
            func=task_mdp.camera_rgb_chw,
            params={"sensor_cfg": SceneEntityCfg("camera_front")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Task-specific reset with no random sampling."""

    reset_scene = EventTerm(func=task_mdp.reset_franka_pick_place, mode="reset")


@configclass
class RewardsCfg:
    """Placeholder reward so this skeleton can use ManagerBasedRLEnv."""

    placeholder = RewTerm(func=task_mdp.zero_reward, weight=1.0)


@configclass
class TerminationsCfg:
    """Success, bounded-workspace failure, and time-limit terminations."""

    success = DoneTerm(
        func=task_evaluation.task_success,
        params={"goal_position_env_m": GOAL_POSITION_ENV_M},
    )
    failure = DoneTerm(func=task_evaluation.task_failure)
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class FrankaPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Deterministic task baseline with an external expert-compatible interface."""

    scene: FrankaPickPlaceSceneCfg = FrankaPickPlaceSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = 6
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 2
        self.viewer.eye = (1.4, 1.2, 1.0)
        self.viewer.lookat = (0.5, 0.0, 0.15)


def apply_episode_parameters(
    env_cfg: FrankaPickPlaceEnvCfg,
    parameters: FrankaPickPlaceEpisodeParameters,
) -> None:
    """Bind one reviewed episode specification before the Isaac environment is created."""

    env_cfg.scene.cube.init_state.pos = parameters.cube_reset_position_env_m
    goal = parameters.goal_position_env_m
    # The semantic goal uses the cube-centre height, while the thin marker rests on the table.
    env_cfg.scene.goal_marker.init_state.pos = (
        goal[0],
        goal[1],
        GOAL_MARKER_SIZE_M[2] / 2.0,
    )
    env_cfg.terminations.success.params = {"goal_position_env_m": goal}
