"""Shared observation and action contracts for Franka pick-and-place."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..action import ActionComponent, ActionRepresentation, ActionSchema
from ..observation import (
    DataType,
    ObservationComponent,
    ObservationField,
    ObservationKind,
    ObservationSchema,
)

CONTROL_HZ = 20.0
CAMERA_HEIGHT = 224
CAMERA_WIDTH = 224

TASK_NAME = "franka-pick-place"
ROBOT_NAME = "franka-panda"
SCENE_NAME = "table-cube-goal-v1"
DEFAULT_INSTRUCTION = "Pick up the cube and place it in the goal."
DEFAULT_INSTRUCTION_ID = "pick-place-cube-goal-en-001"
DEFAULT_INSTRUCTION_LANGUAGE = "en"

# Normalized action values are scaled by the Isaac action adapter before IK.
IK_TRANSLATION_SCALE_M = 0.05
IK_ROTATION_SCALE_RAD = 0.15
GRIPPER_OPEN_ACTION = 1.0
GRIPPER_CLOSE_ACTION = -1.0

# Default task geometry and evaluation values are expressed in each replicated environment
# frame. Collection may override the two positions per episode without changing the schemas.
CUBE_RESET_POSITION_ENV_M = (0.50, 0.0, 0.03)
GOAL_POSITION_ENV_M = (0.65, -0.20, 0.03)
GOAL_MARKER_SIZE_M = (0.11, 0.11, 0.004)
SUCCESS_POSITION_TOLERANCE_M = 0.05
SUCCESS_LINEAR_SPEED_TOLERANCE_M_S = 0.10
SUCCESS_GRIPPER_OPEN_POSITION_M = 0.03
FAILURE_X_BOUNDS_ENV_M = (0.10, 1.00)
FAILURE_Y_BOUNDS_ENV_M = (-0.50, 0.50)
FAILURE_MINIMUM_Z_ENV_M = -0.05

# Conservative centre bounds keep the 5 cm cube and 11 cm goal marker on the table.
CUBE_CENTER_X_BOUNDS_ENV_M = (0.225, 0.875)
CUBE_CENTER_Y_BOUNDS_ENV_M = (-0.375, 0.375)
GOAL_CENTER_X_BOUNDS_ENV_M = (0.255, 0.845)
GOAL_CENTER_Y_BOUNDS_ENV_M = (-0.345, 0.345)


def _position_tuple(
    value: tuple[float, float, float],
    name: str,
) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a three-value tuple")
    if any(isinstance(component, bool) or not isinstance(component, (int, float)) for component in value):
        raise ValueError(f"{name} components must be real numbers")
    converted = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in converted):
        raise ValueError(f"{name} components must be finite")
    return converted


def _require_in_bounds(value: float, bounds: tuple[float, float], name: str) -> None:
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name}={value:g} is outside reviewed bounds {bounds}")


@dataclass(frozen=True, slots=True)
class FrankaPickPlaceEpisodeParameters:
    """Reviewed per-episode scene parameters shared by reset, expert, and evaluation."""

    cube_reset_position_env_m: tuple[float, float, float] = CUBE_RESET_POSITION_ENV_M
    goal_position_env_m: tuple[float, float, float] = GOAL_POSITION_ENV_M

    def __post_init__(self) -> None:
        cube = _position_tuple(self.cube_reset_position_env_m, "cube_reset_position_env_m")
        goal = _position_tuple(self.goal_position_env_m, "goal_position_env_m")
        object.__setattr__(self, "cube_reset_position_env_m", cube)
        object.__setattr__(self, "goal_position_env_m", goal)

        _require_in_bounds(cube[0], CUBE_CENTER_X_BOUNDS_ENV_M, "cube x")
        _require_in_bounds(cube[1], CUBE_CENTER_Y_BOUNDS_ENV_M, "cube y")
        _require_in_bounds(goal[0], GOAL_CENTER_X_BOUNDS_ENV_M, "goal x")
        _require_in_bounds(goal[1], GOAL_CENTER_Y_BOUNDS_ENV_M, "goal y")
        if not math.isclose(cube[2], CUBE_RESET_POSITION_ENV_M[2], abs_tol=1e-6):
            raise ValueError("cube z must keep the cube resting on the reviewed table surface")
        if not math.isclose(goal[2], GOAL_POSITION_ENV_M[2], abs_tol=1e-6):
            raise ValueError("goal z must represent the reviewed cube-centre placement height")
        distance = math.dist(cube, goal)
        if distance <= SUCCESS_POSITION_TOLERANCE_M:
            raise ValueError("cube reset position must begin outside the success tolerance")

    def task_parameters(self) -> dict[str, object]:
        return {"goal_position_env_m": list(self.goal_position_env_m)}

    def reset_parameters(self) -> dict[str, object]:
        return {"cube_position_env_m": list(self.cube_reset_position_env_m)}

_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")

FRANKA_PICK_PLACE_OBSERVATION_SCHEMA = ObservationSchema(
    fields=(
        ObservationField(
            key="robot.joint_position",
            kind=ObservationKind.STATE,
            shape=(9,),
            dtype=DataType.FLOAT32,
            axes=("component",),
            components=tuple(
                ObservationComponent(name=name, unit="rad") for name in _ARM_JOINT_NAMES
            )
            + tuple(
                ObservationComponent(name=name, unit="m") for name in _FINGER_JOINT_NAMES
            ),
        ),
        ObservationField(
            key="robot.joint_velocity",
            kind=ObservationKind.STATE,
            shape=(9,),
            dtype=DataType.FLOAT32,
            axes=("component",),
            components=tuple(
                ObservationComponent(name=name, unit="rad/s") for name in _ARM_JOINT_NAMES
            )
            + tuple(
                ObservationComponent(name=name, unit="m/s") for name in _FINGER_JOINT_NAMES
            ),
        ),
        ObservationField(
            key="object.cube.position",
            kind=ObservationKind.STATE,
            shape=(3,),
            dtype=DataType.FLOAT32,
            axes=("component",),
            components=tuple(
                ObservationComponent(name=axis, unit="m", frame="env")
                for axis in ("x", "y", "z")
            ),
            frame="env",
        ),
        ObservationField(
            key="camera.front.rgb",
            kind=ObservationKind.RGB_IMAGE,
            shape=(3, CAMERA_HEIGHT, CAMERA_WIDTH),
            dtype=DataType.UINT8,
            axes=("channel", "height", "width"),
            frame="camera_front_optical",
        ),
    )
)

FRANKA_PICK_PLACE_ACTION_SCHEMA = ActionSchema(
    representation=ActionRepresentation.END_EFFECTOR_DELTA_POSE,
    components=tuple(
        ActionComponent(name=name, unit="1", lower=-1.0, upper=1.0)
        for name in (
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_roll",
            "delta_pitch",
            "delta_yaw",
            "gripper",
        )
    ),
    control_hz=CONTROL_HZ,
    frame="robot_base",
    normalized=True,
)
