"""Shared observation and action contracts for Franka pick-and-place."""

from __future__ import annotations

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

# Normalized action values are scaled by the Isaac action adapter before IK.
IK_TRANSLATION_SCALE_M = 0.05
IK_ROTATION_SCALE_RAD = 0.15
GRIPPER_OPEN_ACTION = 1.0
GRIPPER_CLOSE_ACTION = -1.0

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
