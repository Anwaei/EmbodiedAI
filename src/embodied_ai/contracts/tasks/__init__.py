"""Dependency-light task-specific contract instances."""

from .franka_pick_place import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONTROL_HZ,
    FRANKA_PICK_PLACE_ACTION_SCHEMA,
    FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
)

__all__ = [
    "CAMERA_HEIGHT",
    "CAMERA_WIDTH",
    "CONTROL_HZ",
    "FRANKA_PICK_PLACE_ACTION_SCHEMA",
    "FRANKA_PICK_PLACE_OBSERVATION_SCHEMA",
]
