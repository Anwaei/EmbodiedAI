"""Versioned action schema shared across isolated runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._validation import (
    require_finite,
    require_identifier,
    require_mapping,
    require_non_empty,
    require_positive_finite,
    require_schema_version,
    require_sequence,
    require_tuple,
)
from .observation import DataType

ACTION_SCHEMA_VERSION = "embodied-ai.action/v1"


class ActionRepresentation(StrEnum):
    """Supported command representations for the Stage 6 baseline."""

    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    END_EFFECTOR_DELTA_POSE = "end_effector_delta_pose"


@dataclass(frozen=True, slots=True)
class ActionComponent:
    """Semantics and valid physical range for one action scalar."""

    name: str
    unit: str
    lower: float
    upper: float
    frame: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.name, "action component name")
        require_non_empty(self.unit, "action component unit")
        lower = require_finite(self.lower, "action component lower bound")
        upper = require_finite(self.upper, "action component upper bound")
        if lower >= upper:
            raise ValueError("action component lower bound must be less than upper bound")
        if self.frame is not None:
            require_identifier(self.frame, "action component frame")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "unit": self.unit,
            "lower": self.lower,
            "upper": self.upper,
        }
        if self.frame is not None:
            result["frame"] = self.frame
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActionComponent:
        source = require_mapping(data, "action component")
        return cls(
            name=source.get("name"),
            unit=source.get("unit"),
            lower=source.get("lower"),
            upper=source.get("upper"),
            frame=source.get("frame"),
        )


@dataclass(frozen=True, slots=True)
class ActionSchema:
    """Action representation, rate and component ordering for one episode."""

    representation: ActionRepresentation
    components: tuple[ActionComponent, ...]
    control_hz: float
    dtype: DataType = DataType.FLOAT32
    frame: str | None = None
    normalized: bool = False
    schema_version: str = ACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema_version(self.schema_version, ACTION_SCHEMA_VERSION)
        if not isinstance(self.representation, ActionRepresentation):
            raise ValueError("action representation must be an ActionRepresentation")
        require_tuple(self.components, "action components")
        if not self.components:
            raise ValueError("at least one action component is required")
        if not all(isinstance(component, ActionComponent) for component in self.components):
            raise ValueError("action components must be ActionComponent objects")
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise ValueError("action component names must be unique")
        require_positive_finite(self.control_hz, "control_hz")
        if not isinstance(self.dtype, DataType) or self.dtype not in (
            DataType.FLOAT32,
            DataType.FLOAT64,
        ):
            raise ValueError("actions must use float32 or float64 storage")
        if self.frame is not None:
            require_identifier(self.frame, "action frame")
        if (
            self.representation is ActionRepresentation.END_EFFECTOR_DELTA_POSE
            and self.frame is None
        ):
            raise ValueError("end-effector delta actions require an explicit frame")
        if not isinstance(self.normalized, bool):
            raise ValueError("normalized must be a boolean")
        if self.normalized:
            for component in self.components:
                if component.unit != "1" or component.lower != -1.0 or component.upper != 1.0:
                    raise ValueError(
                        "normalized action components must use unit '1' and bounds [-1, 1]"
                    )

    @property
    def dimension(self) -> int:
        return len(self.components)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "representation": self.representation.value,
            "components": [component.to_dict() for component in self.components],
            "dimension": self.dimension,
            "control_hz": self.control_hz,
            "dtype": self.dtype.value,
            "normalized": self.normalized,
        }
        if self.frame is not None:
            result["frame"] = self.frame
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActionSchema:
        source = require_mapping(data, "action schema")
        version = source.get("schema_version")
        require_schema_version(version, ACTION_SCHEMA_VERSION)
        components_data = require_sequence(source.get("components"), "action components")
        try:
            representation = ActionRepresentation(source.get("representation"))
            dtype = DataType(source.get("dtype"))
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported action representation or dtype") from error
        components = tuple(
            ActionComponent.from_dict(require_mapping(item, "action component"))
            for item in components_data
        )
        declared_dimension = source.get("dimension", len(components))
        if declared_dimension != len(components):
            raise ValueError("declared action dimension does not match components")
        return cls(
            representation=representation,
            components=components,
            control_hz=source.get("control_hz"),
            dtype=dtype,
            frame=source.get("frame"),
            normalized=source.get("normalized", False),
            schema_version=version,
        )
