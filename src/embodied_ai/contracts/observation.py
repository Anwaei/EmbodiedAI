"""Versioned observation schema shared across isolated runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._validation import (
    require_identifier,
    require_mapping,
    require_non_empty,
    require_positive_int,
    require_schema_version,
    require_sequence,
    require_tuple,
)

OBSERVATION_SCHEMA_VERSION = "embodied-ai.observation/v1"


class DataType(StrEnum):
    """Portable scalar types allowed in contract payloads."""

    BOOL = "bool"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT64 = "int64"
    UINT8 = "uint8"


class ObservationKind(StrEnum):
    """Supported observation families for the Stage 6 baseline."""

    STATE = "state"
    RGB_IMAGE = "rgb_image"


@dataclass(frozen=True, slots=True)
class ObservationComponent:
    """Semantic description of one scalar in a state vector."""

    name: str
    unit: str
    frame: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.name, "observation component name")
        require_non_empty(self.unit, "observation component unit")
        if self.frame is not None:
            require_identifier(self.frame, "observation component frame")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"name": self.name, "unit": self.unit}
        if self.frame is not None:
            result["frame"] = self.frame
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservationComponent:
        source = require_mapping(data, "observation component")
        return cls(
            name=source.get("name"),
            unit=source.get("unit"),
            frame=source.get("frame"),
        )


@dataclass(frozen=True, slots=True)
class ObservationField:
    """Shape, storage type, axes and semantics for one observation stream."""

    key: str
    kind: ObservationKind
    shape: tuple[int, ...]
    dtype: DataType
    axes: tuple[str, ...]
    components: tuple[ObservationComponent, ...] = ()
    frame: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.key, "observation key")
        if not isinstance(self.kind, ObservationKind):
            raise ValueError("observation kind must be an ObservationKind")
        require_tuple(self.shape, "observation shape")
        require_tuple(self.axes, "observation axes")
        require_tuple(self.components, "observation components")
        if not self.shape:
            raise ValueError("observation shape must not be empty")
        for index, dimension in enumerate(self.shape):
            require_positive_int(dimension, f"observation shape[{index}]")
        if not isinstance(self.dtype, DataType):
            raise ValueError("observation dtype must be a DataType")
        if len(self.axes) != len(self.shape):
            raise ValueError("observation axes must match the shape rank")
        for axis in self.axes:
            require_identifier(axis, "observation axis")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("observation axes must be unique")
        if self.frame is not None:
            require_identifier(self.frame, "observation frame")

        if not all(isinstance(component, ObservationComponent) for component in self.components):
            raise ValueError("observation components must be ObservationComponent objects")
        component_names = [component.name for component in self.components]
        if len(set(component_names)) != len(component_names):
            raise ValueError("observation component names must be unique")

        if self.kind is ObservationKind.STATE:
            if len(self.shape) != 1 or self.axes != ("component",):
                raise ValueError("state observations must be rank-1 on the 'component' axis")
            if len(self.components) != self.shape[0]:
                raise ValueError("state observations require one component per scalar")
        elif self.kind is ObservationKind.RGB_IMAGE:
            if self.shape[0] != 3 or self.axes != ("channel", "height", "width"):
                raise ValueError("RGB observations must use a 3-channel CHW layout")
            if self.dtype is not DataType.UINT8:
                raise ValueError("RGB observations must use uint8 storage")
            if self.components:
                raise ValueError("RGB observations must not define scalar components")
            if self.frame is None:
                raise ValueError("RGB observations require an explicit camera frame")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "key": self.key,
            "kind": self.kind.value,
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "axes": list(self.axes),
            "components": [component.to_dict() for component in self.components],
        }
        if self.frame is not None:
            result["frame"] = self.frame
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservationField:
        source = require_mapping(data, "observation field")
        shape = require_sequence(source.get("shape"), "observation shape")
        axes = require_sequence(source.get("axes"), "observation axes")
        components = require_sequence(
            source.get("components", []), "observation components"
        )
        try:
            kind = ObservationKind(source.get("kind"))
            dtype = DataType(source.get("dtype"))
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported observation kind or dtype") from error
        return cls(
            key=source.get("key"),
            kind=kind,
            shape=tuple(shape),
            dtype=dtype,
            axes=tuple(axes),
            components=tuple(
                ObservationComponent.from_dict(require_mapping(item, "observation component"))
                for item in components
            ),
            frame=source.get("frame"),
        )


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    """Complete observation interface for one episode."""

    fields: tuple[ObservationField, ...]
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema_version(self.schema_version, OBSERVATION_SCHEMA_VERSION)
        require_tuple(self.fields, "observation fields")
        if not self.fields:
            raise ValueError("at least one observation field is required")
        if not all(isinstance(field, ObservationField) for field in self.fields):
            raise ValueError("observation fields must be ObservationField objects")
        keys = [field.key for field in self.fields]
        if len(set(keys)) != len(keys):
            raise ValueError("observation field keys must be unique")

    def field(self, key: str) -> ObservationField:
        """Return a field by key, raising KeyError when the key is absent."""

        for field in self.fields:
            if field.key == key:
                return field
        raise KeyError(key)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fields": [field.to_dict() for field in self.fields],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservationSchema:
        source = require_mapping(data, "observation schema")
        version = source.get("schema_version")
        require_schema_version(version, OBSERVATION_SCHEMA_VERSION)
        fields = require_sequence(source.get("fields"), "observation fields")
        return cls(
            fields=tuple(
                ObservationField.from_dict(require_mapping(item, "observation field"))
                for item in fields
            ),
            schema_version=version,
        )
