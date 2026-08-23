"""Versioned episode metadata that has no simulator or ML dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._validation import (
    require_identifier,
    require_mapping,
    require_non_empty,
    require_non_negative_int,
    require_positive_int,
    require_relative_posix_path,
    require_schema_version,
    require_sequence,
    require_sha256,
    require_tuple,
)
from .action import ActionSchema
from .observation import ObservationSchema

EPISODE_SCHEMA_VERSION = "embodied-ai.episode/v1"


class EpisodeOutcome(StrEnum):
    """Terminal classifications kept distinct for evaluation."""

    SUCCESS = "success"
    FAILURE = "failure"
    TRUNCATED = "truncated"


class TimeBase(StrEnum):
    """Clock used by observation and action timestamp arrays."""

    SIMULATION = "simulation"


@dataclass(frozen=True, slots=True)
class PayloadFile:
    """Integrity metadata for one immutable episode payload."""

    path: str
    media_type: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        require_relative_posix_path(self.path, "payload path")
        require_non_empty(self.media_type, "payload media_type")
        require_non_negative_int(self.byte_size, "payload byte_size")
        require_sha256(self.sha256, "payload sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "media_type": self.media_type,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PayloadFile:
        source = require_mapping(data, "payload file")
        return cls(
            path=source.get("path"),
            media_type=source.get("media_type"),
            byte_size=source.get("byte_size"),
            sha256=source.get("sha256"),
        )


@dataclass(frozen=True, slots=True)
class EpisodeProvenance:
    """Revisions needed to reproduce an episode."""

    simulator_name: str
    simulator_version: str
    repository_revision: str
    configuration_revision: str
    environment_lock_sha256: str

    def __post_init__(self) -> None:
        require_non_empty(self.simulator_name, "simulator_name")
        require_non_empty(self.simulator_version, "simulator_version")
        require_non_empty(self.repository_revision, "repository_revision")
        require_non_empty(self.configuration_revision, "configuration_revision")
        require_sha256(self.environment_lock_sha256, "environment_lock_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "simulator_name": self.simulator_name,
            "simulator_version": self.simulator_version,
            "repository_revision": self.repository_revision,
            "configuration_revision": self.configuration_revision,
            "environment_lock_sha256": self.environment_lock_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EpisodeProvenance:
        source = require_mapping(data, "episode provenance")
        return cls(
            simulator_name=source.get("simulator_name"),
            simulator_version=source.get("simulator_version"),
            repository_revision=source.get("repository_revision"),
            configuration_revision=source.get("configuration_revision"),
            environment_lock_sha256=source.get("environment_lock_sha256"),
        )


@dataclass(frozen=True, slots=True)
class EpisodeMetadata:
    """Self-describing manifest for one finalized immutable episode."""

    episode_id: str
    task: str
    robot: str
    scene: str
    random_seed: int
    observation_schema: ObservationSchema
    action_schema: ActionSchema
    step_count: int
    start_time_ns: int
    end_time_ns: int
    outcome: EpisodeOutcome
    termination_reason: str | None
    payloads: tuple[PayloadFile, ...]
    provenance: EpisodeProvenance
    time_base: TimeBase = TimeBase.SIMULATION
    timestamp_unit: str = "ns"
    schema_version: str = EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema_version(self.schema_version, EPISODE_SCHEMA_VERSION)
        require_identifier(self.episode_id, "episode_id")
        require_identifier(self.task, "task")
        require_identifier(self.robot, "robot")
        require_identifier(self.scene, "scene")
        require_non_negative_int(self.random_seed, "random_seed")
        if not isinstance(self.observation_schema, ObservationSchema):
            raise ValueError("observation_schema must be an ObservationSchema")
        if not isinstance(self.action_schema, ActionSchema):
            raise ValueError("action_schema must be an ActionSchema")
        require_positive_int(self.step_count, "step_count")
        start_time = require_non_negative_int(self.start_time_ns, "start_time_ns")
        end_time = require_non_negative_int(self.end_time_ns, "end_time_ns")
        if end_time < start_time:
            raise ValueError("end_time_ns must not precede start_time_ns")
        if not isinstance(self.outcome, EpisodeOutcome):
            raise ValueError("outcome must be an EpisodeOutcome")
        if self.outcome is EpisodeOutcome.SUCCESS:
            if self.termination_reason is not None:
                require_non_empty(self.termination_reason, "termination_reason")
        elif self.termination_reason is None:
            raise ValueError("failure and truncated episodes require a termination_reason")
        else:
            require_non_empty(self.termination_reason, "termination_reason")
        require_tuple(self.payloads, "episode payloads")
        if not self.payloads:
            raise ValueError("at least one episode payload is required")
        if not all(isinstance(payload, PayloadFile) for payload in self.payloads):
            raise ValueError("episode payloads must be PayloadFile objects")
        paths = [payload.path for payload in self.payloads]
        if len(set(paths)) != len(paths):
            raise ValueError("episode payload paths must be unique")
        if not isinstance(self.provenance, EpisodeProvenance):
            raise ValueError("provenance must be EpisodeProvenance")
        if self.time_base is not TimeBase.SIMULATION:
            raise ValueError("Stage 6 episodes must use the simulation time base")
        if self.timestamp_unit != "ns":
            raise ValueError("Stage 6 timestamps must use nanoseconds")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "task": self.task,
            "robot": self.robot,
            "scene": self.scene,
            "random_seed": self.random_seed,
            "observation_schema": self.observation_schema.to_dict(),
            "action_schema": self.action_schema.to_dict(),
            "step_count": self.step_count,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "time_base": self.time_base.value,
            "timestamp_unit": self.timestamp_unit,
            "outcome": self.outcome.value,
            "payloads": [payload.to_dict() for payload in self.payloads],
            "provenance": self.provenance.to_dict(),
        }
        if self.termination_reason is not None:
            result["termination_reason"] = self.termination_reason
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EpisodeMetadata:
        source = require_mapping(data, "episode metadata")
        version = source.get("schema_version")
        require_schema_version(version, EPISODE_SCHEMA_VERSION)
        payload_data = require_sequence(source.get("payloads"), "episode payloads")
        try:
            outcome = EpisodeOutcome(source.get("outcome"))
            time_base = TimeBase(source.get("time_base"))
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported episode outcome or time base") from error
        return cls(
            episode_id=source.get("episode_id"),
            task=source.get("task"),
            robot=source.get("robot"),
            scene=source.get("scene"),
            random_seed=source.get("random_seed"),
            observation_schema=ObservationSchema.from_dict(
                require_mapping(source.get("observation_schema"), "observation schema")
            ),
            action_schema=ActionSchema.from_dict(
                require_mapping(source.get("action_schema"), "action schema")
            ),
            step_count=source.get("step_count"),
            start_time_ns=source.get("start_time_ns"),
            end_time_ns=source.get("end_time_ns"),
            outcome=outcome,
            termination_reason=source.get("termination_reason"),
            payloads=tuple(
                PayloadFile.from_dict(require_mapping(item, "payload file"))
                for item in payload_data
            ),
            provenance=EpisodeProvenance.from_dict(
                require_mapping(source.get("provenance"), "episode provenance")
            ),
            time_base=time_base,
            timestamp_unit=source.get("timestamp_unit"),
            schema_version=version,
        )


# Backward-compatible name retained from the Stage 2 placeholder.
EpisodeManifest = EpisodeMetadata
