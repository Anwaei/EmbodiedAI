"""Dependency-light plan contract for reproducible expert collection batches."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ._validation import (
    require_identifier,
    require_mapping,
    require_non_empty,
    require_non_negative_int,
    require_positive_int,
    require_schema_version,
    require_sequence,
)
from .tasks.franka_pick_place import (
    TASK_NAME,
    FrankaPickPlaceEpisodeParameters,
)

EXPERT_COLLECTION_PLAN_SCHEMA_VERSION = "embodied-ai.expert-collection-plan/v1"


def _position(source: object, name: str) -> tuple[float, float, float]:
    values = require_sequence(source, name)
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{name} values must be real numbers")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _relative_config_path(value: object) -> str:
    path = require_non_empty(value, "expert_config")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or pure.suffix != ".toml":
        raise ValueError("expert_config must be a relative TOML path without '..'")
    return path


@dataclass(frozen=True, slots=True)
class ExpertCollectionEpisodeSpec:
    """Language and controlled scene variation for one immutable episode."""

    episode_id: str
    seed: int
    instruction: str
    instruction_id: str
    instruction_language: str
    parameters: FrankaPickPlaceEpisodeParameters

    def __post_init__(self) -> None:
        require_identifier(self.episode_id, "episode_id")
        require_non_negative_int(self.seed, "seed")
        require_non_empty(self.instruction, "instruction")
        require_identifier(self.instruction_id, "instruction_id")
        require_identifier(self.instruction_language, "instruction_language")
        if not isinstance(self.parameters, FrankaPickPlaceEpisodeParameters):
            raise ValueError("parameters must be FrankaPickPlaceEpisodeParameters")

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "instruction": self.instruction,
            "instruction_id": self.instruction_id,
            "instruction_language": self.instruction_language,
            "cube_reset_position_env_m": list(self.parameters.cube_reset_position_env_m),
            "goal_position_env_m": list(self.parameters.goal_position_env_m),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExpertCollectionEpisodeSpec:
        source = require_mapping(data, "collection episode")
        parameters = FrankaPickPlaceEpisodeParameters(
            cube_reset_position_env_m=_position(
                source.get("cube_reset_position_env_m"),
                "cube_reset_position_env_m",
            ),
            goal_position_env_m=_position(
                source.get("goal_position_env_m"),
                "goal_position_env_m",
            ),
        )
        return cls(
            episode_id=source.get("episode_id"),
            seed=source.get("seed"),
            instruction=source.get("instruction"),
            instruction_id=source.get("instruction_id"),
            instruction_language=source.get("instruction_language"),
            parameters=parameters,
        )


@dataclass(frozen=True, slots=True)
class ExpertCollectionPlan:
    """Fully reviewed first-version matrix consumed by the batch launcher."""

    collection_id: str
    task: str
    expert_identifier: str
    expert_config: str
    max_steps: int
    episodes: tuple[ExpertCollectionEpisodeSpec, ...]
    schema_version: str = EXPERT_COLLECTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema_version(self.schema_version, EXPERT_COLLECTION_PLAN_SCHEMA_VERSION)
        require_identifier(self.collection_id, "collection_id")
        require_identifier(self.task, "task")
        if self.task != TASK_NAME:
            raise ValueError(f"first-version collection supports only task={TASK_NAME!r}")
        require_identifier(self.expert_identifier, "expert_identifier")
        _relative_config_path(self.expert_config)
        require_positive_int(self.max_steps, "max_steps")
        if not isinstance(self.episodes, tuple) or not self.episodes:
            raise ValueError("collection plan must contain at least one episode")
        if not all(isinstance(item, ExpertCollectionEpisodeSpec) for item in self.episodes):
            raise ValueError("episodes must contain ExpertCollectionEpisodeSpec values")

        episode_ids = [item.episode_id for item in self.episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("collection episode IDs must be unique")
        seeds = [item.seed for item in self.episodes]
        if len(set(seeds)) != len(seeds):
            raise ValueError("first-version collection seeds must be unique")

        # A stable instruction ID must never silently refer to different language text.
        variants: dict[str, tuple[str, str]] = {}
        for item in self.episodes:
            value = (item.instruction, item.instruction_language)
            previous = variants.setdefault(item.instruction_id, value)
            if previous != value:
                raise ValueError("instruction_id maps to inconsistent text or language")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "collection_id": self.collection_id,
            "task": self.task,
            "expert_identifier": self.expert_identifier,
            "expert_config": self.expert_config,
            "max_steps": self.max_steps,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExpertCollectionPlan:
        source = require_mapping(data, "expert collection plan")
        episodes = require_sequence(source.get("episodes"), "collection episodes")
        return cls(
            collection_id=source.get("collection_id"),
            task=source.get("task"),
            expert_identifier=source.get("expert_identifier"),
            expert_config=_relative_config_path(source.get("expert_config")),
            max_steps=source.get("max_steps"),
            episodes=tuple(
                ExpertCollectionEpisodeSpec.from_dict(
                    require_mapping(item, "collection episode")
                )
                for item in episodes
            ),
            schema_version=source.get("schema_version"),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> ExpertCollectionPlan:
        with Path(path).expanduser().resolve().open("rb") as stream:
            return cls.from_dict(tomllib.load(stream))


__all__ = [
    "EXPERT_COLLECTION_PLAN_SCHEMA_VERSION",
    "ExpertCollectionEpisodeSpec",
    "ExpertCollectionPlan",
]
