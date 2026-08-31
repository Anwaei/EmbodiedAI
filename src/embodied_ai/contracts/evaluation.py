"""Dependency-light contracts for deterministic Stage 8 evaluations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .episode import PayloadFile
from .tasks.franka_pick_place import FrankaPickPlaceEpisodeParameters

STAGE8_SCENARIOS_SCHEMA_VERSION = "embodied-ai.stage8-scenarios/v1"
STAGE8_ROLLOUT_SCHEMA_VERSION = "embodied-ai.stage8-rollout/v1"


class PolicyKind(StrEnum):
    SCRIPTED_EXPERT = "scripted_expert"
    BASE = "base"
    PEFT_ADAPTER = "peft_adapter"


class RolloutOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    TRUNCATED = "truncated"
    ERROR = "error"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _position(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} must have three numeric values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{label} must contain finite values")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class EvaluationScenario:
    """One held-out geometry/instruction case evaluated with several noise seeds."""

    scenario_id: str
    source_episode_index: int
    simulation_seed: int
    instruction: str
    cube_position_env_m: tuple[float, float, float]
    goal_position_env_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        _text(self.scenario_id, "scenario_id")
        _integer(self.source_episode_index, "source_episode_index")
        _integer(self.simulation_seed, "simulation_seed")
        _text(self.instruction, "instruction")
        cube = _position(self.cube_position_env_m, "cube_position_env_m")
        goal = _position(self.goal_position_env_m, "goal_position_env_m")
        # Reuse the task contract so scenarios cannot bypass reviewed table bounds.
        FrankaPickPlaceEpisodeParameters(cube, goal)
        object.__setattr__(self, "cube_position_env_m", cube)
        object.__setattr__(self, "goal_position_env_m", goal)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "source_episode_index": self.source_episode_index,
            "simulation_seed": self.simulation_seed,
            "instruction": self.instruction,
            "cube_position_env_m": list(self.cube_position_env_m),
            "goal_position_env_m": list(self.goal_position_env_m),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationScenario:
        return cls(
            scenario_id=value.get("scenario_id"),
            source_episode_index=value.get("source_episode_index"),
            simulation_seed=value.get("simulator_seed"),
            instruction=value.get("instruction"),
            cube_position_env_m=_position(
                value.get("cube_reset_position_env_m"), "cube_reset_position_env_m"
            ),
            goal_position_env_m=_position(value.get("goal_position_env_m"), "goal_position_env_m"),
        )


@dataclass(frozen=True, slots=True)
class PolicyDescriptor:
    kind: PolicyKind
    identifier: str
    revision: str
    identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PolicyKind):
            raise ValueError("kind must be a PolicyKind")
        _text(self.identifier, "identifier")
        _text(self.revision, "revision")
        if len(self.identity_sha256) != 64:
            raise ValueError("identity_sha256 must be a SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "identifier": self.identifier,
            "revision": self.revision,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class EvaluationRolloutMetadata:
    """Immutable manifest fields for one policy/scenario/noise rollout."""

    rollout_id: str
    run_id: str
    scenario: EvaluationScenario
    policy: PolicyDescriptor
    noise_seed: int
    prediction_horizon: int
    execute_horizon: int
    step_count: int
    outcome: RolloutOutcome
    termination_reason: str
    metrics: Mapping[str, float | int]
    provenance: Mapping[str, object]
    payloads: tuple[PayloadFile, ...]
    schema_version: str = STAGE8_ROLLOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE8_ROLLOUT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 8 rollout schema")
        _text(self.rollout_id, "rollout_id")
        _text(self.run_id, "run_id")
        if not isinstance(self.scenario, EvaluationScenario):
            raise ValueError("scenario must be an EvaluationScenario")
        if not isinstance(self.policy, PolicyDescriptor):
            raise ValueError("policy must be a PolicyDescriptor")
        _integer(self.noise_seed, "noise_seed")
        if self.prediction_horizon != 50 or self.execute_horizon != 5:
            raise ValueError("Stage 8 requires prediction_horizon=50 and execute_horizon=5")
        _integer(self.step_count, "step_count", minimum=1)
        if not isinstance(self.outcome, RolloutOutcome):
            raise ValueError("outcome must be a RolloutOutcome")
        _text(self.termination_reason, "termination_reason")
        if not self.payloads:
            raise ValueError("rollout requires payload integrity records")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rollout_id": self.rollout_id,
            "run_id": self.run_id,
            "scenario": self.scenario.to_dict(),
            "policy": self.policy.to_dict(),
            "noise_seed": self.noise_seed,
            "prediction_horizon": self.prediction_horizon,
            "execute_horizon": self.execute_horizon,
            "step_count": self.step_count,
            "outcome": self.outcome.value,
            "termination_reason": self.termination_reason,
            "metrics": dict(self.metrics),
            "provenance": dict(self.provenance),
            "payloads": [payload.to_dict() for payload in self.payloads],
        }


__all__ = [
    "EvaluationRolloutMetadata",
    "EvaluationScenario",
    "PolicyDescriptor",
    "PolicyKind",
    "RolloutOutcome",
    "STAGE8_ROLLOUT_SCHEMA_VERSION",
    "STAGE8_SCENARIOS_SCHEMA_VERSION",
]
