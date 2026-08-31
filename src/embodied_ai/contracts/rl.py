"""Dependency-light contracts for Stage 9 reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from ._validation import (
    require_identifier,
    require_non_empty,
    require_positive_finite,
    require_positive_int,
)
from .tasks.franka_pick_place import (
    CONTROL_HZ,
    IK_ROTATION_SCALE_RAD,
    IK_TRANSLATION_SCALE_M,
)

STAGE9_RUN_CONFIG_SCHEMA_VERSION = "embodied-ai.stage9-run-config/v1"
STAGE9_RUN_IDENTITY_SCHEMA_VERSION = "embodied-ai.stage9-run-identity/v1"
STANDALONE_PPO_TASK_ID = "EmbodiedAI-Franka-PickPlace-State-PPO-v0"


class RlMode(StrEnum):
    """Reviewed Stage 9 controller modes."""

    STANDALONE_PPO = "standalone_ppo"
    RESIDUAL_PPO = "residual_ppo"


class PickPlaceRlPhase(IntEnum):
    """Monotonic task phases used by reward gating and diagnostics."""

    REACH = 0
    GRASP = 1
    LIFT = 2
    PLACE = 3
    RELEASE = 4


@dataclass(frozen=True, slots=True)
class RlObservationComponent:
    """One fixed-order segment in a flat RL observation profile."""

    name: str
    dimension: int
    privileged: bool = False

    def __post_init__(self) -> None:
        require_identifier(self.name, "observation component name")
        require_positive_int(self.dimension, f"{self.name} dimension")


@dataclass(frozen=True, slots=True)
class RlObservationProfile:
    """Versioned flat observation layout presented to an RL actor."""

    profile_id: str
    components: tuple[RlObservationComponent, ...]
    excluded_modalities: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.profile_id, "observation profile_id")
        if not self.components:
            raise ValueError("RL observation profile requires components")
        names = tuple(component.name for component in self.components)
        if len(names) != len(set(names)):
            raise ValueError("RL observation component names must be unique")
        for modality in self.excluded_modalities:
            require_identifier(modality, "excluded modality")

    @property
    def dimension(self) -> int:
        return sum(component.dimension for component in self.components)

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "dimension": self.dimension,
            "components": [
                {
                    "name": component.name,
                    "dimension": component.dimension,
                    "privileged": component.privileged,
                }
                for component in self.components
            ],
            "excluded_modalities": list(self.excluded_modalities),
        }


@dataclass(frozen=True, slots=True)
class RlActionProfile:
    """Canonical standalone PPO action and physical adapter identity."""

    profile_id: str
    action_dimension: int
    arm_dimension: int
    gripper_dimension: int
    control_hz: float
    normalized_bound: float
    translation_scale_m: float
    rotation_scale_rad: float
    gripper_threshold: float

    def __post_init__(self) -> None:
        require_identifier(self.profile_id, "action profile_id")
        require_positive_int(self.action_dimension, "action_dimension")
        require_positive_int(self.arm_dimension, "arm_dimension")
        require_positive_int(self.gripper_dimension, "gripper_dimension")
        if self.arm_dimension + self.gripper_dimension != self.action_dimension:
            raise ValueError("arm and gripper dimensions must equal action_dimension")
        require_positive_finite(self.control_hz, "control_hz")
        require_positive_finite(self.normalized_bound, "normalized_bound")
        require_positive_finite(self.translation_scale_m, "translation_scale_m")
        require_positive_finite(self.rotation_scale_rad, "rotation_scale_rad")
        if not -self.normalized_bound < self.gripper_threshold < self.normalized_bound:
            raise ValueError("gripper_threshold must be strictly inside normalized bounds")

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "action_dimension": self.action_dimension,
            "arm_dimension": self.arm_dimension,
            "gripper_dimension": self.gripper_dimension,
            "control_hz": self.control_hz,
            "normalized_bound": self.normalized_bound,
            "translation_scale_m": self.translation_scale_m,
            "rotation_scale_rad": self.rotation_scale_rad,
            "gripper_threshold": self.gripper_threshold,
        }


@dataclass(frozen=True, slots=True)
class RlBackendIdentity:
    """Pinned PPO implementation expected from the Isaac lock."""

    backend_id: str
    algorithm: str
    package: str
    package_version: str
    isaaclab_rl_version: str

    def __post_init__(self) -> None:
        require_identifier(self.backend_id, "backend_id")
        require_identifier(self.algorithm, "algorithm")
        require_identifier(self.package, "backend package")
        require_non_empty(self.package_version, "backend package_version")
        require_non_empty(self.isaaclab_rl_version, "isaaclab_rl_version")

    def to_dict(self) -> dict[str, str]:
        return {
            "backend_id": self.backend_id,
            "algorithm": self.algorithm,
            "package": self.package,
            "package_version": self.package_version,
            "isaaclab_rl_version": self.isaaclab_rl_version,
        }


@dataclass(frozen=True, slots=True)
class Stage9RunIdentity:
    """Small immutable identity embedded in future Stage 9 run manifests."""

    run_id: str
    mode: RlMode
    task_id: str
    observation_profile_id: str
    action_profile_id: str
    reward_profile_id: str
    backend: RlBackendIdentity
    seed: int
    schema_version: str = STAGE9_RUN_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE9_RUN_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 9 run identity schema")
        require_identifier(self.run_id, "run_id")
        if not isinstance(self.mode, RlMode):
            raise ValueError("mode must be an RlMode")
        require_non_empty(self.task_id, "task_id")
        require_identifier(self.observation_profile_id, "observation_profile_id")
        require_identifier(self.action_profile_id, "action_profile_id")
        require_identifier(self.reward_profile_id, "reward_profile_id")
        if not isinstance(self.backend, RlBackendIdentity):
            raise ValueError("backend must be an RlBackendIdentity")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "mode": self.mode.value,
            "task_id": self.task_id,
            "observation_profile_id": self.observation_profile_id,
            "action_profile_id": self.action_profile_id,
            "reward_profile_id": self.reward_profile_id,
            "backend": self.backend.to_dict(),
            "seed": self.seed,
        }


STANDALONE_PPO_OBSERVATION_PROFILE = RlObservationProfile(
    profile_id="franka-pick-place-ppo-state-v1",
    components=(
        RlObservationComponent("robot.joint-position", 9),
        RlObservationComponent("robot.joint-velocity", 9),
        RlObservationComponent("tool-center.position", 3),
        RlObservationComponent("tool-center.quaternion", 4),
        RlObservationComponent("cube.position", 3, privileged=True),
        RlObservationComponent("cube.linear-velocity", 3, privileged=True),
        RlObservationComponent("goal.position", 3, privileged=True),
        RlObservationComponent("tool-to-cube.position", 3, privileged=True),
        RlObservationComponent("cube-to-goal.position", 3, privileged=True),
        RlObservationComponent("previous-action", 7),
        RlObservationComponent("task-phase.one-hot", len(PickPlaceRlPhase), privileged=True),
    ),
    excluded_modalities=("camera.front.rgb", "language.instruction"),
)

STANDALONE_PPO_ACTION_PROFILE = RlActionProfile(
    profile_id="franka-pick-place-ppo-ee-delta-v1",
    action_dimension=7,
    arm_dimension=6,
    gripper_dimension=1,
    control_hz=CONTROL_HZ,
    normalized_bound=1.0,
    translation_scale_m=IK_TRANSLATION_SCALE_M,
    rotation_scale_rad=IK_ROTATION_SCALE_RAD,
    gripper_threshold=0.0,
)

STANDALONE_PPO_REWARD_PROFILE_ID = "franka-pick-place-staged-reward-v2"

RSL_RL_PPO_BACKEND = RlBackendIdentity(
    backend_id="rsl-rl",
    algorithm="ppo",
    package="rsl-rl-lib",
    package_version="3.1.2",
    isaaclab_rl_version="0.4.7",
)

if STANDALONE_PPO_OBSERVATION_PROFILE.dimension != 52:
    raise RuntimeError("standalone PPO observation profile must remain 52D")


__all__ = [
    "RSL_RL_PPO_BACKEND",
    "STAGE9_RUN_CONFIG_SCHEMA_VERSION",
    "STAGE9_RUN_IDENTITY_SCHEMA_VERSION",
    "STANDALONE_PPO_ACTION_PROFILE",
    "STANDALONE_PPO_OBSERVATION_PROFILE",
    "STANDALONE_PPO_REWARD_PROFILE_ID",
    "STANDALONE_PPO_TASK_ID",
    "PickPlaceRlPhase",
    "RlActionProfile",
    "RlBackendIdentity",
    "RlMode",
    "RlObservationComponent",
    "RlObservationProfile",
    "Stage9RunIdentity",
]
