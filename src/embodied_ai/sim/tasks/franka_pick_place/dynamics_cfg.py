"""Reviewed Franka joint-friction profile and Isaac Lab robot config builder.

This module intentionally keeps Isaac Lab imports inside the builder. The TOML
loader and validation can therefore run in the lightweight development environment.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FRANKA_JOINT_FRICTION_SCHEMA_VERSION = "embodied-ai.franka-joint-friction/v1"
FRANKA_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
FRANKA_GRIPPER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_FRANKA_JOINT_FRICTION_CONFIG_PATH = (
    _REPOSITORY_ROOT / "configs" / "sim" / "franka_pick_place" / "joint_friction_v1.toml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_nonnegative_sequence(
    value: object,
    *,
    label: str,
    expected_length: int,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{label} must contain exactly {expected_length} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} values must be numeric")
        number = float(item)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{label} values must be finite and non-negative")
        result.append(number)
    return tuple(result)


def _joint_names(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a non-empty string list")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicate names")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class JointFrictionGroupConfig:
    """Per-joint Coulomb and viscous friction parameters in declared order."""

    joint_names: tuple[str, ...]
    coulomb: tuple[float, ...]
    viscous: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_names": list(self.joint_names),
            "coulomb": list(self.coulomb),
            "viscous": list(self.viscous),
        }

    def coulomb_by_joint(self) -> dict[str, float]:
        return dict(zip(self.joint_names, self.coulomb, strict=True))

    def viscous_by_joint(self) -> dict[str, float]:
        return dict(zip(self.joint_names, self.viscous, strict=True))


@dataclass(frozen=True, slots=True)
class FrankaJointFrictionConfig:
    """Versioned friction parameters applied to one Franka articulation config."""

    schema_version: str
    profile: str
    arm: JointFrictionGroupConfig
    gripper_enabled: bool
    gripper: JointFrictionGroupConfig

    @classmethod
    def from_toml(cls, path: Path) -> FrankaJointFrictionConfig:
        with path.open("rb") as stream:
            source = tomllib.load(stream)
        if set(source) != {"schema_version", "profile", "arm", "gripper"}:
            raise ValueError("joint-friction config has unexpected or missing top-level fields")
        if source["schema_version"] != FRANKA_JOINT_FRICTION_SCHEMA_VERSION:
            raise ValueError("unsupported Franka joint-friction schema")
        if not isinstance(source["profile"], str) or not source["profile"]:
            raise ValueError("joint-friction profile must be a non-empty string")

        arm_source = source["arm"]
        gripper_source = source["gripper"]
        if not isinstance(arm_source, dict) or set(arm_source) != {
            "joint_names",
            "coulomb_friction_nm",
            "viscous_friction_nm_s_rad",
        }:
            raise ValueError("arm friction section has unexpected or missing fields")
        if not isinstance(gripper_source, dict) or set(gripper_source) != {
            "enabled",
            "joint_names",
            "coulomb_friction_n",
            "viscous_friction_n_s_m",
        }:
            raise ValueError("gripper friction section has unexpected or missing fields")
        if not isinstance(gripper_source["enabled"], bool):
            raise ValueError("gripper.enabled must be boolean")

        arm_names = _joint_names(arm_source["joint_names"], label="arm.joint_names")
        gripper_names = _joint_names(gripper_source["joint_names"], label="gripper.joint_names")
        arm = JointFrictionGroupConfig(
            joint_names=arm_names,
            coulomb=_finite_nonnegative_sequence(
                arm_source["coulomb_friction_nm"],
                label="arm.coulomb_friction_nm",
                expected_length=len(arm_names),
            ),
            viscous=_finite_nonnegative_sequence(
                arm_source["viscous_friction_nm_s_rad"],
                label="arm.viscous_friction_nm_s_rad",
                expected_length=len(arm_names),
            ),
        )
        gripper = JointFrictionGroupConfig(
            joint_names=gripper_names,
            coulomb=_finite_nonnegative_sequence(
                gripper_source["coulomb_friction_n"],
                label="gripper.coulomb_friction_n",
                expected_length=len(gripper_names),
            ),
            viscous=_finite_nonnegative_sequence(
                gripper_source["viscous_friction_n_s_m"],
                label="gripper.viscous_friction_n_s_m",
                expected_length=len(gripper_names),
            ),
        )
        config = cls(
            schema_version=source["schema_version"],
            profile=source["profile"],
            arm=arm,
            gripper_enabled=gripper_source["enabled"],
            gripper=gripper,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.arm.joint_names != FRANKA_ARM_JOINT_NAMES:
            raise ValueError("arm.joint_names must use the canonical Panda arm order")
        if self.gripper.joint_names != FRANKA_GRIPPER_JOINT_NAMES:
            raise ValueError("gripper.joint_names must use the canonical Panda finger order")
        if self.gripper_enabled and not any(self.gripper.coulomb + self.gripper.viscous):
            raise ValueError("enabled gripper friction must contain at least one non-zero value")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "arm": self.arm.to_dict(),
            "gripper_enabled": self.gripper_enabled,
            "gripper": self.gripper.to_dict(),
        }


def reviewed_franka_joint_friction_config_path() -> Path:
    return DEFAULT_FRANKA_JOINT_FRICTION_CONFIG_PATH


def load_reviewed_franka_joint_friction_config() -> FrankaJointFrictionConfig:
    return FrankaJointFrictionConfig.from_toml(reviewed_franka_joint_friction_config_path())


def franka_joint_friction_identity() -> dict[str, object]:
    """Return the exact reviewed profile identity for run manifests and reports."""

    path = reviewed_franka_joint_friction_config_path()
    config = FrankaJointFrictionConfig.from_toml(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "config": config.to_dict(),
    }


def _apply_group_to_actuator(
    actuator: Any,
    group: JointFrictionGroupConfig,
    joint_names: tuple[str, ...],
) -> None:
    coulomb = {
        name: value for name, value in group.coulomb_by_joint().items() if name in joint_names
    }
    viscous = {
        name: value for name, value in group.viscous_by_joint().items() if name in joint_names
    }
    if set(coulomb) != set(joint_names) or set(viscous) != set(joint_names):
        raise ValueError("actuator joint partition does not match the reviewed friction profile")

    # Isaac Sim 5.x treats static and dynamic friction as efforts. Setting both
    # to tau_c implements a Coulomb term without adding a separate stiction band.
    actuator.friction = coulomb
    actuator.dynamic_friction = dict(coulomb)
    actuator.viscous_friction = viscous


def make_franka_high_pd_with_joint_friction(
    *,
    prim_path: str,
    activate_contact_sensors: bool = False,
    friction: FrankaJointFrictionConfig | None = None,
) -> Any:
    """Build an independent HIGH_PD Franka config with the reviewed friction profile."""

    # Keep the module importable without Isaac Lab for config review and unit tests.
    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

    selected = friction or load_reviewed_franka_joint_friction_config()
    selected.validate()
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=prim_path)
    robot_cfg.spawn.activate_contact_sensors = activate_contact_sensors
    _apply_group_to_actuator(
        robot_cfg.actuators["panda_shoulder"], selected.arm, FRANKA_ARM_JOINT_NAMES[:4]
    )
    _apply_group_to_actuator(
        robot_cfg.actuators["panda_forearm"], selected.arm, FRANKA_ARM_JOINT_NAMES[4:]
    )
    if selected.gripper_enabled:
        _apply_group_to_actuator(
            robot_cfg.actuators["panda_hand"], selected.gripper, FRANKA_GRIPPER_JOINT_NAMES
        )
    return robot_cfg


__all__ = [
    "DEFAULT_FRANKA_JOINT_FRICTION_CONFIG_PATH",
    "FRANKA_ARM_JOINT_NAMES",
    "FRANKA_GRIPPER_JOINT_NAMES",
    "FRANKA_JOINT_FRICTION_SCHEMA_VERSION",
    "FrankaJointFrictionConfig",
    "JointFrictionGroupConfig",
    "franka_joint_friction_identity",
    "load_reviewed_franka_joint_friction_config",
    "make_franka_high_pd_with_joint_friction",
    "reviewed_franka_joint_friction_config_path",
]
