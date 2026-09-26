"""Tests for the reviewed Franka Coulomb-plus-viscous friction profile."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/embodied_ai/sim/tasks/franka_pick_place/dynamics_cfg.py"
)
_SPEC = importlib.util.spec_from_file_location("franka_joint_friction_config_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

FRANKA_ARM_JOINT_NAMES = _MODULE.FRANKA_ARM_JOINT_NAMES
FRANKA_GRIPPER_JOINT_NAMES = _MODULE.FRANKA_GRIPPER_JOINT_NAMES
FrankaJointFrictionConfig = _MODULE.FrankaJointFrictionConfig
franka_joint_friction_identity = _MODULE.franka_joint_friction_identity
reviewed_franka_joint_friction_config_path = _MODULE.reviewed_franka_joint_friction_config_path


def test_reviewed_joint_friction_profile_loads() -> None:
    config = FrankaJointFrictionConfig.from_toml(reviewed_franka_joint_friction_config_path())

    assert config.profile == "franka-joint-friction-v1"
    assert config.arm.joint_names == FRANKA_ARM_JOINT_NAMES
    assert config.gripper.joint_names == FRANKA_GRIPPER_JOINT_NAMES
    assert not config.gripper_enabled
    assert all(value > 0.0 for value in config.arm.coulomb)
    assert all(value > 0.0 for value in config.arm.viscous)


def test_identity_pins_exact_reviewed_file() -> None:
    path = reviewed_franka_joint_friction_config_path()
    identity = franka_joint_friction_identity()

    assert identity["path"] == str(path)
    assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["config"]["profile"] == "franka-joint-friction-v1"


def test_negative_parameter_is_rejected(tmp_path: Path) -> None:
    source = reviewed_franka_joint_friction_config_path().read_text(encoding="utf-8")
    invalid_path = tmp_path / "invalid.toml"
    invalid_path.write_text(
        source.replace(
            "coulomb_friction_nm = [0.20, 0.20, 0.15, 0.15, 0.10, 0.08, 0.05]",
            "coulomb_friction_nm = [-0.20, 0.20, 0.15, 0.15, 0.10, 0.08, 0.05]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        FrankaJointFrictionConfig.from_toml(invalid_path)


def test_noncanonical_joint_order_is_rejected(tmp_path: Path) -> None:
    source = reviewed_franka_joint_friction_config_path().read_text(encoding="utf-8")
    invalid_path = tmp_path / "invalid-order.toml"
    invalid_path.write_text(
        source.replace(
            '  "panda_joint1",\n  "panda_joint2",',
            '  "panda_joint2",\n  "panda_joint1",',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical Panda arm order"):
        FrankaJointFrictionConfig.from_toml(invalid_path)
