"""Deterministic task-space state-machine expert for Franka pick-and-place."""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import IntEnum
from pathlib import Path

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import math as math_utils

from embodied_ai.contracts import ExpertKind, ExpertMetadata
from embodied_ai.contracts.tasks.franka_pick_place import (
    GRIPPER_CLOSE_ACTION,
    GRIPPER_OPEN_ACTION,
    IK_ROTATION_SCALE_RAD,
    IK_TRANSLATION_SCALE_M,
    TASK_NAME,
)

from .base import ExpertStep, ExpertTaskContext

_CONFIG_SCHEMA_VERSION = "embodied-ai.franka-pick-place-state-machine/v1"


class StateMachinePhase(IntEnum):
    """Per-environment controller phases."""

    MOVE_ABOVE_CUBE = 0
    DESCEND_TO_GRASP = 1
    CLOSE_GRIPPER = 2
    LIFT_CUBE = 3
    MOVE_ABOVE_GOAL = 4
    DESCEND_TO_PLACE = 5
    OPEN_GRIPPER = 6
    RETREAT = 7
    DONE = 8
    FAILED = 9


_PHASE_CONFIG_NAMES = {
    StateMachinePhase.MOVE_ABOVE_CUBE: "move_above_cube",
    StateMachinePhase.DESCEND_TO_GRASP: "descend_to_grasp",
    StateMachinePhase.CLOSE_GRIPPER: "close_gripper",
    StateMachinePhase.LIFT_CUBE: "lift_cube",
    StateMachinePhase.MOVE_ABOVE_GOAL: "move_above_goal",
    StateMachinePhase.DESCEND_TO_PLACE: "descend_to_place",
    StateMachinePhase.OPEN_GRIPPER: "open_gripper",
    StateMachinePhase.RETREAT: "retreat",
}


@dataclass(frozen=True, slots=True)
class FrankaPickPlaceStateMachineConfig:
    """Reviewed numeric parameters loaded from the committed TOML file."""

    identifier: str
    revision: str
    position_gain: float
    rotation_gain: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    pre_grasp_height_m: float
    grasp_height_offset_m: float
    lift_height_m: float
    cube_lift_threshold_m: float
    transfer_height_m: float
    place_height_offset_m: float
    retreat_x_offset_m: float
    close_settle_steps: int
    release_settle_steps: int
    phase_timeout_steps: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        positive_values = (
            self.position_gain,
            self.rotation_gain,
            self.position_tolerance_m,
            self.orientation_tolerance_rad,
            self.pre_grasp_height_m,
            self.lift_height_m,
            self.cube_lift_threshold_m,
            self.transfer_height_m,
        )
        if not all(value > 0.0 for value in positive_values):
            raise ValueError("state-machine gains, tolerances, and heights must be positive")
        if self.close_settle_steps <= 0 or self.release_settle_steps <= 0:
            raise ValueError("state-machine settle steps must be positive")
        timeout_names = {name for name, _ in self.phase_timeout_steps}
        expected_names = set(_PHASE_CONFIG_NAMES.values())
        if timeout_names != expected_names:
            raise ValueError("phase_timeout_steps must define every active phase exactly once")
        if any(steps <= 0 for _, steps in self.phase_timeout_steps):
            raise ValueError("state-machine phase timeouts must be positive")

    def timeout_for(self, phase: StateMachinePhase) -> int:
        name = _PHASE_CONFIG_NAMES[phase]
        return dict(self.phase_timeout_steps)[name]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state_machine_config(
    path: str | Path,
) -> tuple[FrankaPickPlaceStateMachineConfig, str]:
    """Load the reviewed TOML and return it with its immutable content digest."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as stream:
        source = tomllib.load(stream)
    if source.pop("schema_version", None) != _CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported state-machine configuration schema")
    timeouts = source.pop("phase_timeout_steps", None)
    if not isinstance(timeouts, dict):
        raise ValueError("phase_timeout_steps must be a TOML table")
    source["phase_timeout_steps"] = tuple(sorted(timeouts.items()))
    expected_fields = {field.name for field in fields(FrankaPickPlaceStateMachineConfig)}
    if set(source) != expected_fields:
        raise ValueError("state-machine configuration fields do not match the contract")
    try:
        config = FrankaPickPlaceStateMachineConfig(**source)
    except TypeError as error:
        raise ValueError("state-machine configuration contains invalid values") from error
    return config, _sha256(config_path)


class FrankaPickPlaceStateMachineExpert:
    """Vectorized finite-state expert that emits the public normalized action schema."""

    def __init__(
        self,
        *,
        config: FrankaPickPlaceStateMachineConfig,
        configuration_revision: str,
        task_context: ExpertTaskContext,
        num_envs: int,
        device: str | torch.device,
    ) -> None:
        if task_context.task != TASK_NAME:
            raise ValueError(f"state-machine expert does not support task {task_context.task!r}")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.config = config
        self.task_context = task_context
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._metadata = ExpertMetadata(
            kind=ExpertKind.STATE_MACHINE,
            identifier=config.identifier,
            revision=config.revision,
            configuration_revision=configuration_revision,
        )
        self._phase = torch.full(
            (num_envs,),
            int(StateMachinePhase.MOVE_ABOVE_CUBE),
            dtype=torch.int64,
            device=self.device,
        )
        self._phase_steps = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._failure_reasons: list[str | None] = [None] * num_envs
        self._goal = torch.tensor(
            task_context.goal_position_env_m,
            dtype=torch.float32,
            device=self.device,
        ).repeat(num_envs, 1)
        self._grasp_orientation = torch.tensor(
            (0.0, 1.0, 0.0, 0.0),
            dtype=torch.float32,
            device=self.device,
        ).repeat(num_envs, 1)

    @property
    def metadata(self) -> ExpertMetadata:
        return self._metadata

    @property
    def phase_names(self) -> tuple[str, ...]:
        return tuple(StateMachinePhase(int(value)).name.lower() for value in self._phase)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            indices = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            indices = [int(index) for index in env_ids.flatten().tolist()]
        else:
            indices = [int(index) for index in env_ids]
        if not indices:
            return
        index_tensor = torch.tensor(indices, dtype=torch.int64, device=self.device)
        self._phase[index_tensor] = int(StateMachinePhase.MOVE_ABOVE_CUBE)
        self._phase_steps[index_tensor] = 0
        for index in indices:
            self._failure_reasons[index] = None

    def _set_phase(self, env_index: int, phase: StateMachinePhase) -> None:
        self._phase[env_index] = int(phase)
        self._phase_steps[env_index] = 0

    def _fail(self, env_index: int, reason: str) -> None:
        self._set_phase(env_index, StateMachinePhase.FAILED)
        self._failure_reasons[env_index] = reason

    def _advance_phases(
        self,
        tool_position: torch.Tensor,
        tool_orientation: torch.Tensor,
        cube_position: torch.Tensor,
    ) -> None:
        orientation_error = math_utils.compute_pose_error(
            tool_position,
            tool_orientation,
            tool_position,
            self._grasp_orientation,
            rot_error_type="axis_angle",
        )[1]
        orientation_ready = torch.linalg.vector_norm(orientation_error, dim=-1) <= (
            self.config.orientation_tolerance_rad
        )

        for env_index in range(self.num_envs):
            phase = StateMachinePhase(int(self._phase[env_index]))
            if phase in (StateMachinePhase.DONE, StateMachinePhase.FAILED):
                continue
            if self._phase_steps[env_index] >= self.config.timeout_for(phase):
                self._fail(env_index, f"phase-timeout-{phase.name.lower()}")
                continue

            target = self._target_position(phase, cube_position)[env_index]
            position_ready = bool(
                torch.linalg.vector_norm(target - tool_position[env_index])
                <= self.config.position_tolerance_m
            )
            orientation_is_ready = bool(orientation_ready[env_index])

            if phase is StateMachinePhase.MOVE_ABOVE_CUBE:
                if position_ready and orientation_is_ready:
                    self._set_phase(env_index, StateMachinePhase.DESCEND_TO_GRASP)
            elif phase is StateMachinePhase.DESCEND_TO_GRASP:
                if position_ready and orientation_is_ready:
                    self._set_phase(env_index, StateMachinePhase.CLOSE_GRIPPER)
            elif phase is StateMachinePhase.CLOSE_GRIPPER:
                if self._phase_steps[env_index] >= self.config.close_settle_steps:
                    self._set_phase(env_index, StateMachinePhase.LIFT_CUBE)
            elif phase is StateMachinePhase.LIFT_CUBE:
                cube_lifted = bool(
                    cube_position[env_index, 2] >= self.config.cube_lift_threshold_m
                )
                if position_ready and cube_lifted:
                    self._set_phase(env_index, StateMachinePhase.MOVE_ABOVE_GOAL)
            elif phase is StateMachinePhase.MOVE_ABOVE_GOAL:
                if position_ready:
                    self._set_phase(env_index, StateMachinePhase.DESCEND_TO_PLACE)
            elif phase is StateMachinePhase.DESCEND_TO_PLACE:
                if position_ready:
                    self._set_phase(env_index, StateMachinePhase.OPEN_GRIPPER)
            elif phase is StateMachinePhase.OPEN_GRIPPER:
                if self._phase_steps[env_index] >= self.config.release_settle_steps:
                    self._set_phase(env_index, StateMachinePhase.RETREAT)
            elif phase is StateMachinePhase.RETREAT and position_ready:
                self._set_phase(env_index, StateMachinePhase.DONE)

    def _target_position(
        self,
        phase: StateMachinePhase,
        cube_position: torch.Tensor,
    ) -> torch.Tensor:
        target = cube_position.clone()
        if phase is StateMachinePhase.MOVE_ABOVE_CUBE:
            target[:, 2] += self.config.pre_grasp_height_m
        elif phase in (
            StateMachinePhase.DESCEND_TO_GRASP,
            StateMachinePhase.CLOSE_GRIPPER,
        ):
            target[:, 2] += self.config.grasp_height_offset_m
        elif phase is StateMachinePhase.LIFT_CUBE:
            target[:, 2] = self.config.lift_height_m
        elif phase is StateMachinePhase.MOVE_ABOVE_GOAL:
            target = self._goal.clone()
            target[:, 2] = self.config.transfer_height_m
        elif phase in (
            StateMachinePhase.DESCEND_TO_PLACE,
            StateMachinePhase.OPEN_GRIPPER,
        ):
            target = self._goal.clone()
            target[:, 2] += self.config.place_height_offset_m
        elif phase is StateMachinePhase.RETREAT:
            target = self._goal.clone()
            target[:, 0] += self.config.retreat_x_offset_m
            target[:, 2] = self.config.transfer_height_m
        return target

    def act(
        self,
        env: ManagerBasedEnv,
        observations: Mapping[str, object],
    ) -> ExpertStep:
        del observations  # The scripted expert intentionally uses privileged simulator state.
        frame_data = env.scene["ee_frame"].data
        tool_position = frame_data.target_pos_source[:, 0, :]
        tool_orientation = frame_data.target_quat_source[:, 0, :]
        cube_position = env.scene["cube"].data.root_pos_w - env.scene.env_origins

        self._advance_phases(tool_position, tool_orientation, cube_position)
        actions = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)

        # Build one target per environment after transitions so the first action of a new
        # phase immediately moves toward that phase's target.
        for env_index in range(self.num_envs):
            phase = StateMachinePhase(int(self._phase[env_index]))
            if phase in (StateMachinePhase.DONE, StateMachinePhase.FAILED):
                actions[env_index, 6] = GRIPPER_OPEN_ACTION
                continue
            target_position = self._target_position(phase, cube_position)[env_index : env_index + 1]
            position_error, rotation_error = math_utils.compute_pose_error(
                tool_position[env_index : env_index + 1],
                tool_orientation[env_index : env_index + 1],
                target_position,
                self._grasp_orientation[env_index : env_index + 1],
                rot_error_type="axis_angle",
            )
            actions[env_index, :3] = (
                self.config.position_gain * position_error[0] / IK_TRANSLATION_SCALE_M
            )
            actions[env_index, 3:6] = (
                self.config.rotation_gain * rotation_error[0] / IK_ROTATION_SCALE_RAD
            )
            actions[env_index, 6] = (
                GRIPPER_OPEN_ACTION
                if phase
                in (
                    StateMachinePhase.MOVE_ABOVE_CUBE,
                    StateMachinePhase.DESCEND_TO_GRASP,
                    StateMachinePhase.OPEN_GRIPPER,
                    StateMachinePhase.RETREAT,
                )
                else GRIPPER_CLOSE_ACTION
            )

        # Clipping here enforces the same public bounds checked again by the recorder.
        actions.clamp_(-1.0, 1.0)
        active = (self._phase != int(StateMachinePhase.DONE)) & (
            self._phase != int(StateMachinePhase.FAILED)
        )
        self._phase_steps[active] += 1
        return ExpertStep(
            actions=actions,
            phases=self.phase_names,
            done=self._phase == int(StateMachinePhase.DONE),
            failed=self._phase == int(StateMachinePhase.FAILED),
            failure_reasons=tuple(self._failure_reasons),
        )
