"""Shared Isaac-side expert interfaces for demonstration collection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from isaaclab.envs import ManagerBasedEnv

from embodied_ai.contracts import ExpertMetadata


@dataclass(frozen=True, slots=True)
class ExpertTaskContext:
    """Task semantics and language selected for one rollout."""

    task: str
    instruction: str
    instruction_id: str
    instruction_language: str
    goal_position_env_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ExpertStep:
    """One batched action plus expert-local lifecycle signals."""

    actions: torch.Tensor
    phases: tuple[str, ...]
    done: torch.Tensor
    failed: torch.Tensor
    failure_reasons: tuple[str | None, ...]


class Expert(Protocol):
    """Common boundary implemented by scripted, learned, and teleoperated experts."""

    @property
    def metadata(self) -> ExpertMetadata: ...

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None: ...

    def act(
        self,
        env: ManagerBasedEnv,
        observations: Mapping[str, object],
    ) -> ExpertStep: ...
