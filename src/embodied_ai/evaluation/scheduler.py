"""Receding-horizon action scheduler shared by tests and the Robot Client."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScheduledAction:
    raw: tuple[float, ...]
    executed: tuple[float, ...]
    request_index: int
    chunk_offset: int


class RecedingHorizonScheduler:
    def __init__(
        self,
        *,
        prediction_horizon: int = 50,
        execute_horizon: int = 5,
        action_dimension: int = 7,
        hard_action_limit: float = 1.5,
    ) -> None:
        if prediction_horizon != 50 or execute_horizon != 5 or action_dimension != 7:
            raise ValueError("Stage 8 scheduler requires a 50x7 chunk and execute horizon 5")
        self.prediction_horizon = prediction_horizon
        self.execute_horizon = execute_horizon
        self.action_dimension = action_dimension
        self.hard_action_limit = hard_action_limit
        self._queue: deque[ScheduledAction] = deque()
        self.request_count = 0
        self.executed_count = 0

    @property
    def needs_prediction(self) -> bool:
        return not self._queue

    @property
    def discarded_prediction_count(self) -> int:
        return self.request_count * self.prediction_horizon - self.executed_count

    def reset(self) -> None:
        self._queue.clear()
        self.request_count = 0
        self.executed_count = 0

    def accept(self, chunk: Sequence[Sequence[float]]) -> None:
        if not self.needs_prediction:
            raise RuntimeError("cannot replace a chunk before its execution horizon is consumed")
        if len(chunk) != self.prediction_horizon:
            raise ValueError("policy chunk must contain exactly 50 actions")
        request_index = self.request_count
        accepted: list[ScheduledAction] = []
        for offset, action in enumerate(chunk):
            if len(action) != self.action_dimension:
                raise ValueError(f"action {offset} must contain seven values")
            raw = tuple(float(value) for value in action)
            if not all(math.isfinite(value) for value in raw):
                raise ValueError(f"action {offset} contains non-finite values")
            if any(abs(value) > self.hard_action_limit for value in raw):
                raise ValueError(f"action {offset} exceeds hard action limit")
            if offset < self.execute_horizon:
                executed = tuple(max(-1.0, min(1.0, value)) for value in raw)
                accepted.append(ScheduledAction(raw, executed, request_index, offset))
        self._queue.extend(accepted)
        self.request_count += 1

    def pop(self) -> ScheduledAction:
        if not self._queue:
            raise RuntimeError("a new policy prediction is required")
        action = self._queue.popleft()
        self.executed_count += 1
        return action


__all__ = ["RecedingHorizonScheduler", "ScheduledAction"]
