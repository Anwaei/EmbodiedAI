"""Per-rollout and aggregate metrics for Stage 8 policy comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

import numpy as np


def rollout_metrics(
    *,
    cube_positions: np.ndarray,
    goal_errors: np.ndarray,
    executed_actions: np.ndarray,
    inference_latencies_ms: Iterable[float],
    success: bool,
) -> dict[str, float | int]:
    if cube_positions.ndim != 2 or cube_positions.shape[1] != 3:
        raise ValueError("cube_positions must have shape (T, 3)")
    if goal_errors.shape != (cube_positions.shape[0],):
        raise ValueError("goal_errors must have shape (T,)")
    if executed_actions.shape != (cube_positions.shape[0], 7):
        raise ValueError("executed_actions must have shape (T, 7)")
    latency = np.asarray(tuple(inference_latencies_ms), dtype=np.float64)
    path_length = float(
        np.linalg.norm(np.diff(cube_positions.astype(np.float64), axis=0), axis=1).sum()
    ) if len(cube_positions) > 1 else 0.0
    return {
        "success": int(success),
        "steps": int(cube_positions.shape[0]),
        "final_goal_error_m": float(goal_errors[-1]),
        "minimum_goal_error_m": float(goal_errors.min()),
        "cube_path_length_m": path_length,
        "action_saturation_rate": float((np.abs(executed_actions) >= 0.999).mean()),
        "inference_request_count": int(latency.size),
        "mean_inference_latency_ms": float(latency.mean()) if latency.size else 0.0,
        "p95_inference_latency_ms": float(np.quantile(latency, 0.95)) if latency.size else 0.0,
    }


def aggregate_rollouts(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(records)
    if not rows:
        raise ValueError("at least one rollout record is required")
    success = np.asarray([float(row["metrics"]["success"]) for row in rows])  # type: ignore[index]
    goal = np.asarray([float(row["metrics"]["final_goal_error_m"]) for row in rows])  # type: ignore[index]
    steps = np.asarray([float(row["metrics"]["steps"]) for row in rows])  # type: ignore[index]
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for row, succeeded in zip(rows, success, strict=True):
        by_scenario[str(row["scenario_id"])].append(float(succeeded))
    result = {
        "rollout_count": len(rows),
        "success_count": int(success.sum()),
        "success_rate": float(success.mean()),
        "mean_final_goal_error_m": float(goal.mean()),
        "median_final_goal_error_m": float(np.median(goal)),
        "mean_steps": float(steps.mean()),
        "success_rate_by_scenario": {
            key: float(np.mean(values)) for key, values in sorted(by_scenario.items())
        },
    }
    result["outcome_counts"] = dict(
        sorted(Counter(str(row["outcome"]) for row in rows).items())
    )
    return result


__all__ = ["aggregate_rollouts", "rollout_metrics"]
