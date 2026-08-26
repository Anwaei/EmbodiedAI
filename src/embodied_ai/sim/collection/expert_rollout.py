"""Reusable rollout lifecycle for one Isaac expert episode."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv

from embodied_ai.contracts import EpisodeOutcome, ObservationSchema
from embodied_ai.sim.experts import Expert
from embodied_ai.sim.recording import NpyEpisodeRecorder, RecordedEpisode


@dataclass(frozen=True, slots=True)
class ExpertRolloutResult:
    """Published episode and bounded controller diagnostics."""

    recorded: RecordedEpisode
    outcome: EpisodeOutcome
    termination_reason: str | None
    phase_counts: tuple[tuple[str, int], ...]


def policy_terms(observations: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    """Return the named, unconcatenated policy observation group."""

    policy = observations.get("policy")
    if not isinstance(policy, Mapping):
        raise RuntimeError("expected unconcatenated policy observations")
    if not all(isinstance(value, torch.Tensor) for value in policy.values()):
        raise RuntimeError("policy observations must be torch tensors")
    return policy


def contract_observation(
    observations: Mapping[str, object],
    *,
    schema: ObservationSchema,
    term_map: Mapping[str, str],
    env_index: int,
) -> dict[str, np.ndarray]:
    """Adapt one environment from Isaac tensors to contract-keyed NumPy values."""

    policy = policy_terms(observations)
    return {
        field.key: policy[term_map[field.key]][env_index].detach().cpu().numpy()
        for field in schema.fields
    }


def collect_expert_episode(
    *,
    env: object,
    expert: Expert,
    recorder: NpyEpisodeRecorder,
    observation_schema: ObservationSchema,
    observation_term_map: Mapping[str, str],
    control_period_ns: int,
    max_steps: int,
    seed: int,
    env_index: int = 0,
) -> ExpertRolloutResult:
    """Reset once, collect one bounded trajectory, and publish its terminal outcome."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if control_period_ns <= 0:
        raise ValueError("control_period_ns must be positive")

    observations, _ = env.reset(seed=seed)
    unwrapped: ManagerBasedEnv = env.unwrapped
    if unwrapped.num_envs <= env_index:
        raise ValueError("env_index is outside the vectorized environment")
    expert.reset()
    phase_counts: Counter[str] = Counter()

    for step_index in range(max_steps):
        expert_step = expert.act(unwrapped, observations)
        phase = expert_step.phases[env_index]
        phase_counts[phase] += 1

        if bool(expert_step.failed[env_index]):
            reason = expert_step.failure_reasons[env_index] or "unspecified"
            return _finalize(
                recorder,
                EpisodeOutcome.FAILURE,
                f"expert-{reason}",
                phase_counts,
            )

        action = expert_step.actions[env_index]
        recorder.append(
            contract_observation(
                observations,
                schema=observation_schema,
                term_map=observation_term_map,
                env_index=env_index,
            ),
            action.detach().cpu().numpy(),
            step_index * control_period_ns,
        )

        with torch.inference_mode():
            observations, _, terminated, truncated, _ = env.step(expert_step.actions)

        # ManagerBasedRLEnv resets terminal environments inside step(). Read the saved term
        # buffers immediately, before accepting the returned post-reset observation.
        success = unwrapped.termination_manager.get_term("success")
        failure = unwrapped.termination_manager.get_term("failure")
        if bool(success[env_index]):
            return _finalize(recorder, EpisodeOutcome.SUCCESS, None, phase_counts)
        if bool(failure[env_index]):
            return _finalize(
                recorder,
                EpisodeOutcome.FAILURE,
                "task-workspace-failure",
                phase_counts,
            )
        if bool(truncated[env_index]):
            return _finalize(
                recorder,
                EpisodeOutcome.TRUNCATED,
                "task-time-limit",
                phase_counts,
            )
        if bool(terminated[env_index]):
            return _finalize(
                recorder,
                EpisodeOutcome.FAILURE,
                "unknown-task-termination",
                phase_counts,
            )
        if bool(expert_step.done[env_index]):
            return _finalize(
                recorder,
                EpisodeOutcome.FAILURE,
                "expert-finished-without-task-success",
                phase_counts,
            )

    return _finalize(
        recorder,
        EpisodeOutcome.TRUNCATED,
        "collector-step-limit",
        phase_counts,
    )


def _finalize(
    recorder: NpyEpisodeRecorder,
    outcome: EpisodeOutcome,
    termination_reason: str | None,
    phase_counts: Counter[str],
) -> ExpertRolloutResult:
    if recorder.step_count == 0:
        raise RuntimeError("expert failed before producing a recordable action")
    recorded = recorder.finalize(outcome, termination_reason)
    return ExpertRolloutResult(
        recorded=recorded,
        outcome=outcome,
        termination_reason=termination_reason,
        phase_counts=tuple(sorted(phase_counts.items())),
    )
