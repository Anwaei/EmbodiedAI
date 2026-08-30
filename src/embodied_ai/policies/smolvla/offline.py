"""Reusable deterministic offline inference and action-space metrics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .dataset import OfflineAnchor, policy_input_from_sample
from .processing import ProjectProcessors
from .runtime import fixed_noise


@dataclass(frozen=True, slots=True)
class OfflineInferenceResult:
    records: tuple[dict[str, object], ...]
    predictions: np.ndarray
    targets: np.ndarray
    chunks: np.ndarray
    metrics: dict[str, object]
    latency_seconds: tuple[float, ...]
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def _policy_reset(policy: Any) -> None:
    reset = getattr(policy, "reset", None)
    if not callable(reset):
        raise TypeError("loaded policy does not expose reset()")
    reset()


def _policy_chunk(policy: Any, batch: dict[str, Any], noise: torch.Tensor) -> torch.Tensor:
    predict = getattr(policy, "predict_action_chunk", None)
    if not callable(predict):
        raise TypeError("loaded policy does not expose predict_action_chunk()")
    return predict(batch, noise=noise)


def _aggregate_metrics(
    records: list[dict[str, object]],
    predictions: np.ndarray,
    targets: np.ndarray,
    action_names: tuple[str, ...],
) -> dict[str, object]:
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError("offline prediction and target arrays must have matching (N, D) shapes")
    error = predictions - targets
    absolute = np.abs(error)
    squared = np.square(error)
    bound_violations = np.logical_or(predictions < -1.0, predictions > 1.0)

    by_instruction: dict[str, dict[str, object]] = {}
    instructions = sorted({str(record["task"]) for record in records})
    for instruction in instructions:
        selection = np.asarray([record["task"] == instruction for record in records])
        selected_error = error[selection]
        by_instruction[instruction] = {
            "sample_count": int(selection.sum()),
            "mae": float(np.abs(selected_error).mean()),
            "rmse": float(np.sqrt(np.square(selected_error).mean())),
        }

    return {
        "sample_count": int(predictions.shape[0]),
        "finite_output_rate": float(np.isfinite(predictions).all(axis=1).mean()),
        "mae": float(absolute.mean()),
        "rmse": float(math.sqrt(float(squared.mean()))),
        "mae_by_component": {
            name: float(absolute[:, index].mean()) for index, name in enumerate(action_names)
        },
        "rmse_by_component": {
            name: float(np.sqrt(squared[:, index].mean()))
            for index, name in enumerate(action_names)
        },
        "gripper_sign_agreement": float(
            ((predictions[:, -1] >= 0.0) == (targets[:, -1] >= 0.0)).mean()
        ),
        "bound_violation_count": int(bound_violations.sum()),
        "bound_violation_rate": float(bound_violations.mean()),
        "prediction_minimum": float(predictions.min()),
        "prediction_maximum": float(predictions.max()),
        "by_instruction": by_instruction,
    }


def run_offline_inference(
    policy: Any,
    policy_config: Any,
    processors: ProjectProcessors,
    dataset: Any,
    anchors: tuple[OfflineAnchor, ...],
    *,
    noise_seed: int,
    warmup_samples: int,
) -> OfflineInferenceResult:
    """Run independent deterministic chunks without action labels in model input."""

    if not anchors:
        raise ValueError("at least one offline anchor is required")
    policy.eval()

    # Warm up model kernels separately so recorded latency excludes one-time startup.
    warmup_sample = dataset[anchors[0].dataset_index]
    warmup_batch = processors.preprocess_inference(policy_input_from_sample(warmup_sample))
    for warmup_index in range(warmup_samples):
        _policy_reset(policy)
        warmup_noise = fixed_noise(
            policy_config,
            seed=noise_seed - warmup_samples + warmup_index,
            device=policy_config.device,
        )
        with torch.inference_mode():
            _policy_chunk(policy, warmup_batch, warmup_noise)
    torch.cuda.synchronize()

    records: list[dict[str, object]] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    latency_seconds: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for anchor in anchors:
        sample = dataset[anchor.dataset_index]
        if int(sample["episode_index"].item()) != anchor.episode_index:
            raise RuntimeError("offline anchor resolved to the wrong episode")
        if int(sample["frame_index"].item()) != anchor.frame_index:
            raise RuntimeError("offline anchor resolved to the wrong frame")
        batch = processors.preprocess_inference(policy_input_from_sample(sample))
        noise = fixed_noise(
            policy_config,
            seed=noise_seed + anchor.dataset_index,
            device=policy_config.device,
        )
        _policy_reset(policy)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            normalized_chunk = _policy_chunk(policy, batch, noise)
            action_chunk, diagnostics = processors.postprocess(normalized_chunk)
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        target = sample[processors.profile.action_key].detach().cpu().numpy().astype(np.float32)
        predicted = action_chunk[0, 0].detach().numpy().astype(np.float32)
        chunk = action_chunk[0].detach().numpy().astype(np.float32)
        predictions.append(predicted)
        targets.append(target)
        chunks.append(chunk)
        latency_seconds.append(latency)
        records.append(
            {
                **anchor.to_dict(),
                "noise_seed": noise_seed + anchor.dataset_index,
                "latency_seconds": latency,
                "target_action": target.tolist(),
                "predicted_first_action": predicted.tolist(),
                "postprocess_diagnostics": diagnostics,
            }
        )

    prediction_array = np.stack(predictions)
    target_array = np.stack(targets)
    chunk_array = np.stack(chunks)
    metrics = _aggregate_metrics(
        records,
        prediction_array,
        target_array,
        processors.profile.action_names,
    )
    metrics["latency_seconds"] = {
        "mean": float(np.mean(latency_seconds)),
        "median": float(np.median(latency_seconds)),
        "minimum": float(np.min(latency_seconds)),
        "maximum": float(np.max(latency_seconds)),
    }
    return OfflineInferenceResult(
        records=tuple(records),
        predictions=prediction_array,
        targets=target_array,
        chunks=chunk_array,
        metrics=metrics,
        latency_seconds=tuple(latency_seconds),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    )


__all__ = ["OfflineInferenceResult", "run_offline_inference"]
