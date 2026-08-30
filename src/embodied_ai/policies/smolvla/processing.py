"""Project-owned SmolVLA preprocessing, postprocessing, and statistics."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .profile import FRANKA_PICK_PLACE_SMOLVLA_PROFILE, SmolVLAProjectProfile

PROCESSOR_MANIFEST_SCHEMA_VERSION = "embodied-ai.smolvla-processors/v1"
PROCESSOR_MANIFEST_NAME = "embodied_ai_processor_manifest.json"
PROCESSOR_STATS_NAME = "embodied_ai_processor_stats.json"
PREPROCESSOR_CONFIG_NAME = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG_NAME = "policy_postprocessor.json"


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value: Any = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _numeric_stats(values: np.ndarray) -> dict[str, object]:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("normalization values must have shape (N, D) with N > 0")
    if values.dtype != np.float32:
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("normalization values contain non-finite data")
    float64 = values.astype(np.float64)
    return {
        "min": float64.min(axis=0).tolist(),
        "max": float64.max(axis=0).tolist(),
        "mean": float64.mean(axis=0).tolist(),
        "std": float64.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(float64, 0.01, axis=0).tolist(),
        "q10": np.quantile(float64, 0.10, axis=0).tolist(),
        "q50": np.quantile(float64, 0.50, axis=0).tolist(),
        "q90": np.quantile(float64, 0.90, axis=0).tolist(),
        "q99": np.quantile(float64, 0.99, axis=0).tolist(),
    }


def compute_policy_statistics(
    dataset: Any,
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE,
) -> dict[str, dict[str, object]]:
    """Compute state/action stats from the dataset's selected episodes only."""

    profile.validate_dataset_meta(dataset.meta)
    arrays: dict[str, np.ndarray] = {}
    for key, dimension in (
        (profile.state_key, profile.state_dimension),
        (profile.action_key, profile.action_dimension),
    ):
        column = dataset.hf_dataset[key]
        values = np.stack([np.asarray(item, dtype=np.float32) for item in column])
        if values.shape != (len(dataset), dimension):
            raise ValueError(
                f"selected dataset feature {key!r} has unexpected shape {values.shape}"
            )
        arrays[key] = values
    return {key: _numeric_stats(value) for key, value in arrays.items()}


def tensor_ready_statistics(
    value: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for feature, feature_stats in value.items():
        if not isinstance(feature, str) or not isinstance(feature_stats, dict):
            raise ValueError("statistics must map feature names to statistic objects")
        converted: dict[str, np.ndarray] = {}
        for statistic, raw in feature_stats.items():
            if statistic == "count":
                converted[statistic] = np.asarray(raw, dtype=np.int64)
            else:
                converted[statistic] = np.asarray(raw, dtype=np.float32)
        result[feature] = converted
    return result


def validate_statistics(
    value: dict[str, Any],
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE,
) -> None:
    expected = {
        profile.state_key: profile.state_dimension,
        profile.action_key: profile.action_dimension,
    }
    if set(value) != set(expected):
        raise ValueError("processor statistics must contain only project state and action")
    for key, dimension in expected.items():
        feature = value[key]
        if not isinstance(feature, dict):
            raise ValueError(f"statistics for {key!r} must be an object")
        for required in ("min", "max", "mean", "std", "count"):
            if required not in feature:
                raise ValueError(f"statistics for {key!r} are missing {required!r}")
        for statistic in ("min", "max", "mean", "std"):
            array = np.asarray(feature[statistic], dtype=np.float64)
            if array.shape != (dimension,) or not np.isfinite(array).all():
                raise ValueError(f"statistics {key}.{statistic} have invalid shape or values")
        standard_deviation = np.asarray(feature["std"], dtype=np.float64)
        if np.any(standard_deviation <= 0.0):
            raise ValueError(f"statistics {key}.std contain a constant dimension")
        count = np.asarray(feature["count"])
        if count.shape != (1,) or int(count[0]) <= 0:
            raise ValueError(f"statistics {key}.count is invalid")


def build_project_config(
    model_dir: Path,
    vlm_dir: Path,
    *,
    device: str,
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE,
) -> Any:
    """Rebind the pinned checkpoint architecture to project feature dimensions."""

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = PreTrainedConfig.from_pretrained(model_dir, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"expected SmolVLAConfig, got {type(config).__name__}")
    if config.max_state_dim < profile.state_dimension:
        raise ValueError("base model cannot hold the project state dimension")
    if config.max_action_dim < profile.action_dimension:
        raise ValueError("base model cannot hold the project action dimension")
    if config.chunk_size != profile.chunk_size or config.n_action_steps != profile.chunk_size:
        raise ValueError("base model action chunk differs from the project profile")

    inputs, outputs = profile.policy_features()
    # The model weights use fixed 32D projections, so rebinding 9D/7D feature metadata
    # changes only validated unpadding and processor behavior, not checkpoint tensor shapes.
    config.input_features = inputs
    config.output_features = outputs
    config.vlm_model_name = str(vlm_dir)
    config.load_vlm_weights = False
    config.device = device
    config.push_to_hub = False
    config.repo_id = None
    config.empty_cameras = 0
    config.adapt_to_pi_aloha = False
    config.use_delta_joint_actions_aloha = False
    config.validate_features()
    return config


@dataclass(slots=True)
class ProjectProcessors:
    """Validated wrapper around the LeRobot SmolVLA processor pipelines."""

    preprocessor: Any
    postprocessor: Any
    statistics: dict[str, Any]
    statistics_scope: str
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE

    def __post_init__(self) -> None:
        validate_statistics(self.statistics, self.profile)
        if self.statistics_scope not in ("train_split", "full_corpus"):
            raise ValueError("unsupported processor statistics scope")

    @property
    def statistics_sha256(self) -> str:
        return canonical_json_sha256(self.statistics)

    def _reject_privileged_or_unknown_observations(self, batch: dict[str, Any]) -> None:
        allowed = {
            self.profile.state_key,
            self.profile.image_key,
            f"{self.profile.state_key}_is_pad",
            f"{self.profile.image_key}_is_pad",
        }
        observed = {key for key in batch if key.startswith("observation.")}
        unexpected = observed - allowed
        if unexpected:
            raise ValueError(f"unexpected or privileged policy observations: {sorted(unexpected)}")

    def preprocess_inference(self, sample: dict[str, Any]) -> dict[str, Any]:
        self._reject_privileged_or_unknown_observations(sample)
        required = {self.profile.state_key, self.profile.image_key, "task"}
        missing = required - set(sample)
        if missing:
            raise ValueError(f"inference sample is missing {sorted(missing)}")
        if self.profile.action_key in sample:
            raise ValueError("ground-truth action may not enter the inference preprocessor")
        state = sample[self.profile.state_key]
        image = sample[self.profile.image_key]
        task = sample["task"]
        if not isinstance(state, torch.Tensor) or tuple(state.shape) != (
            self.profile.state_dimension,
        ):
            raise ValueError("inference state must be one unbatched 9D tensor")
        if not isinstance(image, torch.Tensor) or tuple(image.shape) != self.profile.image_shape:
            raise ValueError("inference image must be one unbatched CHW tensor")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("inference task must be a non-empty string")
        if not torch.isfinite(state).all() or not torch.isfinite(image).all():
            raise ValueError("inference observation contains non-finite values")
        if image.min().item() < 0.0 or image.max().item() > 1.0:
            raise ValueError("decoded RGB values must remain in [0, 1]")
        processed = self.preprocessor(dict(sample))
        self.validate_processed_inference(processed)
        return processed

    def validate_processed_inference(self, batch: dict[str, Any]) -> None:
        state = batch[self.profile.state_key]
        image = batch[self.profile.image_key]
        tokens = batch.get("observation.language.tokens")
        mask = batch.get("observation.language.attention_mask")
        if tuple(state.shape) != (1, self.profile.state_dimension):
            raise ValueError("processed inference state has an unexpected shape")
        if tuple(image.shape) != (1, *self.profile.image_shape):
            raise ValueError("processed inference image has an unexpected shape")
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("processed language tokens are missing or unbatched")
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(tokens.shape):
            raise ValueError("processed language attention mask is invalid")
        for value in (state, image):
            if not torch.isfinite(value).all():
                raise ValueError("processed inference tensor contains non-finite values")

    def preprocess_training(self, batch: dict[str, Any]) -> dict[str, Any]:
        self._reject_privileged_or_unknown_observations(batch)
        required = {
            self.profile.state_key,
            self.profile.image_key,
            self.profile.action_key,
            "action_is_pad",
            "task",
        }
        missing = required - set(batch)
        if missing:
            raise ValueError(f"training batch is missing {sorted(missing)}")
        state = batch[self.profile.state_key]
        image = batch[self.profile.image_key]
        action = batch[self.profile.action_key]
        padding = batch["action_is_pad"]
        if not isinstance(state, torch.Tensor) or state.ndim not in (2, 3):
            raise ValueError("training state must be batched with an optional observation axis")
        if state.shape[-1] != self.profile.state_dimension:
            raise ValueError("training state dimension is invalid")
        if not isinstance(image, torch.Tensor) or image.ndim not in (4, 5):
            raise ValueError("training image must be batched with an optional observation axis")
        if tuple(image.shape[-3:]) != self.profile.image_shape:
            raise ValueError("training image shape is invalid")
        expected_action = (state.shape[0], self.profile.chunk_size, self.profile.action_dimension)
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != expected_action:
            raise ValueError("training action horizon must have shape (B, 50, 7)")
        if not isinstance(padding, torch.Tensor) or tuple(padding.shape) != expected_action[:2]:
            raise ValueError("action_is_pad must have shape (B, 50)")
        if padding.dtype != torch.bool:
            raise ValueError("action_is_pad must be boolean")
        for key in (
            f"{self.profile.state_key}_is_pad",
            f"{self.profile.image_key}_is_pad",
        ):
            temporal_padding = batch.get(key)
            if temporal_padding is not None and (
                not isinstance(temporal_padding, torch.Tensor)
                or temporal_padding.dtype != torch.bool
                or tuple(temporal_padding.shape) != (state.shape[0], 1)
            ):
                raise ValueError(f"{key} must have shape (B, 1) and boolean dtype")
        if not all(torch.isfinite(value).all() for value in (state, image, action)):
            raise ValueError("training batch contains non-finite tensors")
        processed = self.preprocessor(dict(batch))
        if tuple(processed[self.profile.action_key].shape) != expected_action:
            raise ValueError("processed training action horizon changed shape")
        if tuple(processed["action_is_pad"].shape) != expected_action[:2]:
            raise ValueError("processed action padding mask changed shape")
        return processed

    def postprocess(self, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        if action.ndim not in (2, 3) or action.shape[-1] != self.profile.action_dimension:
            raise ValueError("model action must have shape (B, 7) or (B, T, 7)")
        result = self.postprocessor(action)
        if not isinstance(result, torch.Tensor) or tuple(result.shape) != tuple(action.shape):
            raise ValueError("postprocessor changed the action shape")
        if result.device.type != "cpu":
            raise ValueError("postprocessed action must be on CPU")
        finite = bool(torch.isfinite(result).all().item())
        if not finite:
            raise ValueError("postprocessed action contains non-finite values")
        violations = torch.logical_or(result < -1.0, result > 1.0)
        diagnostics: dict[str, object] = {
            "finite": finite,
            "bound_violation_count": int(violations.sum().item()),
            "bound_violation_rate": float(violations.float().mean().item()),
            "minimum": float(result.min().item()),
            "maximum": float(result.max().item()),
        }
        return result, diagnostics

    def save(self, directory: Path, *, provenance: dict[str, object]) -> Path:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"refusing to overwrite processor artifacts: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        self.preprocessor.save_pretrained(
            directory,
            config_filename=PREPROCESSOR_CONFIG_NAME,
        )
        self.postprocessor.save_pretrained(
            directory,
            config_filename=POSTPROCESSOR_CONFIG_NAME,
        )
        atomic_write_json(directory / PROCESSOR_STATS_NAME, self.statistics)
        manifest = {
            "schema_version": PROCESSOR_MANIFEST_SCHEMA_VERSION,
            "profile": self.profile.to_dict(),
            "statistics_scope": self.statistics_scope,
            "statistics_sha256": self.statistics_sha256,
            "provenance": provenance,
        }
        atomic_write_json(directory / PROCESSOR_MANIFEST_NAME, manifest)
        return directory


def build_project_processors(
    config: Any,
    statistics: dict[str, Any],
    *,
    statistics_scope: str,
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE,
) -> ProjectProcessors:
    from lerobot.policies import make_pre_post_processors

    validate_statistics(statistics, profile)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=tensor_ready_statistics(statistics),
    )
    return ProjectProcessors(
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        statistics=statistics,
        statistics_scope=statistics_scope,
        profile=profile,
    )


def load_project_processors(
    directory: Path,
    *,
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE,
) -> ProjectProcessors:
    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    manifest = load_json_object(directory / PROCESSOR_MANIFEST_NAME)
    if manifest.get("schema_version") != PROCESSOR_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported processor artifact schema version")
    if manifest.get("profile") != profile.to_dict():
        raise ValueError("processor artifact profile differs from the project profile")
    statistics = load_json_object(directory / PROCESSOR_STATS_NAME)
    validate_statistics(statistics, profile)
    if manifest.get("statistics_sha256") != canonical_json_sha256(statistics):
        raise ValueError("processor statistics hash does not match the manifest")
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        directory,
        config_filename=PREPROCESSOR_CONFIG_NAME,
        local_files_only=True,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        directory,
        config_filename=POSTPROCESSOR_CONFIG_NAME,
        local_files_only=True,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return ProjectProcessors(
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        statistics=statistics,
        statistics_scope=manifest.get("statistics_scope"),
        profile=profile,
    )


__all__ = [
    "POSTPROCESSOR_CONFIG_NAME",
    "PREPROCESSOR_CONFIG_NAME",
    "PROCESSOR_MANIFEST_NAME",
    "PROCESSOR_MANIFEST_SCHEMA_VERSION",
    "PROCESSOR_STATS_NAME",
    "ProjectProcessors",
    "atomic_write_json",
    "build_project_config",
    "build_project_processors",
    "canonical_json_sha256",
    "compute_policy_statistics",
    "load_json_object",
    "load_project_processors",
    "sha256_file",
    "tensor_ready_statistics",
    "validate_statistics",
]
