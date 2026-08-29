"""Validate a converted LeRobotDataset against immutable contract episodes."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embodied_ai.contracts import EpisodeMetadata, EpisodeOutcome
from embodied_ai.sim.recording import validate_npy_episode

from .lerobot_converter import (
    CONVERSION_MANIFEST_PATH,
    LEROBOT_CONVERSION_SCHEMA_VERSION,
)
from .lerobot_mapping import (
    ContractLeRobotMapping,
    FRANKA_PICK_PLACE_LEROBOT_MAPPING,
    LEROBOT_ACTION_KEY,
    LEROBOT_STATE_KEY,
)

LEROBOT_VALIDATION_SCHEMA_VERSION = "embodied-ai.lerobot-validation/v1"


@dataclass(frozen=True, slots=True)
class NormalizationInputSummary:
    """Recomputed statistics for one policy normalization input."""

    feature: str
    count: int
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    constant_dimensions: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "count": self.count,
            "min": list(self.minimum),
            "max": list(self.maximum),
            "mean": list(self.mean),
            "std": list(self.standard_deviation),
            "constant_dimensions": list(self.constant_dimensions),
        }


@dataclass(frozen=True, slots=True)
class LeRobotDatasetValidationReport:
    """Serializable proof that one local dataset passed the Stage 7 gates."""

    dataset_root: Path
    repo_id: str
    mapping_profile: str
    storage: str
    fps: int
    episode_count: int
    frame_count: int
    task_count: int
    instructions: tuple[str, ...]
    decoded_image_samples: int
    deterministic_reload_samples: int
    conversion_manifest_sha256: str
    source_manifest_sha256: tuple[str, ...]
    table_fingerprint_sha256: str
    reload_fingerprint_sha256: str
    normalization_inputs: tuple[NormalizationInputSummary, ...]
    schema_version: str = LEROBOT_VALIDATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": "passed",
            "dataset_root": self.dataset_root.as_posix(),
            "repo_id": self.repo_id,
            "mapping_profile": self.mapping_profile,
            "storage": self.storage,
            "fps": self.fps,
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "task_count": self.task_count,
            "instructions": list(self.instructions),
            "decoded_image_samples": self.decoded_image_samples,
            "deterministic_reload_samples": self.deterministic_reload_samples,
            "conversion_manifest_sha256": self.conversion_manifest_sha256,
            "source_manifest_sha256": list(self.source_manifest_sha256),
            "table_fingerprint_sha256": self.table_fingerprint_sha256,
            "reload_fingerprint_sha256": self.reload_fingerprint_sha256,
            "normalization_inputs": [
                summary.to_dict() for summary in self.normalization_inputs
            ],
        }


@dataclass(frozen=True, slots=True)
class _SourceEpisode:
    root: Path
    metadata: EpisodeMetadata
    manifest_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observation_payload_path(key: str) -> Path:
    filename = key.replace(".", "_").replace("-", "_")
    return Path("observations") / f"{filename}.npy"


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _lerobot_dataset_class() -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def _load_json(path: Path, label: str) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_source(
    root: Path,
    mapping: ContractLeRobotMapping,
) -> _SourceEpisode:
    metadata = validate_npy_episode(root)
    mapping.validate_episode(metadata)
    if metadata.outcome is not EpisodeOutcome.SUCCESS:
        raise ValueError(f"source episode is not successful: {metadata.episode_id}")
    if metadata.instruction is None or metadata.expert is None:
        raise ValueError(f"source episode lacks training metadata: {metadata.episode_id}")
    return _SourceEpisode(
        root=root,
        metadata=metadata,
        manifest_sha256=_sha256(root / "manifest.json"),
    )


def _expected_source_record(source: _SourceEpisode, episode_index: int) -> dict[str, object]:
    metadata = source.metadata
    assert metadata.instruction is not None
    assert metadata.instruction_id is not None
    assert metadata.instruction_language is not None
    assert metadata.expert is not None
    return {
        "lerobot_episode_index": episode_index,
        "source_episode_id": metadata.episode_id,
        "source_manifest_sha256": source.manifest_sha256,
        "step_count": metadata.step_count,
        "start_time_ns": metadata.start_time_ns,
        "end_time_ns": metadata.end_time_ns,
        "task": metadata.task,
        "task_parameters": metadata.task_parameters,
        "reset_parameters": metadata.reset_parameters,
        "instruction": metadata.instruction,
        "instruction_id": metadata.instruction_id,
        "instruction_language": metadata.instruction_language,
        "expert": metadata.expert.to_dict(),
        "source_provenance": metadata.provenance.to_dict(),
    }


def _validate_conversion_manifest(
    root: Path,
    *,
    repo_id: str,
    sources_by_id: dict[str, _SourceEpisode],
    mapping: ContractLeRobotMapping,
) -> tuple[dict[str, object], tuple[_SourceEpisode, ...], bool, int]:
    manifest = _load_json(root / CONVERSION_MANIFEST_PATH, "conversion manifest")
    if manifest.get("schema_version") != LEROBOT_CONVERSION_SCHEMA_VERSION:
        raise ValueError("unsupported or missing conversion manifest schema version")
    if manifest.get("repo_id") != repo_id:
        raise ValueError("conversion manifest repo_id does not match the requested dataset")
    if manifest.get("mapping") != mapping.to_dict():
        raise ValueError("conversion mapping does not match the requested mapping profile")

    storage = manifest.get("storage")
    if storage not in ("image", "video"):
        raise ValueError("conversion storage must be 'image' or 'video'")
    use_videos = storage == "video"
    fps = manifest.get("fps")
    expected_fps = round(mapping.action_schema.control_hz)
    if isinstance(fps, bool) or not isinstance(fps, int) or fps != expected_fps:
        raise ValueError("conversion fps does not match the action contract")

    records = manifest.get("source_episodes")
    if not isinstance(records, list) or not records:
        raise ValueError("conversion manifest must contain source episode records")
    ordered_sources: list[_SourceEpisode] = []
    seen_ids: set[str] = set()
    for episode_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("conversion source episode record must be an object")
        source_id = record.get("source_episode_id")
        if not isinstance(source_id, str) or source_id not in sources_by_id:
            raise ValueError(f"conversion references an unknown source episode: {source_id!r}")
        if source_id in seen_ids:
            raise ValueError(f"conversion repeats source episode: {source_id}")
        source = sources_by_id[source_id]
        if record != _expected_source_record(source, episode_index):
            raise ValueError(f"conversion provenance differs from source: {source_id}")
        ordered_sources.append(source)
        seen_ids.add(source_id)
    if seen_ids != set(sources_by_id):
        raise ValueError("source episode set differs from conversion provenance")

    expected_frames = sum(source.metadata.step_count for source in ordered_sources)
    if manifest.get("episode_count") != len(ordered_sources):
        raise ValueError("conversion episode_count does not match its source records")
    if manifest.get("frame_count") != expected_frames:
        raise ValueError("conversion frame_count does not match its source records")
    return manifest, tuple(ordered_sources), use_videos, fps


def _feature_shape(feature: dict[str, object]) -> tuple[int, ...]:
    shape = feature.get("shape")
    if not isinstance(shape, (tuple, list)):
        raise ValueError("LeRobot feature shape is missing")
    return tuple(int(value) for value in shape)


def _validate_features(
    features: dict[str, dict[str, object]],
    mapping: ContractLeRobotMapping,
    *,
    use_videos: bool,
) -> None:
    expected = mapping.lerobot_features(use_videos=use_videos)
    for key, expected_feature in expected.items():
        actual = features.get(key)
        if actual is None:
            raise ValueError(f"LeRobotDataset is missing feature {key!r}")
        if actual.get("dtype") != expected_feature["dtype"]:
            raise ValueError(f"LeRobot feature {key!r} has an unexpected dtype")
        if _feature_shape(actual) != tuple(expected_feature["shape"]):
            raise ValueError(f"LeRobot feature {key!r} has an unexpected shape")
        if actual.get("names") != expected_feature["names"]:
            raise ValueError(f"LeRobot feature {key!r} has unexpected component names")


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _raw_column(dataset: Any, key: str) -> np.ndarray:
    values = dataset.hf_dataset[key]
    return np.stack([_as_numpy(value) for value in values])


def _source_policy_arrays(
    sources: tuple[_SourceEpisode, ...],
    mapping: ContractLeRobotMapping,
) -> tuple[np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for source in sources:
        state_parts = [
            np.asarray(_load_npy(source.root / _observation_payload_path(key)))
            for key in mapping.state_source_keys
        ]
        state = state_parts[0] if len(state_parts) == 1 else np.concatenate(state_parts, axis=1)
        states.append(np.asarray(state, dtype=np.float32))
        actions.append(np.asarray(_load_npy(source.root / "actions/data.npy")))
    return np.concatenate(states), np.concatenate(actions)


def _validate_table(
    dataset: Any,
    sources: tuple[_SourceEpisode, ...],
    mapping: ContractLeRobotMapping,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], str]:
    state = _raw_column(dataset, LEROBOT_STATE_KEY).astype(np.float32, copy=False)
    action = _raw_column(dataset, LEROBOT_ACTION_KEY).astype(np.float32, copy=False)
    timestamp = _raw_column(dataset, "timestamp").reshape(-1)
    frame_index = _raw_column(dataset, "frame_index").reshape(-1)
    episode_index = _raw_column(dataset, "episode_index").reshape(-1)
    global_index = _raw_column(dataset, "index").reshape(-1)
    task_index = _raw_column(dataset, "task_index").reshape(-1)
    expected_state, expected_action = _source_policy_arrays(sources, mapping)
    if not np.array_equal(state, expected_state):
        raise ValueError("LeRobot observation.state differs from the contract source")
    if not np.array_equal(action, expected_action):
        raise ValueError("LeRobot action differs from the contract source")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("policy normalization inputs contain non-finite values")

    tasks = dataset.meta.tasks
    task_lookup = {str(task): int(row["task_index"]) for task, row in tasks.iterrows()}
    expected_instructions = {
        source.metadata.instruction for source in sources if source.metadata.instruction is not None
    }
    if set(task_lookup) != expected_instructions:
        raise ValueError("LeRobot task table differs from source instructions")

    sample_indices: list[int] = []
    offset = 0
    for expected_episode_index, source in enumerate(sources):
        metadata = source.metadata
        assert metadata.instruction is not None
        stop = offset + metadata.step_count
        selection = slice(offset, stop)
        expected_timestamp = np.arange(metadata.step_count, dtype=np.float64) / dataset.fps
        if not np.allclose(timestamp[selection], expected_timestamp, rtol=0.0, atol=1e-6):
            raise ValueError(f"LeRobot timestamps are invalid for episode {metadata.episode_id}")
        if not np.array_equal(frame_index[selection], np.arange(metadata.step_count)):
            raise ValueError(f"LeRobot frame_index is invalid for episode {metadata.episode_id}")
        if not np.all(episode_index[selection] == expected_episode_index):
            raise ValueError(f"LeRobot episode_index is invalid for {metadata.episode_id}")
        if not np.all(task_index[selection] == task_lookup[metadata.instruction]):
            raise ValueError(f"LeRobot task mapping is invalid for {metadata.episode_id}")
        # Decode both boundaries so every episode/video segment participates in validation.
        sample_indices.extend((offset, stop - 1))
        offset = stop
    if not np.array_equal(global_index, np.arange(offset)):
        raise ValueError("LeRobot global frame indices are not contiguous")

    digest = hashlib.sha256()
    for key, values in (
        (LEROBOT_STATE_KEY, state),
        (LEROBOT_ACTION_KEY, action),
        ("timestamp", timestamp),
        ("frame_index", frame_index),
        ("episode_index", episode_index),
        ("index", global_index),
        ("task_index", task_index),
    ):
        digest.update(key.encode("utf-8"))
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return state, action, tuple(dict.fromkeys(sample_indices)), digest.hexdigest()


def _validate_episode_metadata(dataset: Any, sources: tuple[_SourceEpisode, ...]) -> None:
    if len(dataset.meta.episodes) != len(sources):
        raise ValueError("LeRobot episode metadata row count is incorrect")
    offset = 0
    for episode_index, source in enumerate(sources):
        row = dataset.meta.episodes[episode_index]
        metadata = source.metadata
        assert metadata.instruction is not None
        stop = offset + metadata.step_count
        expected = {
            "episode_index": episode_index,
            "length": metadata.step_count,
            "dataset_from_index": offset,
            "dataset_to_index": stop,
            "tasks": [metadata.instruction],
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"LeRobot episode metadata {key!r} is invalid for {metadata.episode_id}"
                )
        offset = stop


def _normalization_summary(
    dataset: Any,
    key: str,
    values: np.ndarray,
) -> NormalizationInputSummary:
    recomputed = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
    }
    stored = dataset.meta.stats.get(key)
    if not isinstance(stored, dict):
        raise ValueError(f"LeRobot normalization statistics are missing for {key!r}")
    count = np.asarray(stored.get("count")).reshape(-1)
    if count.size != 1 or int(count[0]) != values.shape[0]:
        raise ValueError(f"LeRobot normalization count is invalid for {key!r}")
    for statistic, expected in recomputed.items():
        actual = np.asarray(stored.get(statistic))
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f"LeRobot normalization {statistic} is invalid for {key!r}")
        if not np.allclose(actual, expected, rtol=2e-5, atol=2e-6):
            raise ValueError(f"LeRobot normalization {statistic} differs for {key!r}")
    constant = tuple(int(index) for index in np.flatnonzero(recomputed["std"] <= 1e-8))
    return NormalizationInputSummary(
        feature=key,
        count=values.shape[0],
        minimum=tuple(float(value) for value in recomputed["min"]),
        maximum=tuple(float(value) for value in recomputed["max"]),
        mean=tuple(float(value) for value in recomputed["mean"]),
        standard_deviation=tuple(float(value) for value in recomputed["std"]),
        constant_dimensions=constant,
    )


def _sample_fingerprint(
    first: Any,
    second: Any,
    sample_indices: tuple[int, ...],
    image_keys: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    compared_keys = (
        *image_keys,
        LEROBOT_STATE_KEY,
        LEROBOT_ACTION_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    )
    for index in sample_indices:
        first_sample = first[index]
        second_sample = second[index]
        if first_sample["task"] != second_sample["task"]:
            raise ValueError(f"task text changed after deterministic reload at frame {index}")
        digest.update(str(index).encode("ascii"))
        digest.update(first_sample["task"].encode("utf-8"))
        for key in compared_keys:
            first_value = _as_numpy(first_sample[key])
            second_value = _as_numpy(second_sample[key])
            if first_value.shape != second_value.shape or not np.array_equal(
                first_value, second_value
            ):
                raise ValueError(f"LeRobot reload is not deterministic for {key!r} at frame {index}")
            if key in image_keys:
                expected_shape = _feature_shape(first.features[key])
                if first_value.shape != expected_shape:
                    raise ValueError(f"decoded image {key!r} has an invalid shape")
                if not np.isfinite(first_value).all():
                    raise ValueError(f"decoded image {key!r} contains non-finite values")
                if first_value.min() < 0.0 or first_value.max() > 1.0:
                    raise ValueError(f"decoded image {key!r} is outside the [0, 1] range")
            contiguous = np.ascontiguousarray(first_value)
            digest.update(key.encode("utf-8"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_lerobot_dataset(
    dataset_root: str | Path,
    source_episode_directories: tuple[str | Path, ...],
    *,
    repo_id: str,
    mapping: ContractLeRobotMapping = FRANKA_PICK_PLACE_LEROBOT_MAPPING,
) -> LeRobotDatasetValidationReport:
    """Validate provenance, full table contents, media samples, stats, and reloads."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LeRobotDataset root does not exist: {root}")
    if not source_episode_directories:
        raise ValueError("at least one source episode is required")

    sources = tuple(
        _validate_source(Path(path).expanduser().resolve(), mapping)
        for path in source_episode_directories
    )
    sources_by_id = {source.metadata.episode_id: source for source in sources}
    if len(sources_by_id) != len(sources):
        raise ValueError("source episode identifiers must be unique")
    _, ordered_sources, use_videos, fps = _validate_conversion_manifest(
        root,
        repo_id=repo_id,
        sources_by_id=sources_by_id,
        mapping=mapping,
    )

    dataset_class = _lerobot_dataset_class()
    first = dataset_class(repo_id, root=root)
    expected_frames = sum(source.metadata.step_count for source in ordered_sources)
    if first.num_episodes != len(ordered_sources) or len(first) != expected_frames:
        raise ValueError("LeRobotDataset episode/frame counts differ from the source corpus")
    if first.fps != fps:
        raise ValueError("LeRobotDataset fps differs from conversion provenance")
    _validate_features(first.features, mapping, use_videos=use_videos)
    _validate_episode_metadata(first, ordered_sources)
    state, action, sample_indices, table_fingerprint = _validate_table(
        first, ordered_sources, mapping
    )
    normalization = (
        _normalization_summary(first, LEROBOT_STATE_KEY, state),
        _normalization_summary(first, LEROBOT_ACTION_KEY, action),
    )

    # A fresh object exercises metadata reload, Parquet reload, task lookup, and media decode.
    second = dataset_class(repo_id, root=root)
    if (
        len(second) != len(first)
        or second.num_episodes != first.num_episodes
        or second.fps != first.fps
        or second.features != first.features
    ):
        raise ValueError("LeRobotDataset metadata changed across independent reloads")
    image_keys = tuple(image.target_key for image in mapping.image_features)
    reload_fingerprint = _sample_fingerprint(first, second, sample_indices, image_keys)

    instructions = tuple(sorted({source.metadata.instruction for source in ordered_sources}))
    assert all(instruction is not None for instruction in instructions)
    return LeRobotDatasetValidationReport(
        dataset_root=root,
        repo_id=repo_id,
        mapping_profile=mapping.profile,
        storage="video" if use_videos else "image",
        fps=fps,
        episode_count=len(ordered_sources),
        frame_count=expected_frames,
        task_count=len(instructions),
        instructions=instructions,
        decoded_image_samples=len(sample_indices) * len(image_keys),
        deterministic_reload_samples=len(sample_indices),
        conversion_manifest_sha256=_sha256(root / CONVERSION_MANIFEST_PATH),
        source_manifest_sha256=tuple(source.manifest_sha256 for source in ordered_sources),
        table_fingerprint_sha256=table_fingerprint,
        reload_fingerprint_sha256=reload_fingerprint,
        normalization_inputs=normalization,
    )


def write_validation_report(
    report: LeRobotDatasetValidationReport,
    report_path: str | Path,
) -> Path:
    """Atomically replace the derived validation report for repeatable reruns."""

    path = Path(report_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "LEROBOT_VALIDATION_SCHEMA_VERSION",
    "LeRobotDatasetValidationReport",
    "NormalizationInputSummary",
    "validate_lerobot_dataset",
    "write_validation_report",
]
