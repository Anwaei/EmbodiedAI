"""Convert validated immutable contract episodes into a local LeRobotDataset."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from embodied_ai.contracts import EpisodeMetadata, EpisodeOutcome
from embodied_ai.sim.recording import validate_npy_episode

from .lerobot_mapping import (
    ContractLeRobotMapping,
    FRANKA_PICK_PLACE_LEROBOT_MAPPING,
    LEROBOT_ACTION_KEY,
    LEROBOT_STATE_KEY,
)

LEROBOT_CONVERSION_SCHEMA_VERSION = "embodied-ai.lerobot-conversion/v1"
CONVERSION_MANIFEST_PATH = Path("meta/embodied_ai_conversion.json")


@dataclass(frozen=True, slots=True)
class ConvertedLeRobotDataset:
    """Summary of one atomically published local conversion."""

    root: Path
    repo_id: str
    episode_count: int
    frame_count: int
    fps: int
    use_videos: bool


@dataclass(frozen=True, slots=True)
class _ValidatedSourceEpisode:
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
    # Keep LeRobot's PyTorch-heavy import out of mapping-only tools and tests.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def _integer_fps(control_hz: float) -> int:
    rounded = round(control_hz)
    if not math.isclose(control_hz, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"LeRobot requires an integer fps, got control_hz={control_hz}")
    if rounded <= 0:
        raise ValueError("LeRobot fps must be positive")
    return rounded


def _validate_source_episode(
    episode_directory: str | Path,
    mapping: ContractLeRobotMapping,
) -> _ValidatedSourceEpisode:
    root = Path(episode_directory).expanduser().resolve()
    metadata = validate_npy_episode(root)
    mapping.validate_episode(metadata)
    if metadata.outcome is not EpisodeOutcome.SUCCESS:
        raise ValueError(
            f"training conversion requires a successful episode: {metadata.episode_id}"
        )
    if metadata.instruction is None or metadata.expert is None:
        raise ValueError(
            f"training episode lacks instruction/expert metadata: {metadata.episode_id}"
        )

    timestamps = _load_npy(root / "observations/timestamps_ns.npy")
    period_ns = round(1_000_000_000 / metadata.action_schema.control_hz)
    relative_timestamps = timestamps - timestamps[0]
    expected_timestamps = np.arange(metadata.step_count, dtype=np.int64) * period_ns
    if not np.array_equal(relative_timestamps, expected_timestamps):
        raise ValueError(
            f"episode timestamps are not a regular {metadata.action_schema.control_hz:g} Hz grid: "
            f"{metadata.episode_id}"
        )

    actions = _load_npy(root / "actions/data.npy")
    if not np.isfinite(actions).all():
        raise ValueError(f"episode actions contain non-finite values: {metadata.episode_id}")
    for index, component in enumerate(metadata.action_schema.components):
        below = np.any(actions[:, index] < component.lower)
        above = np.any(actions[:, index] > component.upper)
        if below or above:
            raise ValueError(
                f"episode action component {component.name!r} exceeds contract bounds: "
                f"{metadata.episode_id}"
            )

    for source_key in mapping.state_source_keys:
        values = _load_npy(root / _observation_payload_path(source_key))
        if not np.isfinite(values).all():
            raise ValueError(
                f"episode state observation {source_key!r} contains non-finite values: "
                f"{metadata.episode_id}"
            )

    return _ValidatedSourceEpisode(
        root=root,
        metadata=metadata,
        manifest_sha256=_sha256(root / "manifest.json"),
    )


def _source_record(source: _ValidatedSourceEpisode, episode_index: int) -> dict[str, object]:
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


def _write_conversion_manifest(
    root: Path,
    *,
    repo_id: str,
    mapping: ContractLeRobotMapping,
    use_videos: bool,
    fps: int,
    sources: tuple[_ValidatedSourceEpisode, ...],
) -> None:
    manifest = {
        "schema_version": LEROBOT_CONVERSION_SCHEMA_VERSION,
        "repo_id": repo_id,
        "lerobot_version": version("lerobot"),
        "storage": "video" if use_videos else "image",
        "fps": fps,
        "episode_count": len(sources),
        "frame_count": sum(source.metadata.step_count for source in sources),
        "mapping": mapping.to_dict(),
        "source_episodes": [
            _source_record(source, episode_index)
            for episode_index, source in enumerate(sources)
        ],
    }
    path = root / CONVERSION_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_source_episode(
    dataset: Any,
    source: _ValidatedSourceEpisode,
    mapping: ContractLeRobotMapping,
) -> None:
    metadata = source.metadata
    assert metadata.instruction is not None
    state_arrays = {
        key: _load_npy(source.root / _observation_payload_path(key))
        for key in mapping.state_source_keys
    }
    image_arrays = {
        image.source_key: _load_npy(source.root / _observation_payload_path(image.source_key))
        for image in mapping.image_features
    }
    actions = _load_npy(source.root / "actions/data.npy")

    for frame_index in range(metadata.step_count):
        state_parts = [
            np.asarray(state_arrays[key][frame_index], dtype=np.float32)
            for key in mapping.state_source_keys
        ]
        state = state_parts[0].copy() if len(state_parts) == 1 else np.concatenate(state_parts)
        frame: dict[str, object] = {
            LEROBOT_STATE_KEY: state,
            LEROBOT_ACTION_KEY: np.asarray(actions[frame_index]).copy(),
            # LeRobot records language as a per-frame task index; the exact episode
            # instruction is also retained in the conversion provenance sidecar.
            "task": metadata.instruction,
        }
        for image_mapping in mapping.image_features:
            frame[image_mapping.target_key] = np.asarray(
                image_arrays[image_mapping.source_key][frame_index]
            )
        dataset.add_frame(frame)

    # One contract directory is always one LeRobot episode; never merge episode boundaries.
    dataset.save_episode(parallel_encoding=False)


def convert_contract_episodes_to_lerobot(
    episode_directories: tuple[str | Path, ...],
    output_root: str | Path,
    *,
    repo_id: str,
    mapping: ContractLeRobotMapping = FRANKA_PICK_PLACE_LEROBOT_MAPPING,
    use_videos: bool = True,
) -> ConvertedLeRobotDataset:
    """Validate, convert, finalize, reopen, and atomically publish a local dataset."""

    if not episode_directories:
        raise ValueError("at least one source episode is required")
    if not repo_id or any(character.isspace() for character in repo_id):
        raise ValueError("repo_id must be a non-empty identifier without whitespace")

    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing dataset: {output}")

    sources = tuple(
        _validate_source_episode(episode_directory, mapping)
        for episode_directory in episode_directories
    )
    source_roots = [source.root for source in sources]
    source_ids = [source.metadata.episode_id for source in sources]
    if len(set(source_roots)) != len(source_roots):
        raise ValueError("source episode directories must be unique")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source episode identifiers must be unique")

    fps = _integer_fps(mapping.action_schema.control_hz)
    frame_count = sum(source.metadata.step_count for source in sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    dataset_class = _lerobot_dataset_class()

    try:
        dataset = dataset_class.create(
            repo_id=repo_id,
            fps=fps,
            root=partial,
            robot_type=mapping.robot,
            features=mapping.lerobot_features(use_videos=use_videos),
            use_videos=use_videos,
            image_writer_processes=0,
            image_writer_threads=0,
            encoder_threads=1,
            metadata_buffer_size=1,
            batch_encoding_size=1,
            streaming_encoding=False,
        )
        for source in sources:
            _add_source_episode(dataset, source, mapping)
        dataset.finalize()

        # Reopen before publication so an incomplete Parquet footer or metadata flush
        # can never appear at the final immutable destination.
        reloaded = dataset_class(repo_id, root=partial)
        if len(reloaded) != frame_count or reloaded.num_episodes != len(sources):
            raise RuntimeError(
                "reloaded LeRobotDataset counts do not match the source conversion: "
                f"frames={len(reloaded)}, episodes={reloaded.num_episodes}"
            )
        _write_conversion_manifest(
            partial,
            repo_id=repo_id,
            mapping=mapping,
            use_videos=use_videos,
            fps=fps,
            sources=sources,
        )
        if output.exists():
            raise FileExistsError(f"dataset destination appeared during conversion: {output}")
        partial.rename(output)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    return ConvertedLeRobotDataset(
        root=output,
        repo_id=repo_id,
        episode_count=len(sources),
        frame_count=frame_count,
        fps=fps,
        use_videos=use_videos,
    )


__all__ = [
    "CONVERSION_MANIFEST_PATH",
    "ConvertedLeRobotDataset",
    "LEROBOT_CONVERSION_SCHEMA_VERSION",
    "convert_contract_episodes_to_lerobot",
]
