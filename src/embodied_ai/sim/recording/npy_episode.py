"""Bounded in-memory NPY episode recorder for simulator smoke and short rollouts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from embodied_ai.contracts import (
    ActionSchema,
    EpisodeMetadata,
    EpisodeOutcome,
    EpisodeProvenance,
    ExpertMetadata,
    ObservationSchema,
    PayloadFile,
)

_EPISODE_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_NPY_MEDIA_TYPE = "application/x-npy"


@dataclass(frozen=True, slots=True)
class RecordedEpisode:
    """Published directory and the manifest object written inside it."""

    directory: Path
    metadata: EpisodeMetadata


def _observation_payload_path(key: str) -> str:
    filename = key.replace(".", "_").replace("-", "_")
    return f"observations/{filename}.npy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_metadata(root: Path, relative_path: str) -> PayloadFile:
    path = root / PurePosixPath(relative_path)
    return PayloadFile(
        path=relative_path,
        media_type=_NPY_MEDIA_TYPE,
        byte_size=path.stat().st_size,
        sha256=_sha256(path),
    )


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _write_manifest(path: Path, metadata: EpisodeMetadata) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class NpyEpisodeRecorder:
    """Collect synchronized contract values and atomically publish one episode."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        episode_id: str,
        task: str,
        robot: str,
        scene: str,
        random_seed: int,
        observation_schema: ObservationSchema,
        action_schema: ActionSchema,
        provenance: EpisodeProvenance,
        task_parameters: Mapping[str, object] | None = None,
        reset_parameters: Mapping[str, object] | None = None,
        instruction: str | None = None,
        instruction_id: str | None = None,
        instruction_language: str | None = None,
        expert: ExpertMetadata | None = None,
    ) -> None:
        if not _EPISODE_ID_RE.fullmatch(episode_id):
            raise ValueError("episode_id is not a contract-compatible identifier")
        self.output_root = Path(output_root).expanduser().resolve()
        self.episode_id = episode_id
        self.task = task
        self.robot = robot
        self.scene = scene
        self.random_seed = random_seed
        self.observation_schema = observation_schema
        self.action_schema = action_schema
        self.provenance = provenance
        self.task_parameters = task_parameters
        self.reset_parameters = reset_parameters
        self.instruction = instruction
        self.instruction_id = instruction_id
        self.instruction_language = instruction_language
        self.expert = expert
        self.final_directory = self.output_root / episode_id
        if self.final_directory.exists():
            raise FileExistsError(
                f"immutable episode directory already exists: {self.final_directory}"
            )

        payload_paths = [
            _observation_payload_path(field.key)
            for field in self.observation_schema.fields
        ]
        if len(set(payload_paths)) != len(payload_paths):
            raise ValueError("observation keys collide after NPY filename normalization")

        self._observations: dict[str, list[np.ndarray]] = {
            field.key: [] for field in self.observation_schema.fields
        }
        self._actions: list[np.ndarray] = []
        self._timestamps_ns: list[int] = []
        self._finalized = False

    @property
    def step_count(self) -> int:
        return len(self._actions)

    def append(
        self,
        observation: Mapping[str, np.ndarray],
        action: np.ndarray,
        timestamp_ns: int,
    ) -> None:
        """Append one pre-action observation/action pair on the simulation clock."""

        if self._finalized:
            raise RuntimeError("cannot append to a finalized episode")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise ValueError("timestamp_ns must be an integer")
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self._timestamps_ns and timestamp_ns <= self._timestamps_ns[-1]:
            raise ValueError("timestamps must be strictly increasing")

        expected_keys = {field.key for field in self.observation_schema.fields}
        if set(observation) != expected_keys:
            raise ValueError("observation keys must exactly match the observation schema")

        converted_observations: dict[str, np.ndarray] = {}
        for field in self.observation_schema.fields:
            value = np.asarray(observation[field.key])
            expected_dtype = np.dtype(field.dtype.value)
            if value.shape != field.shape:
                raise ValueError(
                    f"observation {field.key!r} has shape {value.shape}, "
                    f"expected {field.shape}"
                )
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"observation {field.key!r} has dtype {value.dtype}, "
                    f"expected {expected_dtype}"
                )
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"observation {field.key!r} contains non-finite values")
            converted_observations[field.key] = np.array(value, copy=True)

        action_value = np.asarray(action)
        expected_action_shape = (self.action_schema.dimension,)
        expected_action_dtype = np.dtype(self.action_schema.dtype.value)
        if action_value.shape != expected_action_shape:
            raise ValueError(
                f"action has shape {action_value.shape}, expected {expected_action_shape}"
            )
        if action_value.dtype != expected_action_dtype:
            raise ValueError(
                f"action has dtype {action_value.dtype}, expected {expected_action_dtype}"
            )
        if not np.isfinite(action_value).all():
            raise ValueError("action contains non-finite values")
        for index, component in enumerate(self.action_schema.components):
            if not component.lower <= action_value[index] <= component.upper:
                raise ValueError(f"action component {component.name!r} is out of bounds")

        for key, value in converted_observations.items():
            self._observations[key].append(value)
        self._actions.append(np.array(action_value, copy=True))
        self._timestamps_ns.append(timestamp_ns)

    def finalize(
        self,
        outcome: EpisodeOutcome,
        termination_reason: str | None,
    ) -> RecordedEpisode:
        """Write payloads, then manifest, and publish via a same-filesystem rename."""

        if self._finalized:
            raise RuntimeError("episode is already finalized")
        if not self._actions:
            raise ValueError("cannot finalize an empty episode")
        if self.final_directory.exists():
            raise FileExistsError(
                f"immutable episode directory already exists: {self.final_directory}"
            )

        self.output_root.mkdir(parents=True, exist_ok=True)
        partial_directory = self.output_root / (
            f".{self.episode_id}.partial-{uuid.uuid4().hex}"
        )
        partial_directory.mkdir()
        try:
            payloads: list[PayloadFile] = []
            for field in self.observation_schema.fields:
                relative_path = _observation_payload_path(field.key)
                _write_npy(
                    partial_directory / PurePosixPath(relative_path),
                    np.stack(self._observations[field.key]),
                )
                payloads.append(_payload_metadata(partial_directory, relative_path))

            timestamps = np.asarray(self._timestamps_ns, dtype=np.int64)
            observation_timestamps_path = "observations/timestamps_ns.npy"
            _write_npy(
                partial_directory / PurePosixPath(observation_timestamps_path),
                timestamps,
            )
            payloads.append(
                _payload_metadata(partial_directory, observation_timestamps_path)
            )

            action_path = "actions/data.npy"
            _write_npy(
                partial_directory / PurePosixPath(action_path),
                np.stack(self._actions),
            )
            payloads.append(_payload_metadata(partial_directory, action_path))

            action_timestamps_path = "actions/timestamps_ns.npy"
            _write_npy(
                partial_directory / PurePosixPath(action_timestamps_path),
                timestamps,
            )
            payloads.append(_payload_metadata(partial_directory, action_timestamps_path))

            metadata = EpisodeMetadata(
                episode_id=self.episode_id,
                task=self.task,
                robot=self.robot,
                scene=self.scene,
                random_seed=self.random_seed,
                observation_schema=self.observation_schema,
                action_schema=self.action_schema,
                step_count=self.step_count,
                start_time_ns=self._timestamps_ns[0],
                end_time_ns=self._timestamps_ns[-1],
                outcome=outcome,
                termination_reason=termination_reason,
                payloads=tuple(payloads),
                provenance=self.provenance,
                task_parameters=self.task_parameters,
                reset_parameters=self.reset_parameters,
                instruction=self.instruction,
                instruction_id=self.instruction_id,
                instruction_language=self.instruction_language,
                expert=self.expert,
            )
            _write_manifest(partial_directory / "manifest.json", metadata)
            self.final_directory.parent.mkdir(parents=True, exist_ok=True)
            partial_directory.rename(self.final_directory)
        except Exception:
            shutil.rmtree(partial_directory, ignore_errors=True)
            raise

        self._finalized = True
        return RecordedEpisode(directory=self.final_directory, metadata=metadata)


def validate_npy_episode(episode_directory: str | Path) -> EpisodeMetadata:
    """Validate checksums, synchronized timestamps, shapes, and dtypes."""

    root = Path(episode_directory).expanduser().resolve()
    manifest_path = root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        metadata = EpisodeMetadata.from_dict(json.load(stream))

    payload_by_path = {payload.path: payload for payload in metadata.payloads}
    for payload in metadata.payloads:
        payload_path = root / PurePosixPath(payload.path)
        if not payload_path.is_file():
            raise ValueError(f"missing episode payload: {payload.path}")
        if payload_path.stat().st_size != payload.byte_size:
            raise ValueError(f"payload byte size mismatch: {payload.path}")
        if _sha256(payload_path) != payload.sha256:
            raise ValueError(f"payload checksum mismatch: {payload.path}")

    observation_timestamps_path = "observations/timestamps_ns.npy"
    action_timestamps_path = "actions/timestamps_ns.npy"
    action_path = "actions/data.npy"
    required_paths = {
        observation_timestamps_path,
        action_timestamps_path,
        action_path,
        *(
            _observation_payload_path(field.key)
            for field in metadata.observation_schema.fields
        ),
    }
    missing_paths = required_paths - set(payload_by_path)
    if missing_paths:
        raise ValueError(f"manifest is missing required NPY payloads: {missing_paths}")

    observation_timestamps = np.load(
        root / observation_timestamps_path, allow_pickle=False
    )
    action_timestamps = np.load(root / action_timestamps_path, allow_pickle=False)
    expected_timestamp_shape = (metadata.step_count,)
    for name, value in (
        ("observation", observation_timestamps),
        ("action", action_timestamps),
    ):
        if value.shape != expected_timestamp_shape or value.dtype != np.dtype("int64"):
            raise ValueError(f"{name} timestamps have an invalid shape or dtype")
        if value.size > 1 and not np.all(np.diff(value) > 0):
            raise ValueError(f"{name} timestamps are not strictly increasing")
    if not np.array_equal(observation_timestamps, action_timestamps):
        raise ValueError("observation and action timestamps are not synchronized")
    if (
        int(observation_timestamps[0]) != metadata.start_time_ns
        or int(observation_timestamps[-1]) != metadata.end_time_ns
    ):
        raise ValueError("manifest timestamp range does not match payload timestamps")

    actions = np.load(root / action_path, allow_pickle=False, mmap_mode="r")
    expected_action_shape = (metadata.step_count, metadata.action_schema.dimension)
    if actions.shape != expected_action_shape:
        raise ValueError(f"actions have shape {actions.shape}, expected {expected_action_shape}")
    if actions.dtype != np.dtype(metadata.action_schema.dtype.value):
        raise ValueError("action payload dtype does not match the action schema")

    for field in metadata.observation_schema.fields:
        relative_path = _observation_payload_path(field.key)
        values = np.load(root / relative_path, allow_pickle=False, mmap_mode="r")
        expected_shape = (metadata.step_count, *field.shape)
        if values.shape != expected_shape:
            raise ValueError(
                f"observation {field.key!r} has shape {values.shape}, "
                f"expected {expected_shape}"
            )
        if values.dtype != np.dtype(field.dtype.value):
            raise ValueError(
                f"observation {field.key!r} dtype does not match its schema"
            )

    return metadata
