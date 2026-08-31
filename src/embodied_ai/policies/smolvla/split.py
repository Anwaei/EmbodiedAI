"""Dependency-light episode split contract for Stage 7 offline work."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGE7_SPLIT_SCHEMA_VERSION = "embodied-ai.stage7-split/v1"
STAGE7_FORMAL_SPLIT_SCHEMA_VERSION = "embodied-ai.stage7-split/v2"


def _episode_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{label} must contain non-negative integer indices")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate episode indices")
    return result


def _optional_episode_tuple(value: object, label: str) -> tuple[int, ...]:
    if value is None or value == []:
        return ()
    return _episode_tuple(value, label)


@dataclass(frozen=True, slots=True)
class Stage7EpisodeSplit:
    """Frozen whole-episode split with source identity and review rationale."""

    dataset_repo_id: str
    dataset_root_name: str
    mapping_profile: str
    train_episode_indices: tuple[int, ...]
    validation_episode_indices: tuple[int, ...]
    test_episode_indices: tuple[int, ...]
    source_episode_ids: tuple[str, ...]
    rationale: str
    schema_version: str = STAGE7_SPLIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in (
            STAGE7_SPLIT_SCHEMA_VERSION,
            STAGE7_FORMAL_SPLIT_SCHEMA_VERSION,
        ):
            raise ValueError("unsupported Stage 7 split schema version")
        if not self.dataset_repo_id or not self.dataset_root_name or not self.mapping_profile:
            raise ValueError("dataset and mapping identities are required")
        partitions = (
            set(self.train_episode_indices),
            set(self.validation_episode_indices),
            set(self.test_episode_indices),
        )
        if any(
            partitions[left] & partitions[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("training, validation, and test episodes must be disjoint")
        combined = (
            self.train_episode_indices
            + self.validation_episode_indices
            + self.test_episode_indices
        )
        if self.schema_version == STAGE7_SPLIT_SCHEMA_VERSION:
            if (len(self.train_episode_indices), len(self.validation_episode_indices)) != (15, 5):
                raise ValueError("Stage 7 Step 6A requires a 15/5 split")
            if self.test_episode_indices:
                raise ValueError("Stage 7 Step 6A does not define a test partition")
            expected_count = 20
        else:
            if (
                len(self.train_episode_indices),
                len(self.validation_episode_indices),
                len(self.test_episode_indices),
            ) != (80, 10, 10):
                raise ValueError("Stage 7 Step 6B requires an 80/10/10 split")
            expected_count = 100
        if set(combined) != set(range(expected_count)):
            raise ValueError(f"the split must cover each of the {expected_count} episodes once")
        if (
            len(self.source_episode_ids) != expected_count
            or len(set(self.source_episode_ids)) != expected_count
        ):
            raise ValueError(
                f"source_episode_ids must contain {expected_count} unique ordered identities"
            )
        if not self.rationale.strip():
            raise ValueError("split rationale is required")

    @classmethod
    def from_json(cls, path: Path) -> Stage7EpisodeSplit:
        with path.open(encoding="utf-8") as stream:
            value: Any = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("split file must contain a JSON object")
        source_ids = value.get("source_episode_ids")
        if not isinstance(source_ids, list) or not all(
            isinstance(item, str) and item for item in source_ids
        ):
            raise ValueError("source_episode_ids must contain non-empty strings")
        return cls(
            dataset_repo_id=value.get("dataset_repo_id"),
            dataset_root_name=value.get("dataset_root_name"),
            mapping_profile=value.get("mapping_profile"),
            train_episode_indices=_episode_tuple(
                value.get("train_episode_indices"), "train_episode_indices"
            ),
            validation_episode_indices=_episode_tuple(
                value.get("validation_episode_indices"), "validation_episode_indices"
            ),
            test_episode_indices=_optional_episode_tuple(
                value.get("test_episode_indices"), "test_episode_indices"
            ),
            source_episode_ids=tuple(source_ids),
            rationale=value.get("rationale"),
            schema_version=value.get("schema_version"),
        )

    def validate_conversion_manifest(self, manifest: dict[str, object]) -> None:
        if manifest.get("repo_id") != self.dataset_repo_id:
            raise ValueError("split repo_id differs from conversion provenance")
        mapping = manifest.get("mapping")
        if not isinstance(mapping, dict) or mapping.get("profile") != self.mapping_profile:
            raise ValueError("split mapping profile differs from conversion provenance")
        records = manifest.get("source_episodes")
        if not isinstance(records, list) or len(records) != len(self.source_episode_ids):
            raise ValueError("conversion provenance episode count differs from the split")
        ordered: list[str] = []
        for expected_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError("conversion source episode record must be an object")
            if record.get("lerobot_episode_index") != expected_index:
                raise ValueError("conversion episode indices are not ordered")
            source_id = record.get("source_episode_id")
            if not isinstance(source_id, str):
                raise ValueError("conversion source episode identity is missing")
            ordered.append(source_id)
        if tuple(ordered) != self.source_episode_ids:
            raise ValueError("split source identities differ from conversion provenance")

    def validate_formal_balance(self, manifest: dict[str, object]) -> None:
        """Verify the reviewed Step 6B instruction/cube/goal partition balance."""

        if self.schema_version != STAGE7_FORMAL_SPLIT_SCHEMA_VERSION:
            return
        records = manifest.get("source_episodes")
        if not isinstance(records, list) or len(records) != 100:
            raise ValueError("formal split balance requires 100 conversion records")

        def signature(record: object) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
            if not isinstance(record, dict):
                raise ValueError("formal split conversion record must be an object")
            reset = record.get("reset_parameters")
            task = record.get("task_parameters")
            instruction = record.get("instruction")
            if not isinstance(reset, dict) or not isinstance(task, dict):
                raise ValueError("formal split record lacks task/reset parameters")
            cube = reset.get("cube_position_env_m")
            goal = task.get("goal_position_env_m")
            if (
                not isinstance(instruction, str)
                or not isinstance(cube, list)
                or not isinstance(goal, list)
            ):
                raise ValueError("formal split record lacks instruction/cube/goal values")
            return instruction, tuple(float(value) for value in cube), tuple(
                float(value) for value in goal
            )

        signatures = tuple(signature(record) for record in records)
        expectations = (
            ("training", self.train_episode_indices, 16, 8),
            ("validation", self.validation_episode_indices, 2, 1),
            ("test", self.test_episode_indices, 2, 1),
        )
        for label, indices, instruction_count, spatial_count in expectations:
            selected = tuple(signatures[index] for index in indices)
            instructions = Counter(item[0] for item in selected)
            cubes = Counter(item[1] for item in selected)
            goals = Counter(item[2] for item in selected)
            if len(instructions) != 5 or set(instructions.values()) != {instruction_count}:
                raise ValueError(f"{label} instruction balance differs from the reviewed split")
            if len(cubes) != 10 or set(cubes.values()) != {spatial_count}:
                raise ValueError(f"{label} cube-reset balance differs from the reviewed split")
            if len(goals) != 10 or set(goals.values()) != {spatial_count}:
                raise ValueError(f"{label} goal balance differs from the reviewed split")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_repo_id": self.dataset_repo_id,
            "dataset_root_name": self.dataset_root_name,
            "mapping_profile": self.mapping_profile,
            "train_episode_indices": list(self.train_episode_indices),
            "validation_episode_indices": list(self.validation_episode_indices),
            "test_episode_indices": list(self.test_episode_indices),
            "source_episode_ids": list(self.source_episode_ids),
            "rationale": self.rationale,
        }


__all__ = [
    "STAGE7_FORMAL_SPLIT_SCHEMA_VERSION",
    "STAGE7_SPLIT_SCHEMA_VERSION",
    "Stage7EpisodeSplit",
]
