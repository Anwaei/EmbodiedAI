"""Dependency-light episode split contract for Stage 7 offline work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGE7_SPLIT_SCHEMA_VERSION = "embodied-ai.stage7-split/v1"


def _episode_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{label} must contain non-negative integer indices")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate episode indices")
    return result


@dataclass(frozen=True, slots=True)
class Stage7EpisodeSplit:
    """Frozen whole-episode split with source identity and review rationale."""

    dataset_repo_id: str
    dataset_root_name: str
    mapping_profile: str
    train_episode_indices: tuple[int, ...]
    validation_episode_indices: tuple[int, ...]
    source_episode_ids: tuple[str, ...]
    rationale: str
    schema_version: str = STAGE7_SPLIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE7_SPLIT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 7 split schema version")
        if not self.dataset_repo_id or not self.dataset_root_name or not self.mapping_profile:
            raise ValueError("dataset and mapping identities are required")
        if len(self.train_episode_indices) != 15:
            raise ValueError("Stage 7 Step 6A requires exactly 15 training episodes")
        if len(self.validation_episode_indices) != 5:
            raise ValueError("Stage 7 Step 6A requires exactly 5 validation episodes")
        if set(self.train_episode_indices) & set(self.validation_episode_indices):
            raise ValueError("training and validation episodes must be disjoint")
        combined = self.train_episode_indices + self.validation_episode_indices
        if set(combined) != set(range(20)):
            raise ValueError("the split must cover each of the 20 episodes exactly once")
        if len(self.source_episode_ids) != 20 or len(set(self.source_episode_ids)) != 20:
            raise ValueError("source_episode_ids must contain 20 unique ordered identities")
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
        if not isinstance(records, list) or len(records) != 20:
            raise ValueError("conversion provenance must contain 20 episodes")
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

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_repo_id": self.dataset_repo_id,
            "dataset_root_name": self.dataset_root_name,
            "mapping_profile": self.mapping_profile,
            "train_episode_indices": list(self.train_episode_indices),
            "validation_episode_indices": list(self.validation_episode_indices),
            "source_episode_ids": list(self.source_episode_ids),
            "rationale": self.rationale,
        }


__all__ = ["STAGE7_SPLIT_SCHEMA_VERSION", "Stage7EpisodeSplit"]
