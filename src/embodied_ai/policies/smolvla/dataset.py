"""Validated LeRobotDataset access for Stage 7 SmolVLA jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Stage7RunConfig
from .processing import load_json_object
from .profile import FRANKA_PICK_PLACE_SMOLVLA_PROFILE, SmolVLAProjectProfile
from .split import Stage7EpisodeSplit

CONVERSION_MANIFEST_RELATIVE_PATH = Path("meta/embodied_ai_conversion.json")


@dataclass(frozen=True, slots=True)
class OfflineAnchor:
    episode_index: int
    frame_index: int
    dataset_index: int
    position: str
    task: str

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "dataset_index": self.dataset_index,
            "position": self.position,
            "task": self.task,
        }


def resolve_dataset_root(config: Stage7RunConfig) -> Path:
    import os

    datasets_root = Path(
        os.environ.get("EMBODIEDAI_DATASETS", "/root/autodl-tmp/EmbodiedAI/datasets")
    ).expanduser().resolve()
    root = (datasets_root / config.dataset_root_name).resolve()
    if not root.is_relative_to(datasets_root):
        raise ValueError("dataset root must remain under EMBODIEDAI_DATASETS")
    if not root.is_dir():
        raise FileNotFoundError(f"project dataset does not exist: {root}")
    return root


def validate_dataset_and_split(
    config: Stage7RunConfig,
    split: Stage7EpisodeSplit,
    *,
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE,
) -> tuple[Path, dict[str, Any]]:
    root = resolve_dataset_root(config)
    if split.dataset_repo_id != config.dataset_repo_id:
        raise ValueError("split and run config dataset repo_id differ")
    if split.dataset_root_name != config.dataset_root_name:
        raise ValueError("split and run config dataset roots differ")
    if split.mapping_profile != config.mapping_profile or split.mapping_profile != profile.name:
        raise ValueError("split, run config, and project mapping profiles differ")
    manifest = load_json_object(root / CONVERSION_MANIFEST_RELATIVE_PATH)
    split.validate_conversion_manifest(manifest)
    return root, manifest


def open_dataset(
    config: Stage7RunConfig,
    *,
    episodes: tuple[int, ...] | None = None,
    policy_config: Any | None = None,
    action_horizon: bool = False,
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE,
) -> Any:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = resolve_dataset_root(config)
    delta_timestamps = None
    if action_horizon:
        if policy_config is None:
            raise ValueError("policy_config is required for action-horizon loading")
        metadata = LeRobotDatasetMetadata(config.dataset_repo_id, root=root)
        profile.validate_dataset_meta(metadata)
        delta_timestamps = resolve_delta_timestamps(policy_config, metadata)
    dataset = LeRobotDataset(
        config.dataset_repo_id,
        root=root,
        episodes=None if episodes is None else list(episodes),
        delta_timestamps=delta_timestamps,
        video_backend=config.video_backend,
    )
    profile.validate_dataset_meta(dataset.meta)
    return dataset


def policy_input_from_sample(
    sample: dict[str, Any],
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE,
) -> dict[str, Any]:
    """Drop labels and indexing metadata from a causal inference sample."""

    return {
        profile.state_key: sample[profile.state_key],
        profile.image_key: sample[profile.image_key],
        "task": sample["task"],
    }


def offline_anchors(
    dataset: Any,
    episode_indices: tuple[int, ...],
) -> tuple[OfflineAnchor, ...]:
    """Select deterministic first/middle/last frames for each held-out episode."""

    rows = {int(row["episode_index"]): row for row in dataset.meta.episodes}
    anchors: list[OfflineAnchor] = []
    for episode_index in episode_indices:
        row = rows.get(episode_index)
        if row is None:
            raise ValueError(f"dataset metadata lacks episode {episode_index}")
        length = int(row["length"])
        start = int(row["dataset_from_index"])
        task_values = row.get("tasks")
        if not isinstance(task_values, list) or len(task_values) != 1:
            raise ValueError(f"episode {episode_index} must have one exact task")
        task = str(task_values[0])
        selections = (
            ("first", 0),
            ("middle", length // 2),
            ("last", length - 1),
        )
        for position, frame_index in selections:
            anchors.append(
                OfflineAnchor(
                    episode_index=episode_index,
                    frame_index=frame_index,
                    dataset_index=start + frame_index,
                    position=position,
                    task=task,
                )
            )
    return tuple(anchors)


__all__ = [
    "CONVERSION_MANIFEST_RELATIVE_PATH",
    "OfflineAnchor",
    "offline_anchors",
    "open_dataset",
    "policy_input_from_sample",
    "resolve_dataset_root",
    "validate_dataset_and_split",
]
