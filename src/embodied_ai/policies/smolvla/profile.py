"""Versioned project feature binding for the Franka SmolVLA policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from embodied_ai.data.lerobot_mapping import (
    FRANKA_PICK_PLACE_LEROBOT_MAPPING,
    LEROBOT_ACTION_KEY,
    LEROBOT_FRONT_IMAGE_KEY,
    LEROBOT_STATE_KEY,
)

SMOLVLA_PROFILE_SCHEMA_VERSION = "embodied-ai.smolvla-profile/v1"


@dataclass(frozen=True, slots=True)
class SmolVLAProjectProfile:
    """Bind one validated LeRobot mapping to SmolVLA policy features."""

    name: str
    dataset_repo_id: str
    state_key: str
    state_names: tuple[str, ...]
    image_key: str
    image_shape: tuple[int, int, int]
    action_key: str
    action_names: tuple[str, ...]
    control_hz: int
    chunk_size: int
    schema_version: str = SMOLVLA_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SMOLVLA_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported SmolVLA profile schema version")
        if not self.name or not self.dataset_repo_id:
            raise ValueError("profile name and dataset repo_id are required")
        if len(self.state_names) != 9:
            raise ValueError("the Franka SmolVLA profile requires a 9D state")
        if self.image_shape != (3, 224, 224):
            raise ValueError("the Franka SmolVLA profile requires 3x224x224 front RGB")
        if len(self.action_names) != 7:
            raise ValueError("the Franka SmolVLA profile requires a 7D action")
        if self.control_hz != 20 or self.chunk_size != 50:
            raise ValueError("the Stage 7 profile requires 20 Hz and a 50-step chunk")

    @property
    def state_dimension(self) -> int:
        return len(self.state_names)

    @property
    def action_dimension(self) -> int:
        return len(self.action_names)

    def policy_features(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return LeRobot policy features without importing simulator modules."""

        from lerobot.configs import FeatureType, PolicyFeature

        inputs = {
            self.state_key: PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.state_dimension,),
            ),
            self.image_key: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=self.image_shape,
            ),
        }
        outputs = {
            self.action_key: PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dimension,),
            )
        }
        return inputs, outputs

    def validate_dataset_meta(self, meta: Any) -> None:
        """Reject dataset identity, dimension, name, or frequency drift."""

        if meta.repo_id != self.dataset_repo_id:
            raise ValueError(
                f"dataset repo_id {meta.repo_id!r} does not match {self.dataset_repo_id!r}"
            )
        if int(meta.fps) != self.control_hz:
            raise ValueError("dataset fps does not match the action contract")
        expected = {
            self.state_key: ("float32", (self.state_dimension,), list(self.state_names)),
            self.image_key: ("video", self.image_shape, ["channels", "height", "width"]),
            self.action_key: ("float32", (self.action_dimension,), list(self.action_names)),
        }
        for key, (dtype, shape, names) in expected.items():
            feature = meta.features.get(key)
            if feature is None:
                raise ValueError(f"dataset is missing required feature {key!r}")
            if feature.get("dtype") != dtype:
                raise ValueError(f"dataset feature {key!r} has unexpected dtype")
            if tuple(feature.get("shape", ())) != tuple(shape):
                raise ValueError(f"dataset feature {key!r} has unexpected shape")
            if feature.get("names") != names:
                raise ValueError(f"dataset feature {key!r} has unexpected component names")

        policy_observations = {
            key for key in meta.features if key.startswith("observation.")
        }
        allowed = {self.state_key, self.image_key}
        if not allowed.issubset(policy_observations):
            raise ValueError("dataset does not expose the required policy observations")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "dataset_repo_id": self.dataset_repo_id,
            "state": {"key": self.state_key, "names": list(self.state_names)},
            "image": {"key": self.image_key, "shape": list(self.image_shape)},
            "action": {"key": self.action_key, "names": list(self.action_names)},
            "control_hz": self.control_hz,
            "chunk_size": self.chunk_size,
        }


_MAPPING = FRANKA_PICK_PLACE_LEROBOT_MAPPING

FRANKA_PICK_PLACE_SMOLVLA_PROFILE = SmolVLAProjectProfile(
    name=_MAPPING.profile,
    dataset_repo_id="embodiedai/franka-pick-place-stage7-batch-v1",
    state_key=LEROBOT_STATE_KEY,
    state_names=_MAPPING.state_component_names,
    image_key=LEROBOT_FRONT_IMAGE_KEY,
    image_shape=(3, 224, 224),
    action_key=LEROBOT_ACTION_KEY,
    action_names=_MAPPING.action_component_names,
    control_hz=round(_MAPPING.action_schema.control_hz),
    chunk_size=50,
)


def franka_pick_place_smolvla_profile(dataset_repo_id: str) -> SmolVLAProjectProfile:
    """Bind the stable feature mapping to one immutable dataset repository identity."""

    if not isinstance(dataset_repo_id, str) or not dataset_repo_id.strip():
        raise ValueError("dataset_repo_id must be a non-empty string")
    return SmolVLAProjectProfile(
        name=FRANKA_PICK_PLACE_SMOLVLA_PROFILE.name,
        dataset_repo_id=dataset_repo_id,
        state_key=FRANKA_PICK_PLACE_SMOLVLA_PROFILE.state_key,
        state_names=FRANKA_PICK_PLACE_SMOLVLA_PROFILE.state_names,
        image_key=FRANKA_PICK_PLACE_SMOLVLA_PROFILE.image_key,
        image_shape=FRANKA_PICK_PLACE_SMOLVLA_PROFILE.image_shape,
        action_key=FRANKA_PICK_PLACE_SMOLVLA_PROFILE.action_key,
        action_names=FRANKA_PICK_PLACE_SMOLVLA_PROFILE.action_names,
        control_hz=FRANKA_PICK_PLACE_SMOLVLA_PROFILE.control_hz,
        chunk_size=FRANKA_PICK_PLACE_SMOLVLA_PROFILE.chunk_size,
    )

__all__ = [
    "FRANKA_PICK_PLACE_SMOLVLA_PROFILE",
    "SMOLVLA_PROFILE_SCHEMA_VERSION",
    "SmolVLAProjectProfile",
    "franka_pick_place_smolvla_profile",
]
