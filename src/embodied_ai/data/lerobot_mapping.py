"""Versioned mappings from dependency-light contracts to LeRobot features."""

from __future__ import annotations

from dataclasses import dataclass

from embodied_ai.contracts import (
    ActionSchema,
    DataType,
    EpisodeMetadata,
    ObservationKind,
    ObservationSchema,
)
from embodied_ai.contracts.tasks.franka_pick_place import (
    FRANKA_PICK_PLACE_ACTION_SCHEMA,
    FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
    ROBOT_NAME,
    SCENE_NAME,
    TASK_NAME,
)

LEROBOT_MAPPING_SCHEMA_VERSION = "embodied-ai.lerobot-mapping/v1"
LEROBOT_STATE_KEY = "observation.state"
LEROBOT_ACTION_KEY = "action"
LEROBOT_FRONT_IMAGE_KEY = "observation.images.front"


@dataclass(frozen=True, slots=True)
class ImageFeatureMapping:
    """Map one contract RGB stream to one LeRobot visual feature."""

    source_key: str
    target_key: str

    def to_dict(self) -> dict[str, str]:
        return {"source_key": self.source_key, "target_key": self.target_key}


@dataclass(frozen=True, slots=True)
class ExcludedObservation:
    """Record an intentional non-policy observation and the reason it is excluded."""

    source_key: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"source_key": self.source_key, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ContractLeRobotMapping:
    """Complete, reviewable mapping profile for one task contract."""

    profile: str
    task: str
    robot: str
    scene: str
    observation_schema: ObservationSchema
    action_schema: ActionSchema
    state_source_keys: tuple[str, ...]
    image_features: tuple[ImageFeatureMapping, ...]
    excluded_observations: tuple[ExcludedObservation, ...]
    schema_version: str = LEROBOT_MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LEROBOT_MAPPING_SCHEMA_VERSION:
            raise ValueError(f"unsupported LeRobot mapping version: {self.schema_version!r}")
        if not self.profile or not self.task or not self.robot or not self.scene:
            raise ValueError("mapping identity fields must be non-empty")
        if not self.state_source_keys:
            raise ValueError("at least one state source is required")
        if not self.image_features:
            raise ValueError("at least one image mapping is required")

        state_keys = set(self.state_source_keys)
        image_source_keys = {mapping.source_key for mapping in self.image_features}
        image_target_keys = {mapping.target_key for mapping in self.image_features}
        excluded_keys = {field.source_key for field in self.excluded_observations}
        if len(state_keys) != len(self.state_source_keys):
            raise ValueError("state source keys must be unique")
        if len(image_source_keys) != len(self.image_features):
            raise ValueError("image source keys must be unique")
        if len(image_target_keys) != len(self.image_features):
            raise ValueError("image target keys must be unique")
        if len(excluded_keys) != len(self.excluded_observations):
            raise ValueError("excluded observation keys must be unique")
        roles_overlap = (
            state_keys & image_source_keys
            or state_keys & excluded_keys
            or image_source_keys & excluded_keys
        )
        if roles_overlap:
            raise ValueError("every contract observation must have exactly one mapping role")

        schema_keys = {field.key for field in self.observation_schema.fields}
        classified_keys = state_keys | image_source_keys | excluded_keys
        if classified_keys != schema_keys:
            missing = sorted(schema_keys - classified_keys)
            unknown = sorted(classified_keys - schema_keys)
            raise ValueError(
                "mapping must classify every observation exactly once: "
                f"missing={missing}, unknown={unknown}"
            )

        for source_key in self.state_source_keys:
            field = self.observation_schema.field(source_key)
            if field.kind is not ObservationKind.STATE:
                raise ValueError(f"state source {source_key!r} is not a state observation")
            if field.dtype not in (DataType.FLOAT32, DataType.FLOAT64):
                raise ValueError(f"state source {source_key!r} must be floating point")
        for image_mapping in self.image_features:
            field = self.observation_schema.field(image_mapping.source_key)
            if field.kind is not ObservationKind.RGB_IMAGE:
                raise ValueError(f"image source {image_mapping.source_key!r} is not RGB")
            if not image_mapping.target_key.startswith("observation.images."):
                raise ValueError("LeRobot image targets must start with 'observation.images.'")

    @property
    def state_dimension(self) -> int:
        return sum(self.observation_schema.field(key).shape[0] for key in self.state_source_keys)

    @property
    def state_component_names(self) -> tuple[str, ...]:
        # Prefix names so future concatenated state sources remain unambiguous.
        return tuple(
            f"{source_key}.{component.name}"
            for source_key in self.state_source_keys
            for component in self.observation_schema.field(source_key).components
        )

    @property
    def action_component_names(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.action_schema.components)

    def validate_episode(self, metadata: EpisodeMetadata) -> None:
        """Reject task or schema drift before any destination files are written."""

        if (metadata.task, metadata.robot, metadata.scene) != (self.task, self.robot, self.scene):
            raise ValueError(
                "episode task identity does not match the mapping profile: "
                f"{(metadata.task, metadata.robot, metadata.scene)!r}"
            )
        if metadata.observation_schema != self.observation_schema:
            raise ValueError("episode observation schema does not match the mapping profile")
        if metadata.action_schema != self.action_schema:
            raise ValueError("episode action schema does not match the mapping profile")

    def lerobot_features(self, *, use_videos: bool) -> dict[str, dict[str, object]]:
        """Return the LeRobot 0.6 feature declaration for this profile."""

        features: dict[str, dict[str, object]] = {
            LEROBOT_STATE_KEY: {
                "dtype": "float32",
                "shape": (self.state_dimension,),
                "names": list(self.state_component_names),
            },
            LEROBOT_ACTION_KEY: {
                "dtype": self.action_schema.dtype.value,
                "shape": (self.action_schema.dimension,),
                "names": list(self.action_component_names),
            },
        }
        for image_mapping in self.image_features:
            source = self.observation_schema.field(image_mapping.source_key)
            features[image_mapping.target_key] = {
                "dtype": "video" if use_videos else "image",
                "shape": source.shape,
                "names": ["channels", "height", "width"],
            }
        return features

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "task": self.task,
            "robot": self.robot,
            "scene": self.scene,
            "state": {
                "source_keys": list(self.state_source_keys),
                "target_key": LEROBOT_STATE_KEY,
                "component_names": list(self.state_component_names),
            },
            "images": [mapping.to_dict() for mapping in self.image_features],
            "action": {
                "target_key": LEROBOT_ACTION_KEY,
                "representation": self.action_schema.representation.value,
                "component_names": list(self.action_component_names),
                "normalized": self.action_schema.normalized,
                "frame": self.action_schema.frame,
            },
            "excluded_observations": [
                observation.to_dict() for observation in self.excluded_observations
            ],
        }


FRANKA_PICK_PLACE_LEROBOT_MAPPING = ContractLeRobotMapping(
    profile="franka-pick-place-smolvla-v1",
    task=TASK_NAME,
    robot=ROBOT_NAME,
    scene=SCENE_NAME,
    observation_schema=FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
    action_schema=FRANKA_PICK_PLACE_ACTION_SCHEMA,
    state_source_keys=("robot.joint_position",),
    image_features=(
        ImageFeatureMapping(
            source_key="camera.front.rgb",
            target_key=LEROBOT_FRONT_IMAGE_KEY,
        ),
    ),
    excluded_observations=(
        ExcludedObservation(
            source_key="robot.joint_velocity",
            reason="not part of the initial SmolVLA proprioceptive-state baseline",
        ),
        ExcludedObservation(
            source_key="object.cube.position",
            reason="privileged simulator state; the policy must infer object state from RGB",
        ),
    ),
)


__all__ = [
    "ContractLeRobotMapping",
    "ExcludedObservation",
    "FRANKA_PICK_PLACE_LEROBOT_MAPPING",
    "ImageFeatureMapping",
    "LEROBOT_ACTION_KEY",
    "LEROBOT_FRONT_IMAGE_KEY",
    "LEROBOT_MAPPING_SCHEMA_VERSION",
    "LEROBOT_STATE_KEY",
]
