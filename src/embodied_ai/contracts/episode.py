"""Versioned episode metadata that has no simulator or ML dependencies."""

from dataclasses import dataclass

EPISODE_SCHEMA_VERSION = "embodied-ai.episode/v1"


@dataclass(frozen=True, slots=True)
class EpisodeManifest:
    """Minimal metadata required at the simulation/training boundary."""

    episode_id: str
    task: str
    robot: str
    observation_keys: tuple[str, ...]
    action_dimension: int
    control_hz: float
    schema_version: str = EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if not self.task:
            raise ValueError("task must not be empty")
        if not self.robot:
            raise ValueError("robot must not be empty")
        if not self.observation_keys:
            raise ValueError("at least one observation key is required")
        if self.action_dimension <= 0:
            raise ValueError("action_dimension must be positive")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
