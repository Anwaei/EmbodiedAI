import pytest

from embodied_ai.contracts import EPISODE_SCHEMA_VERSION, EpisodeManifest


def test_episode_manifest_accepts_valid_boundary_metadata() -> None:
    manifest = EpisodeManifest(
        episode_id="episode-000001",
        task="pick-and-place",
        robot="franka-panda",
        observation_keys=("camera.front", "robot.joint_state"),
        action_dimension=7,
        control_hz=20.0,
    )

    assert manifest.schema_version == EPISODE_SCHEMA_VERSION


@pytest.mark.parametrize("field,value", [("action_dimension", 0), ("control_hz", 0.0)])
def test_episode_manifest_rejects_non_positive_control_values(field: str, value: float) -> None:
    values = {
        "episode_id": "episode-000001",
        "task": "pick-and-place",
        "robot": "franka-panda",
        "observation_keys": ("robot.joint_state",),
        "action_dimension": 7,
        "control_hz": 20.0,
    }
    values[field] = value

    with pytest.raises(ValueError):
        EpisodeManifest(**values)
