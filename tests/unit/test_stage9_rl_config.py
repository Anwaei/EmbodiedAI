"""Tests for the reviewed standalone PPO TOML and lock identity."""

import tomllib
from pathlib import Path

import pytest

from embodied_ai.contracts.rl import RSL_RL_PPO_BACKEND
from embodied_ai.rl.config import Stage9StandalonePpoConfig, reviewed_stage9_config_path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_reviewed_standalone_ppo_config_loads_and_stays_external() -> None:
    repository = _repository_root()
    data_root = Path("/root/autodl-tmp/EmbodiedAI")
    config = Stage9StandalonePpoConfig.from_toml(
        reviewed_stage9_config_path(repository),
        repository_root=repository,
        data_root=data_root,
    )

    assert config.identity.mode.value == "standalone_ppo"
    assert config.num_envs == 128
    assert config.max_episode_steps == 200
    assert config.evaluation_scenarios_path.is_file()
    assert config.output_dir.is_relative_to(data_root)
    assert config.ppo.num_steps_per_env == 24
    assert config.reward.lift_target_height_m > config.phase.lift_height_m
    assert config.reward.failure_weight > 0.0


def test_backend_versions_match_isaac_lock() -> None:
    lock = tomllib.loads((_repository_root() / "env/isaac/uv.lock").read_text(encoding="utf-8"))
    versions = {package["name"]: package["version"] for package in lock["package"]}

    assert versions[RSL_RL_PPO_BACKEND.package] == RSL_RL_PPO_BACKEND.package_version
    assert versions["isaaclab-rl"] == RSL_RL_PPO_BACKEND.isaaclab_rl_version


def test_reset_distribution_rejects_overlap_with_success_region(tmp_path: Path) -> None:
    repository = _repository_root()
    source = reviewed_stage9_config_path(repository).read_text(encoding="utf-8")
    source = source.replace(
        "goal_x_m = [0.605, 0.695]",
        "goal_x_m = [0.500, 0.550]",
    ).replace(
        "goal_y_m = [-0.255, -0.145]",
        "goal_y_m = [-0.050, 0.050]",
    )
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="success tolerance"):
        Stage9StandalonePpoConfig.from_toml(
            invalid,
            repository_root=repository,
            data_root=Path("/root/autodl-tmp/EmbodiedAI"),
        )
