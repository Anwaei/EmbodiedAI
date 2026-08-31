from pathlib import Path

from embodied_ai.evaluation.config import Stage8RunConfig, load_scenarios


def test_reviewed_stage8_config() -> None:
    repository = Path(__file__).resolve().parents[2]
    config = Stage8RunConfig.from_toml(
        repository / "configs/evaluation/stage8_franka_pick_place_v1.toml",
        repository_root=repository,
        data_root=Path("/root/autodl-tmp/EmbodiedAI"),
    )
    assert config.execute_horizon == 5
    assert config.prediction_horizon == 50
    assert len(load_scenarios(config.scenarios_path)) == 5


def test_expanded_stage8_config_uses_ten_test_scenarios() -> None:
    repository = Path(__file__).resolve().parents[2]
    config = Stage8RunConfig.from_toml(
        repository / "configs/evaluation/stage8_franka_pick_place_step6b_v1.toml",
        repository_root=repository,
        data_root=Path("/root/autodl-tmp/EmbodiedAI"),
    )
    scenarios = load_scenarios(
        config.scenarios_path,
        expected_count=config.expected_scenario_count,
    )
    assert len(scenarios) == 10
    assert {scenario.source_episode_index for scenario in scenarios} == {
        3,
        18,
        27,
        31,
        45,
        54,
        66,
        79,
        82,
        90,
    }
