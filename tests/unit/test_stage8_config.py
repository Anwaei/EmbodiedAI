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
