from pathlib import Path


def test_isaac_stage8_client_does_not_import_lerobot() -> None:
    repository = Path(__file__).resolve().parents[2]
    paths = (
        repository / "src/embodied_ai/sim/evaluation/robot_client.py",
        repository / "scripts/sim/evaluate_franka_pick_place_policy.py",
    )
    for path in paths:
        assert "import lerobot" not in path.read_text(encoding="utf-8")


def test_policy_server_does_not_import_isaac() -> None:
    repository = Path(__file__).resolve().parents[2]
    paths = (
        repository / "src/embodied_ai/policies/smolvla/online.py",
        repository / "scripts/policy/serve_smolvla.py",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "import isaaclab" not in text
        assert "import isaacsim" not in text
