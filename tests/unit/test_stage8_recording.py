import json
from pathlib import Path

import numpy as np

from embodied_ai.contracts.evaluation import (
    EvaluationScenario,
    PolicyDescriptor,
    PolicyKind,
    RolloutOutcome,
)
from embodied_ai.evaluation.recording import EvaluationRolloutRecorder


def test_evaluation_recorder_atomically_publishes_payloads(tmp_path: Path) -> None:
    output = tmp_path / "rollout"
    recorder = EvaluationRolloutRecorder(output)
    recorder.append(
        joint_position=np.zeros(9, dtype=np.float32),
        camera_rgb=np.zeros((3, 224, 224), dtype=np.uint8),
        raw_action=np.zeros(7, dtype=np.float32),
        executed_action=np.zeros(7, dtype=np.float32),
        cube_position=np.asarray((0.5, 0.0, 0.03), dtype=np.float32),
        goal_error_m=0.25,
        cube_speed_m_s=0.0,
        gripper_open=True,
        timestamp_ns=50_000_000,
    )
    scenario = EvaluationScenario(
        "scenario", 0, 1, "Do task.", (0.5, 0.0, 0.03), (0.65, -0.2, 0.03)
    )
    descriptor = PolicyDescriptor(PolicyKind.BASE, "policy", "revision", "a" * 64)
    manifest = recorder.finalize(
        rollout_id="rollout",
        run_id="run",
        scenario=scenario,
        policy=descriptor,
        noise_seed=1,
        outcome=RolloutOutcome.ERROR,
        termination_reason="test",
        provenance={"test": True},
        record_video=False,
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["step_count"] == 1
    assert value["metrics"]["success"] == 0
    assert len(value["payloads"]) == 10
    assert not list(tmp_path.glob("*.partial"))
