"""CPU tests for Stage 7 LeRobotDataset validation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from embodied_ai.contracts import EpisodeOutcome, EpisodeProvenance, ExpertKind, ExpertMetadata
from embodied_ai.contracts.tasks.franka_pick_place import (
    FRANKA_PICK_PLACE_ACTION_SCHEMA,
    FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
    ROBOT_NAME,
    SCENE_NAME,
    TASK_NAME,
)

_VALIDATION_DEPS_AVAILABLE = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("lerobot") is not None
)


@unittest.skipUnless(
    _VALIDATION_DEPS_AVAILABLE,
    "NumPy and LeRobot are intentionally absent from the dev lock",
)
class LeRobotValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        from embodied_ai.data.lerobot_converter import convert_contract_episodes_to_lerobot
        from embodied_ai.data.lerobot_validation import (
            LEROBOT_VALIDATION_SCHEMA_VERSION,
            validate_lerobot_dataset,
            write_validation_report,
        )
        from embodied_ai.sim.recording import NpyEpisodeRecorder

        self.np = np
        self.convert = convert_contract_episodes_to_lerobot
        self.recorder_type = NpyEpisodeRecorder
        self.schema_version = LEROBOT_VALIDATION_SCHEMA_VERSION
        self.validate = validate_lerobot_dataset
        self.write_report = write_validation_report

    def _record_source_episode(
        self,
        output_root: Path,
        *,
        episode_number: int,
        instruction: str,
    ) -> Path:
        recorder = self.recorder_type(
            output_root=output_root,
            episode_id=f"episode-stage7-validation-{episode_number:06d}",
            task=TASK_NAME,
            robot=ROBOT_NAME,
            scene=SCENE_NAME,
            random_seed=episode_number,
            observation_schema=FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
            action_schema=FRANKA_PICK_PLACE_ACTION_SCHEMA,
            task_parameters={"goal_position_env_m": [0.62, -0.18, 0.03]},
            reset_parameters={"cube_position_env_m": [0.46, -0.05, 0.03]},
            provenance=EpisodeProvenance(
                simulator_name="test-simulator",
                simulator_version="1.0",
                repository_revision="test-revision",
                configuration_revision="test-configuration",
                environment_lock_sha256="a" * 64,
            ),
            instruction=instruction,
            instruction_id=f"validation-instruction-{episode_number:03d}",
            instruction_language="en",
            expert=ExpertMetadata(
                kind=ExpertKind.STATE_MACHINE,
                identifier="test-state-machine",
                revision="v1",
                configuration_revision="b" * 64,
            ),
        )
        for step in range(3):
            image = self.np.zeros((3, 224, 224), dtype=self.np.uint8)
            image[(step + episode_number) % 3] = 32 * episode_number + step
            action = self.np.zeros((7,), dtype=self.np.float32)
            action[0] = (episode_number + step) / 10
            recorder.append(
                {
                    "robot.joint_position": self.np.full(
                        (9,), episode_number + step / 10, dtype=self.np.float32
                    ),
                    "robot.joint_velocity": self.np.zeros((9,), dtype=self.np.float32),
                    "object.cube.position": self.np.asarray(
                        [0.5, 0.0, 0.03], dtype=self.np.float32
                    ),
                    "camera.front.rgb": image,
                },
                action,
                step * 50_000_000,
            )
        return recorder.finalize(EpisodeOutcome.SUCCESS, termination_reason=None).directory

    def test_validates_multi_episode_dataset_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "contract"
            source_a = self._record_source_episode(
                source_root,
                episode_number=1,
                instruction="Pick up the cube and place it in the goal.",
            )
            source_b = self._record_source_episode(
                source_root,
                episode_number=2,
                instruction="Move the cube to the green goal.",
            )
            dataset_root = root / "lerobot"
            repo_id = "embodiedai/stage7-validation-unit"
            self.convert(
                (source_a, source_b),
                dataset_root,
                repo_id=repo_id,
                use_videos=False,
            )

            # Source argument order is intentionally reversed; provenance defines dataset order.
            report = self.validate(
                dataset_root,
                (source_b, source_a),
                repo_id=repo_id,
            )
            self.assertEqual(report.episode_count, 2)
            self.assertEqual(report.frame_count, 6)
            self.assertEqual(report.task_count, 2)
            self.assertEqual(report.decoded_image_samples, 4)
            self.assertEqual(report.deterministic_reload_samples, 4)
            self.assertEqual(report.normalization_inputs[0].count, 6)
            self.assertEqual(report.normalization_inputs[1].count, 6)
            self.assertEqual(len(report.table_fingerprint_sha256), 64)
            self.assertEqual(len(report.reload_fingerprint_sha256), 64)

            report_path = self.write_report(report, root / "runs" / "validation.json")
            serialized = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(serialized["schema_version"], self.schema_version)
            self.assertEqual(serialized["status"], "passed")
            self.assertEqual(serialized["episode_count"], 2)
            self.assertEqual(serialized["frame_count"], 6)

    def test_rejects_incomplete_source_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "contract"
            source_a = self._record_source_episode(
                source_root,
                episode_number=1,
                instruction="Pick up the cube and place it in the goal.",
            )
            source_b = self._record_source_episode(
                source_root,
                episode_number=2,
                instruction="Move the cube to the green goal.",
            )
            dataset_root = root / "lerobot"
            repo_id = "embodiedai/stage7-validation-missing"
            self.convert(
                (source_a, source_b),
                dataset_root,
                repo_id=repo_id,
                use_videos=False,
            )

            with self.assertRaisesRegex(ValueError, "unknown source episode"):
                self.validate(dataset_root, (source_a,), repo_id=repo_id)


if __name__ == "__main__":
    unittest.main()
