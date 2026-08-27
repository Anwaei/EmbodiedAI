"""CPU tests for converting immutable contract episodes to LeRobot format."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from embodied_ai.contracts import EpisodeOutcome, EpisodeProvenance, ExpertKind, ExpertMetadata
from embodied_ai.contracts.tasks.franka_pick_place import (
    DEFAULT_INSTRUCTION,
    DEFAULT_INSTRUCTION_ID,
    DEFAULT_INSTRUCTION_LANGUAGE,
    FRANKA_PICK_PLACE_ACTION_SCHEMA,
    FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
    ROBOT_NAME,
    SCENE_NAME,
    TASK_NAME,
)
from embodied_ai.data.lerobot_mapping import (
    LEROBOT_ACTION_KEY,
    LEROBOT_FRONT_IMAGE_KEY,
    LEROBOT_STATE_KEY,
)

_CONVERTER_DEPS_AVAILABLE = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("lerobot") is not None
)


@unittest.skipUnless(
    _CONVERTER_DEPS_AVAILABLE,
    "NumPy and LeRobot are intentionally absent from the dev lock",
)
class LeRobotConverterTest(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from embodied_ai.data.lerobot_converter import (
            CONVERSION_MANIFEST_PATH,
            convert_contract_episodes_to_lerobot,
        )
        from embodied_ai.sim.recording import NpyEpisodeRecorder

        self.np = np
        self.dataset_type = LeRobotDataset
        self.conversion_manifest_path = CONVERSION_MANIFEST_PATH
        self.convert = convert_contract_episodes_to_lerobot
        self.recorder_type = NpyEpisodeRecorder

    def _record_source_episode(self, output_root: Path) -> Path:
        recorder = self.recorder_type(
            output_root=output_root,
            episode_id="episode-stage7-unit-000001",
            task=TASK_NAME,
            robot=ROBOT_NAME,
            scene=SCENE_NAME,
            random_seed=7,
            observation_schema=FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
            action_schema=FRANKA_PICK_PLACE_ACTION_SCHEMA,
            provenance=EpisodeProvenance(
                simulator_name="test-simulator",
                simulator_version="1.0",
                repository_revision="test-revision",
                configuration_revision="test-configuration",
                environment_lock_sha256="a" * 64,
            ),
            instruction=DEFAULT_INSTRUCTION,
            instruction_id=DEFAULT_INSTRUCTION_ID,
            instruction_language=DEFAULT_INSTRUCTION_LANGUAGE,
            expert=ExpertMetadata(
                kind=ExpertKind.STATE_MACHINE,
                identifier="test-state-machine",
                revision="v1",
                configuration_revision="b" * 64,
            ),
        )
        for step in range(3):
            image = self.np.zeros((3, 224, 224), dtype=self.np.uint8)
            image[step % 3] = 64 + step
            recorder.append(
                {
                    "robot.joint_position": self.np.full(
                        (9,), step / 10, dtype=self.np.float32
                    ),
                    "robot.joint_velocity": self.np.zeros((9,), dtype=self.np.float32),
                    "object.cube.position": self.np.asarray(
                        [0.5, 0.0, 0.03], dtype=self.np.float32
                    ),
                    "camera.front.rgb": image,
                },
                self.np.zeros((7,), dtype=self.np.float32),
                step * 50_000_000,
            )
        return recorder.finalize(EpisodeOutcome.SUCCESS, termination_reason=None).directory

    def test_converts_reopens_and_preserves_source_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._record_source_episode(root / "contract")
            original_files = tuple(
                sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file())
            )
            output = root / "lerobot" / "stage7-unit"
            repo_id = "embodiedai/stage7-unit"

            result = self.convert(
                (source,),
                output,
                repo_id=repo_id,
                use_videos=False,
            )

            self.assertEqual(result.episode_count, 1)
            self.assertEqual(result.frame_count, 3)
            self.assertEqual(result.fps, 20)
            self.assertFalse(result.use_videos)
            self.assertEqual(
                tuple(
                    sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file())
                ),
                original_files,
            )
            self.assertEqual(list(output.parent.glob(".*.partial-*")), [])

            conversion = json.loads(
                (output / self.conversion_manifest_path).read_text(encoding="utf-8")
            )
            self.assertEqual(conversion["episode_count"], 1)
            self.assertEqual(conversion["frame_count"], 3)
            self.assertEqual(conversion["source_episodes"][0]["source_episode_id"], source.name)
            self.assertEqual(
                conversion["source_episodes"][0]["instruction"], DEFAULT_INSTRUCTION
            )
            self.assertEqual(
                conversion["source_episodes"][0]["expert"]["kind"], "state_machine"
            )

            reloaded = self.dataset_type(repo_id, root=output)
            self.assertEqual(len(reloaded), 3)
            self.assertEqual(reloaded.num_episodes, 1)
            sample = reloaded[2]
            self.np.testing.assert_allclose(
                sample[LEROBOT_STATE_KEY].numpy(),
                self.np.full((9,), 0.2, dtype=self.np.float32),
            )
            self.np.testing.assert_allclose(
                sample[LEROBOT_ACTION_KEY].numpy(), self.np.zeros((7,), dtype=self.np.float32)
            )
            self.assertEqual(tuple(sample[LEROBOT_FRONT_IMAGE_KEY].shape), (3, 224, 224))
            self.assertEqual(sample["task"], DEFAULT_INSTRUCTION)

    def test_refuses_to_overwrite_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._record_source_episode(root / "contract")
            output = root / "existing"
            output.mkdir()

            with self.assertRaises(FileExistsError):
                self.convert(
                    (source,),
                    output,
                    repo_id="embodiedai/stage7-existing",
                    use_videos=False,
                )


if __name__ == "__main__":
    unittest.main()
