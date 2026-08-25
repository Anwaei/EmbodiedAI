"""Tests for the simulator-side immutable NPY episode writer."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from embodied_ai.contracts import (
    ActionComponent,
    ActionRepresentation,
    ActionSchema,
    DataType,
    EpisodeOutcome,
    EpisodeProvenance,
    ObservationComponent,
    ObservationField,
    ObservationKind,
    ObservationSchema,
)

_NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(_NUMPY_AVAILABLE, "NumPy is intentionally absent from the dev lock")
class NpyEpisodeRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        from embodied_ai.sim.recording import NpyEpisodeRecorder, validate_npy_episode

        self.np = np
        self.recorder_type = NpyEpisodeRecorder
        self.validate_episode = validate_npy_episode
        self.observation_schema = ObservationSchema(
            fields=(
                ObservationField(
                    key="robot.state",
                    kind=ObservationKind.STATE,
                    shape=(2,),
                    dtype=DataType.FLOAT32,
                    axes=("component",),
                    components=(
                        ObservationComponent("position", "m"),
                        ObservationComponent("velocity", "m/s"),
                    ),
                ),
                ObservationField(
                    key="camera.front.rgb",
                    kind=ObservationKind.RGB_IMAGE,
                    shape=(3, 2, 2),
                    dtype=DataType.UINT8,
                    axes=("channel", "height", "width"),
                    frame="camera-front-optical",
                ),
            )
        )
        self.action_schema = ActionSchema(
            representation=ActionRepresentation.JOINT_POSITION,
            components=(ActionComponent("joint-1", "rad", -1.0, 1.0),),
            control_hz=20.0,
        )
        self.provenance = EpisodeProvenance(
            simulator_name="test-simulator",
            simulator_version="1.0",
            repository_revision="test-revision",
            configuration_revision="test-configuration",
            environment_lock_sha256="a" * 64,
        )

    def _make_recorder(self, output_root: Path):
        return self.recorder_type(
            output_root=output_root,
            episode_id="episode-000001",
            task="test-task",
            robot="test-robot",
            scene="test-scene",
            random_seed=7,
            observation_schema=self.observation_schema,
            action_schema=self.action_schema,
            provenance=self.provenance,
        )

    def _observation(self, value: float):
        return {
            "robot.state": self.np.asarray([value, 0.0], dtype=self.np.float32),
            "camera.front.rgb": self.np.full(
                (3, 2, 2), int(value), dtype=self.np.uint8
            ),
        }

    def test_round_trip_and_immutable_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            recorder = self._make_recorder(output_root)
            recorder.append(
                self._observation(1.0),
                self.np.asarray([0.0], dtype=self.np.float32),
                0,
            )
            recorder.append(
                self._observation(2.0),
                self.np.asarray([0.5], dtype=self.np.float32),
                50_000_000,
            )

            recorded = recorder.finalize(
                EpisodeOutcome.TRUNCATED,
                termination_reason="unit-test-limit",
            )
            validated = self.validate_episode(recorded.directory)

            self.assertEqual(validated, recorded.metadata)
            self.assertEqual(validated.step_count, 2)
            self.assertEqual(validated.start_time_ns, 0)
            self.assertEqual(validated.end_time_ns, 50_000_000)
            self.assertTrue((recorded.directory / "manifest.json").is_file())
            self.assertEqual(list(output_root.glob(".*.partial-*")), [])
            with self.assertRaisesRegex(RuntimeError, "finalized"):
                recorder.append(
                    self._observation(3.0),
                    self.np.asarray([0.0], dtype=self.np.float32),
                    100_000_000,
                )
            with self.assertRaises(FileExistsError):
                self._make_recorder(output_root)

    def test_rejects_non_monotonic_timestamps_and_out_of_range_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recorder = self._make_recorder(Path(temporary_directory))
            recorder.append(
                self._observation(1.0),
                self.np.asarray([0.0], dtype=self.np.float32),
                0,
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                recorder.append(
                    self._observation(2.0),
                    self.np.asarray([0.0], dtype=self.np.float32),
                    0,
                )
            with self.assertRaisesRegex(ValueError, "out of bounds"):
                recorder.append(
                    self._observation(2.0),
                    self.np.asarray([2.0], dtype=self.np.float32),
                    50_000_000,
                )


if __name__ == "__main__":
    unittest.main()
