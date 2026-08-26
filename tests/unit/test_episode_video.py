"""Tests for derived MP4 previews from immutable NPY episodes."""

import importlib.util
import shutil
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
    ObservationField,
    ObservationKind,
    ObservationSchema,
)

_NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@unittest.skipUnless(
    _NUMPY_AVAILABLE and _FFMPEG_AVAILABLE,
    "NumPy and FFmpeg are required for the episode video test",
)
class EpisodeVideoTest(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        from embodied_ai.data.episode_video import encode_episode_camera_video
        from embodied_ai.sim.recording import NpyEpisodeRecorder

        self.np = np
        self.encode_video = encode_episode_camera_video
        self.recorder_type = NpyEpisodeRecorder
        self.observation_schema = ObservationSchema(
            fields=(
                ObservationField(
                    key="camera.front.rgb",
                    kind=ObservationKind.RGB_IMAGE,
                    shape=(3, 4, 6),
                    dtype=DataType.UINT8,
                    axes=("channel", "height", "width"),
                    frame="test-camera-optical",
                ),
            )
        )
        self.action_schema = ActionSchema(
            representation=ActionRepresentation.JOINT_POSITION,
            components=(ActionComponent("joint-1", "rad", -1.0, 1.0),),
            control_hz=20.0,
        )

    def _record_episode(self, output_root: Path) -> Path:
        recorder = self.recorder_type(
            output_root=output_root,
            episode_id="episode-video-test",
            task="test-task",
            robot="test-robot",
            scene="test-scene",
            random_seed=1,
            observation_schema=self.observation_schema,
            action_schema=self.action_schema,
            provenance=EpisodeProvenance(
                simulator_name="test-simulator",
                simulator_version="1.0",
                repository_revision="test-revision",
                configuration_revision="test-configuration",
                environment_lock_sha256="a" * 64,
            ),
        )
        for step in range(3):
            frame = self.np.zeros((3, 4, 6), dtype=self.np.uint8)
            frame[step % 3] = 64 + step * 64
            recorder.append(
                {"camera.front.rgb": frame},
                self.np.asarray([0.0], dtype=self.np.float32),
                step * 50_000_000,
            )
        return recorder.finalize(
            EpisodeOutcome.TRUNCATED,
            termination_reason="unit-test-limit",
        ).directory

    def test_encodes_validated_camera_payload_without_mutating_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode = self._record_episode(root / "datasets")
            original_files = tuple(
                sorted(path.relative_to(episode) for path in episode.rglob("*"))
            )
            output = root / "artifacts" / "preview.mp4"

            encoded = self.encode_video(episode, output)

            self.assertEqual(encoded.path, output.resolve())
            self.assertEqual(encoded.frame_count, 3)
            self.assertEqual((encoded.width, encoded.height), (6, 4))
            self.assertEqual(encoded.fps, 20.0)
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(
                tuple(sorted(path.relative_to(episode) for path in episode.rglob("*"))),
                original_files,
            )

    def test_rejects_video_inside_immutable_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            episode = self._record_episode(Path(temporary_directory) / "datasets")

            with self.assertRaisesRegex(ValueError, "outside the immutable episode"):
                self.encode_video(episode, episode / "preview.mp4")


if __name__ == "__main__":
    unittest.main()
