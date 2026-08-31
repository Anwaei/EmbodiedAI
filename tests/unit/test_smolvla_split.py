"""Tests for the frozen Stage 7 train/validation split."""

import json
import unittest
from pathlib import Path

from embodied_ai.policies.smolvla.config import Stage7RunConfig, Stage7Step6BConfig
from embodied_ai.policies.smolvla.split import Stage7EpisodeSplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG_PATH = REPOSITORY_ROOT / "configs/policy/smolvla_franka_pick_place_v1.toml"
FORMAL_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/policy/smolvla_franka_pick_place_v2_100_step6b.toml"
)


class Stage7SplitTest(unittest.TestCase):
    def test_reviewed_config_and_split_are_consistent(self) -> None:
        config = Stage7RunConfig.from_toml(RUN_CONFIG_PATH, repository_root=REPOSITORY_ROOT)
        split = Stage7EpisodeSplit.from_json(config.split_path)

        self.assertEqual(split.dataset_repo_id, config.dataset_repo_id)
        self.assertEqual(split.mapping_profile, config.mapping_profile)
        self.assertEqual(len(split.train_episode_indices), 15)
        self.assertEqual(len(split.validation_episode_indices), 5)
        self.assertEqual(
            set(split.train_episode_indices) | set(split.validation_episode_indices),
            set(range(20)),
        )
        self.assertEqual(split.validation_episode_indices, (0, 6, 7, 18, 9))
        self.assertLessEqual(config.max_optimizer_steps, 200)

    def test_split_rejects_frame_leakage(self) -> None:
        with RUN_CONFIG_PATH.with_name("smolvla_stage7_split_v1.json").open(
            encoding="utf-8"
        ) as stream:
            value = json.load(stream)
        value["validation_episode_indices"][0] = value["train_episode_indices"][0]
        temporary = REPOSITORY_ROOT / "tests/unit/.stage7-split-invalid.json"
        try:
            temporary.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disjoint|cover"):
                Stage7EpisodeSplit.from_json(temporary)
        finally:
            temporary.unlink(missing_ok=True)

    def test_formal_split_is_complete_and_disjoint(self) -> None:
        formal = Stage7Step6BConfig.from_toml(
            FORMAL_CONFIG_PATH,
            repository_root=REPOSITORY_ROOT,
        )
        split = Stage7EpisodeSplit.from_json(formal.run.split_path)

        self.assertEqual(formal.epochs, 5)
        self.assertEqual(len(split.train_episode_indices), 80)
        self.assertEqual(len(split.validation_episode_indices), 10)
        self.assertEqual(len(split.test_episode_indices), 10)
        combined = (
            split.train_episode_indices
            + split.validation_episode_indices
            + split.test_episode_indices
        )
        self.assertEqual(set(combined), set(range(100)))
        self.assertEqual(len(combined), len(set(combined)))


if __name__ == "__main__":
    unittest.main()
