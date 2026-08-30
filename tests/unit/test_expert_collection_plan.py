"""Tests for the dependency-light Stage 6 expert collection matrix."""

import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from embodied_ai.contracts import ExpertCollectionPlan
from embodied_ai.contracts.tasks.franka_pick_place import (
    FrankaPickPlaceEpisodeParameters,
)


class ExpertCollectionPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan_path = (
            Path(__file__).resolve().parents[2]
            / "configs/sim/franka_pick_place/expert_collection_v1.toml"
        )
        cls.plan = ExpertCollectionPlan.from_toml(cls.plan_path)

    def test_first_matrix_is_complete_and_controlled(self) -> None:
        self.assertEqual(len(self.plan.episodes), 20)
        self.assertEqual(self.plan.task, "franka-pick-place")
        self.assertEqual(
            self.plan.expert_identifier,
            "franka-pick-place-state-machine",
        )
        self.assertEqual(len({episode.episode_id for episode in self.plan.episodes}), 20)
        self.assertEqual(len({episode.seed for episode in self.plan.episodes}), 20)
        self.assertEqual(len({episode.instruction_id for episode in self.plan.episodes}), 5)
        self.assertEqual(
            len(
                {
                    episode.parameters.cube_reset_position_env_m
                    for episode in self.plan.episodes
                }
            ),
            5,
        )
        self.assertEqual(
            len(
                {
                    episode.parameters.goal_position_env_m
                    for episode in self.plan.episodes
                }
            ),
            4,
        )

    def test_plan_dictionary_round_trip(self) -> None:
        self.assertEqual(ExpertCollectionPlan.from_dict(self.plan.to_dict()), self.plan)

    def test_duplicate_episode_or_inconsistent_instruction_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode IDs"):
            replace(self.plan, episodes=(self.plan.episodes[0], self.plan.episodes[0]))

        conflicting = replace(
            self.plan.episodes[1],
            instruction_id=self.plan.episodes[0].instruction_id,
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            replace(self.plan, episodes=(self.plan.episodes[0], conflicting))

    def test_positions_must_stay_on_the_reviewed_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside reviewed bounds"):
            FrankaPickPlaceEpisodeParameters(
                cube_reset_position_env_m=(0.1, 0.0, 0.03),
                goal_position_env_m=(0.65, -0.2, 0.03),
            )


class ExpandedExpertCollectionPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan_path = (
            Path(__file__).resolve().parents[2]
            / "configs/sim/franka_pick_place/expert_collection_v2_100.toml"
        )
        cls.plan = ExpertCollectionPlan.from_toml(cls.plan_path)

    def test_expanded_matrix_has_balanced_full_spatial_coverage(self) -> None:
        self.assertEqual(len(self.plan.episodes), 100)
        self.assertEqual(
            self.plan.collection_id,
            "stage6-franka-pick-place-expert-v2-100",
        )
        self.assertEqual(
            len({episode.episode_id for episode in self.plan.episodes}),
            100,
        )
        self.assertEqual(len({episode.seed for episode in self.plan.episodes}), 100)

        cubes = {
            episode.parameters.cube_reset_position_env_m
            for episode in self.plan.episodes
        }
        goals = {
            episode.parameters.goal_position_env_m
            for episode in self.plan.episodes
        }
        spatial_pairs = {
            (
                episode.parameters.cube_reset_position_env_m,
                episode.parameters.goal_position_env_m,
            )
            for episode in self.plan.episodes
        }
        self.assertEqual(len(cubes), 10)
        self.assertEqual(len(goals), 10)
        self.assertEqual(len(spatial_pairs), 100)

        instruction_counts = Counter(
            episode.instruction_id for episode in self.plan.episodes
        )
        self.assertEqual(len(instruction_counts), 5)
        self.assertEqual(set(instruction_counts.values()), {20})
        for cube in cubes:
            per_cube = Counter(
                episode.instruction_id
                for episode in self.plan.episodes
                if episode.parameters.cube_reset_position_env_m == cube
            )
            self.assertEqual(set(per_cube.values()), {2})
        for goal in goals:
            per_goal = Counter(
                episode.instruction_id
                for episode in self.plan.episodes
                if episode.parameters.goal_position_env_m == goal
            )
            self.assertEqual(set(per_goal.values()), {2})

    def test_expanded_plan_dictionary_round_trip(self) -> None:
        self.assertEqual(ExpertCollectionPlan.from_dict(self.plan.to_dict()), self.plan)


if __name__ == "__main__":
    unittest.main()
