"""Tests for the versioned Contract to LeRobot mapping profile."""

import unittest
from dataclasses import replace

from embodied_ai.data.lerobot_mapping import (
    FRANKA_PICK_PLACE_LEROBOT_MAPPING,
    LEROBOT_ACTION_KEY,
    LEROBOT_FRONT_IMAGE_KEY,
    LEROBOT_MAPPING_SCHEMA_VERSION,
    LEROBOT_STATE_KEY,
)


class LeRobotMappingTest(unittest.TestCase):
    def test_franka_mapping_is_complete_and_versioned(self) -> None:
        mapping = FRANKA_PICK_PLACE_LEROBOT_MAPPING

        self.assertEqual(mapping.schema_version, LEROBOT_MAPPING_SCHEMA_VERSION)
        self.assertEqual(mapping.profile, "franka-pick-place-smolvla-v1")
        self.assertEqual(mapping.state_source_keys, ("robot.joint_position",))
        self.assertEqual(mapping.state_dimension, 9)
        self.assertEqual(len(mapping.state_component_names), 9)
        self.assertEqual(len(mapping.action_component_names), 7)
        self.assertEqual(
            {field.source_key for field in mapping.excluded_observations},
            {"robot.joint_velocity", "object.cube.position"},
        )

    def test_lerobot_features_match_smolvla_keys_and_contract_shapes(self) -> None:
        image_features = FRANKA_PICK_PLACE_LEROBOT_MAPPING.lerobot_features(
            use_videos=False
        )
        video_features = FRANKA_PICK_PLACE_LEROBOT_MAPPING.lerobot_features(
            use_videos=True
        )

        self.assertEqual(image_features[LEROBOT_STATE_KEY]["shape"], (9,))
        self.assertEqual(image_features[LEROBOT_ACTION_KEY]["shape"], (7,))
        self.assertEqual(image_features[LEROBOT_FRONT_IMAGE_KEY]["shape"], (3, 224, 224))
        self.assertEqual(image_features[LEROBOT_FRONT_IMAGE_KEY]["dtype"], "image")
        self.assertEqual(video_features[LEROBOT_FRONT_IMAGE_KEY]["dtype"], "video")

    def test_mapping_rejects_an_unclassified_contract_observation(self) -> None:
        mapping = FRANKA_PICK_PLACE_LEROBOT_MAPPING

        with self.assertRaisesRegex(ValueError, "classify every observation"):
            replace(mapping, excluded_observations=mapping.excluded_observations[:1])

    def test_serialized_mapping_records_policy_exclusions(self) -> None:
        serialized = FRANKA_PICK_PLACE_LEROBOT_MAPPING.to_dict()

        self.assertEqual(serialized["state"]["target_key"], LEROBOT_STATE_KEY)
        self.assertEqual(serialized["action"]["target_key"], LEROBOT_ACTION_KEY)
        excluded = {
            item["source_key"]: item["reason"]
            for item in serialized["excluded_observations"]
        }
        self.assertIn("privileged simulator state", excluded["object.cube.position"])


if __name__ == "__main__":
    unittest.main()
