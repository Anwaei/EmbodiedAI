"""Unit tests for the task-specific Franka pick-and-place contract."""

import json
import unittest

from embodied_ai.contracts import ActionSchema, ObservationSchema
from embodied_ai.contracts.tasks.franka_pick_place import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONTROL_HZ,
    FRANKA_PICK_PLACE_ACTION_SCHEMA,
    FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
    IK_ROTATION_SCALE_RAD,
    IK_TRANSLATION_SCALE_M,
)


class FrankaPickPlaceContractTest(unittest.TestCase):
    def test_observation_keys_shapes_and_json_round_trip(self) -> None:
        schema = FRANKA_PICK_PLACE_OBSERVATION_SCHEMA

        self.assertEqual(
            tuple(field.key for field in schema.fields),
            (
                "robot.joint_position",
                "robot.joint_velocity",
                "object.cube.position",
                "camera.front.rgb",
            ),
        )
        self.assertEqual(schema.field("robot.joint_position").shape, (9,))
        self.assertEqual(schema.field("robot.joint_velocity").shape, (9,))
        self.assertEqual(schema.field("object.cube.position").shape, (3,))
        self.assertEqual(
            schema.field("camera.front.rgb").shape,
            (3, CAMERA_HEIGHT, CAMERA_WIDTH),
        )
        encoded = json.loads(json.dumps(schema.to_dict()))
        self.assertEqual(ObservationSchema.from_dict(encoded), schema)

    def test_action_contract_matches_normalized_ik_and_gripper_interface(self) -> None:
        schema = FRANKA_PICK_PLACE_ACTION_SCHEMA

        self.assertEqual(schema.dimension, 7)
        self.assertEqual(schema.control_hz, CONTROL_HZ)
        self.assertTrue(schema.normalized)
        self.assertEqual(schema.frame, "robot_base")
        self.assertEqual(
            tuple(component.name for component in schema.components),
            (
                "delta_x",
                "delta_y",
                "delta_z",
                "delta_roll",
                "delta_pitch",
                "delta_yaw",
                "gripper",
            ),
        )
        self.assertGreater(IK_TRANSLATION_SCALE_M, 0.0)
        self.assertGreater(IK_ROTATION_SCALE_RAD, 0.0)
        encoded = json.loads(json.dumps(schema.to_dict()))
        self.assertEqual(ActionSchema.from_dict(encoded), schema)


if __name__ == "__main__":
    unittest.main()
