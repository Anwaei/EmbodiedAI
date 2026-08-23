"""Unit tests for dependency-light action contracts."""

import json
import unittest
from typing import cast

from embodied_ai.contracts import (
    ACTION_SCHEMA_VERSION,
    ActionComponent,
    ActionRepresentation,
    ActionSchema,
    DataType,
)


def make_schema() -> ActionSchema:
    return ActionSchema(
        representation=ActionRepresentation.END_EFFECTOR_DELTA_POSE,
        components=(
            ActionComponent("delta_x", "m", -0.05, 0.05, "robot_base"),
            ActionComponent("delta_y", "m", -0.05, 0.05, "robot_base"),
            ActionComponent("delta_z", "m", -0.05, 0.05, "robot_base"),
            ActionComponent("delta_roll", "rad", -0.2, 0.2, "robot_base"),
            ActionComponent("delta_pitch", "rad", -0.2, 0.2, "robot_base"),
            ActionComponent("delta_yaw", "rad", -0.2, 0.2, "robot_base"),
            ActionComponent("gripper", "1", -1.0, 1.0),
        ),
        control_hz=20.0,
        frame="robot_base",
    )


class ActionContractTest(unittest.TestCase):
    def test_dimension_is_derived_and_json_round_trip_is_lossless(self) -> None:
        schema = make_schema()
        encoded = json.loads(json.dumps(schema.to_dict()))

        self.assertEqual(schema.dimension, 7)
        self.assertEqual(schema.schema_version, ACTION_SCHEMA_VERSION)
        self.assertEqual(ActionSchema.from_dict(encoded), schema)

    def test_declared_dimension_must_match_components(self) -> None:
        encoded = make_schema().to_dict()
        encoded["dimension"] = 6

        with self.assertRaisesRegex(ValueError, "dimension"):
            ActionSchema.from_dict(encoded)

    def test_end_effector_actions_require_a_frame(self) -> None:
        schema = make_schema()
        with self.assertRaisesRegex(ValueError, "explicit frame"):
            ActionSchema(
                representation=schema.representation,
                components=schema.components,
                control_hz=schema.control_hz,
            )

    def test_component_bounds_must_be_finite_and_ordered(self) -> None:
        with self.assertRaisesRegex(ValueError, "less than"):
            ActionComponent("joint1", "rad", 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            ActionComponent("joint1", "rad", float("-inf"), 1.0)

    def test_normalized_actions_have_canonical_units_and_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "bounds"):
            ActionSchema(
                representation=ActionRepresentation.JOINT_POSITION,
                components=(ActionComponent("joint1", "rad", -1.0, 1.0),),
                control_hz=20.0,
                dtype=DataType.FLOAT32,
                normalized=True,
            )

    def test_actions_use_a_float_storage_type(self) -> None:
        schema = make_schema()
        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            ActionSchema(
                representation=schema.representation,
                components=schema.components,
                control_hz=schema.control_hz,
                dtype=DataType.UINT8,
                frame=schema.frame,
            )

        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            ActionSchema(
                representation=schema.representation,
                components=schema.components,
                control_hz=schema.control_hz,
                dtype=cast(DataType, "float32"),
                frame=schema.frame,
            )


if __name__ == "__main__":
    unittest.main()
