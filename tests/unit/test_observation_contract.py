"""Unit tests for dependency-light observation contracts."""

import json
import unittest
from dataclasses import FrozenInstanceError
from typing import cast

from embodied_ai.contracts import (
    OBSERVATION_SCHEMA_VERSION,
    DataType,
    ObservationComponent,
    ObservationField,
    ObservationKind,
    ObservationSchema,
)


def make_schema() -> ObservationSchema:
    return ObservationSchema(
        fields=(
            ObservationField(
                key="robot.joint_position",
                kind=ObservationKind.STATE,
                shape=(2,),
                dtype=DataType.FLOAT32,
                axes=("component",),
                components=(
                    ObservationComponent(name="panda_joint1", unit="rad"),
                    ObservationComponent(name="panda_joint2", unit="rad"),
                ),
            ),
            ObservationField(
                key="camera.front.rgb",
                kind=ObservationKind.RGB_IMAGE,
                shape=(3, 480, 640),
                dtype=DataType.UINT8,
                axes=("channel", "height", "width"),
                frame="camera_front_optical",
            ),
        )
    )


class ObservationContractTest(unittest.TestCase):
    def test_json_round_trip_preserves_schema(self) -> None:
        schema = make_schema()
        encoded = json.loads(json.dumps(schema.to_dict()))

        self.assertEqual(ObservationSchema.from_dict(encoded), schema)
        self.assertEqual(schema.schema_version, OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(schema.field("camera.front.rgb").shape, (3, 480, 640))

    def test_additive_fields_are_accepted_within_v1(self) -> None:
        encoded = make_schema().to_dict()
        encoded["future_optional_field"] = "ignored-by-v1-reader"
        fields = cast(list[dict[str, object]], encoded["fields"])
        first_field = fields[0]
        first_field["future_optional_field"] = True

        self.assertEqual(ObservationSchema.from_dict(encoded), make_schema())

    def test_unknown_schema_version_is_rejected(self) -> None:
        encoded = make_schema().to_dict()
        encoded["schema_version"] = "embodied-ai.observation/v2"

        with self.assertRaisesRegex(ValueError, "unsupported schema version"):
            ObservationSchema.from_dict(encoded)

    def test_rgb_layout_and_frame_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "3-channel CHW"):
            ObservationField(
                key="camera.front.rgb",
                kind=ObservationKind.RGB_IMAGE,
                shape=(480, 640, 3),
                dtype=DataType.UINT8,
                axes=("height", "width", "channel"),
                frame="camera_front_optical",
            )

        with self.assertRaisesRegex(ValueError, "explicit camera frame"):
            ObservationField(
                key="camera.front.rgb",
                kind=ObservationKind.RGB_IMAGE,
                shape=(3, 480, 640),
                dtype=DataType.UINT8,
                axes=("channel", "height", "width"),
            )

    def test_state_component_count_must_match_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "one component per scalar"):
            ObservationField(
                key="robot.joint_position",
                kind=ObservationKind.STATE,
                shape=(2,),
                dtype=DataType.FLOAT32,
                axes=("component",),
                components=(ObservationComponent(name="panda_joint1", unit="rad"),),
            )

    def test_contracts_are_frozen_and_require_immutable_sequences(self) -> None:
        schema = make_schema()
        with self.assertRaises(FrozenInstanceError):
            attribute_name = "schema_version"
            setattr(schema, attribute_name, "changed")
        with self.assertRaisesRegex(ValueError, "must be a tuple"):
            mutable_fields = cast(tuple[ObservationField, ...], list(schema.fields))
            ObservationSchema(fields=mutable_fields)


if __name__ == "__main__":
    unittest.main()
