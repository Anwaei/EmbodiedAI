"""Unit tests for dependency-light episode metadata contracts."""

import json
import unittest
from dataclasses import replace

from embodied_ai.contracts import (
    EPISODE_SCHEMA_VERSION,
    ActionComponent,
    ActionRepresentation,
    ActionSchema,
    DataType,
    EpisodeManifest,
    EpisodeMetadata,
    EpisodeOutcome,
    EpisodeProvenance,
    ExpertKind,
    ExpertMetadata,
    ObservationComponent,
    ObservationField,
    ObservationKind,
    ObservationSchema,
    PayloadFile,
)

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def make_expert_metadata() -> ExpertMetadata:
    return ExpertMetadata(
        kind=ExpertKind.STATE_MACHINE,
        identifier="franka-pick-place-state-machine",
        revision="v1",
        configuration_revision="d" * 64,
    )


def make_metadata(outcome: EpisodeOutcome = EpisodeOutcome.SUCCESS) -> EpisodeMetadata:
    observation_schema = ObservationSchema(
        fields=(
            ObservationField(
                key="robot.joint_position",
                kind=ObservationKind.STATE,
                shape=(1,),
                dtype=DataType.FLOAT32,
                axes=("component",),
                components=(ObservationComponent("panda_joint1", "rad"),),
            ),
        )
    )
    action_schema = ActionSchema(
        representation=ActionRepresentation.JOINT_POSITION,
        components=(ActionComponent("panda_joint1", "rad", -2.9, 2.9),),
        control_hz=20.0,
    )
    reason = None if outcome is EpisodeOutcome.SUCCESS else "time_limit"
    return EpisodeMetadata(
        episode_id="episode-000001",
        task="pick-and-place",
        robot="franka-panda",
        scene="table-cube-v1",
        random_seed=7,
        observation_schema=observation_schema,
        action_schema=action_schema,
        step_count=2,
        start_time_ns=0,
        end_time_ns=50_000_000,
        outcome=outcome,
        termination_reason=reason,
        payloads=(
            PayloadFile("actions/data.npy", "application/x-npy", 128, _DIGEST_A),
            PayloadFile(
                "actions/timestamps_ns.npy", "application/x-npy", 96, _DIGEST_B
            ),
        ),
        provenance=EpisodeProvenance(
            simulator_name="Isaac Sim",
            simulator_version="5.1.0",
            repository_revision="173f55b",
            configuration_revision="stage6-default-v1",
            environment_lock_sha256="c" * 64,
        ),
    )


class EpisodeContractTest(unittest.TestCase):
    def test_manifest_alias_and_json_round_trip(self) -> None:
        metadata = make_metadata()
        encoded = json.loads(json.dumps(metadata.to_dict()))

        self.assertIs(EpisodeManifest, EpisodeMetadata)
        self.assertEqual(metadata.schema_version, EPISODE_SCHEMA_VERSION)
        self.assertEqual(EpisodeMetadata.from_dict(encoded), metadata)

    def test_additive_v1_fields_are_accepted(self) -> None:
        encoded = make_metadata().to_dict()
        encoded["future_optional_field"] = {"value": 1}

        self.assertEqual(EpisodeMetadata.from_dict(encoded), make_metadata())

    def test_expert_demonstration_metadata_round_trip(self) -> None:
        metadata = replace(
            make_metadata(),
            task_parameters={"goal_position_env_m": [0.65, -0.20, 0.03]},
            reset_parameters={"cube_position_env_m": [0.50, 0.00, 0.03]},
            instruction="Pick up the cube and place it in the goal.",
            instruction_id="pick-place-cube-goal-en-001",
            instruction_language="en",
            expert=make_expert_metadata(),
        )

        encoded = json.loads(json.dumps(metadata.to_dict()))

        self.assertEqual(EpisodeMetadata.from_dict(encoded), metadata)
        self.assertEqual(encoded["expert"]["kind"], "state_machine")
        self.assertEqual(
            encoded["task_parameters"]["goal_position_env_m"],
            [0.65, -0.20, 0.03],
        )

    def test_episode_parameters_are_json_safe(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            replace(make_metadata(), task_parameters={"bad": float("nan")})
        with self.assertRaisesRegex(ValueError, "JSON-compatible"):
            replace(make_metadata(), reset_parameters={"bad": object()})

    def test_expert_demonstration_fields_are_atomic(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            replace(make_metadata(), instruction="Incomplete demonstration")

        with self.assertRaisesRegex(ValueError, "unsupported expert kind"):
            ExpertMetadata.from_dict(
                {
                    "kind": "unknown",
                    "identifier": "unknown-expert",
                    "revision": "v1",
                    "configuration_revision": "d" * 64,
                }
            )

    def test_failure_and_truncation_require_a_reason(self) -> None:
        for outcome in (EpisodeOutcome.FAILURE, EpisodeOutcome.TRUNCATED):
            with self.subTest(outcome=outcome):
                valid = make_metadata(outcome)
                with self.assertRaisesRegex(ValueError, "termination_reason"):
                    replace(valid, termination_reason=None)

    def test_payload_paths_are_relative_and_digests_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative POSIX"):
            PayloadFile("../escape.npy", "application/x-npy", 1, _DIGEST_A)
        with self.assertRaisesRegex(ValueError, "relative POSIX"):
            PayloadFile("actions\\data.npy", "application/x-npy", 1, _DIGEST_A)
        with self.assertRaisesRegex(ValueError, "relative POSIX"):
            PayloadFile(".", "application/x-npy", 1, _DIGEST_A)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            PayloadFile("actions.npy", "application/x-npy", 1, "not-a-digest")

    def test_time_range_and_step_count_are_validated(self) -> None:
        valid = make_metadata()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            replace(valid, end_time_ns=-1)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            replace(valid, step_count=0)


if __name__ == "__main__":
    unittest.main()
