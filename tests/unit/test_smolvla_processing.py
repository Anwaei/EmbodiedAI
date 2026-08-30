"""Dependency-light tests for SmolVLA profiles and processor statistics."""

import ast
import unittest
from pathlib import Path

import numpy as np

from embodied_ai.policies.smolvla.processing import (
    canonical_json_sha256,
    tensor_ready_statistics,
    validate_statistics,
)
from embodied_ai.policies.smolvla.profile import FRANKA_PICK_PLACE_SMOLVLA_PROFILE


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _statistics() -> dict[str, dict[str, object]]:
    profile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE
    return {
        profile.state_key: {
            "min": [-1.0] * 9,
            "max": [1.0] * 9,
            "mean": [0.0] * 9,
            "std": [0.5] * 9,
            "count": [15],
        },
        profile.action_key: {
            "min": [-1.0] * 7,
            "max": [1.0] * 7,
            "mean": [0.0] * 7,
            "std": [0.5] * 7,
            "count": [15],
        },
    }


class SmolVLAProcessingTest(unittest.TestCase):
    def test_profile_matches_contract_mapping(self) -> None:
        profile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE

        self.assertEqual(profile.state_dimension, 9)
        self.assertEqual(profile.image_shape, (3, 224, 224))
        self.assertEqual(profile.action_dimension, 7)
        self.assertEqual(profile.chunk_size, 50)
        self.assertIn("delta_x", profile.action_names)

    def test_statistics_are_valid_and_deterministically_hashed(self) -> None:
        statistics = _statistics()
        validate_statistics(statistics)
        converted = tensor_ready_statistics(statistics)

        self.assertEqual(converted["observation.state"]["mean"].dtype, np.float32)
        self.assertEqual(converted["action"]["count"].dtype, np.int64)
        self.assertEqual(canonical_json_sha256(statistics), canonical_json_sha256(statistics))

    def test_statistics_reject_checkpoint_dimensions(self) -> None:
        statistics = _statistics()
        statistics["observation.state"]["mean"] = [0.0] * 6

        with self.assertRaisesRegex(ValueError, "invalid shape"):
            validate_statistics(statistics)

    def test_vla_policy_package_has_no_simulator_import(self) -> None:
        package = REPOSITORY_ROOT / "src/embodied_ai/policies/smolvla"
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertFalse(
                any(
                    name == "embodied_ai.sim" or name.startswith("embodied_ai.sim.")
                    for name in imports
                ),
                path,
            )


if __name__ == "__main__":
    unittest.main()
