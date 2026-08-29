"""Isaac-runtime tests for per-episode goal and reset parameter binding."""

import importlib.util
import os
import unittest

_ISAAC_AVAILABLE = importlib.util.find_spec("isaaclab") is not None
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_ISAAC_APP_ACTIVE = os.environ.get("EMBODIEDAI_ISAAC_APP_ACTIVE") == "1"


@unittest.skipUnless(
    _ISAAC_AVAILABLE and _TORCH_AVAILABLE and _ISAAC_APP_ACTIVE,
    "test requires an active Isaac AppLauncher",
)
class FrankaPickPlaceEpisodeParametersTest(unittest.TestCase):
    def test_custom_goal_drives_evaluation(self) -> None:
        import torch

        from embodied_ai.sim.tasks.franka_pick_place.evaluation import evaluate_cube_state

        goal = (0.62, -0.18, 0.03)
        cube_position = torch.tensor(
            [[0.62, -0.18, 0.03], [0.65, -0.10, 0.03]],
            dtype=torch.float32,
        )
        result = evaluate_cube_state(
            cube_position,
            torch.zeros_like(cube_position),
            torch.ones(2, dtype=torch.bool),
            goal_position_env_m=goal,
        )

        self.assertEqual(result.success.tolist(), [True, False])
        self.assertTrue(torch.allclose(result.goal_position_env_m[0], torch.tensor(goal)))

    def test_scene_marker_reset_and_termination_share_one_specification(self) -> None:
        from embodied_ai.contracts.tasks.franka_pick_place import (
            GOAL_MARKER_SIZE_M,
            FrankaPickPlaceEpisodeParameters,
        )
        from embodied_ai.sim.tasks.franka_pick_place.env_cfg import (
            FrankaPickPlaceEnvCfg,
            apply_episode_parameters,
        )

        parameters = FrankaPickPlaceEpisodeParameters(
            cube_reset_position_env_m=(0.46, 0.05, 0.03),
            goal_position_env_m=(0.62, -0.18, 0.03),
        )
        env_cfg = FrankaPickPlaceEnvCfg()
        apply_episode_parameters(env_cfg, parameters)

        self.assertEqual(env_cfg.scene.cube.init_state.pos, parameters.cube_reset_position_env_m)
        self.assertEqual(
            env_cfg.scene.goal_marker.init_state.pos,
            (0.62, -0.18, GOAL_MARKER_SIZE_M[2] / 2.0),
        )
        self.assertEqual(
            env_cfg.terminations.success.params["goal_position_env_m"],
            parameters.goal_position_env_m,
        )


if __name__ == "__main__":
    unittest.main()
