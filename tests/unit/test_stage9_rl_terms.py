"""Pure Torch checks for Stage 9 state, phase, and reward terms."""

from __future__ import annotations

import importlib.util

import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="test requires the Isaac Torch runtime",
)


def test_state_observation_has_reviewed_order_and_dimension() -> None:
    import torch

    from embodied_ai.sim.tasks.franka_pick_place.rl_terms import assemble_state_observation

    batch = 2
    values = {
        "joint_position": torch.full((batch, 9), 1.0),
        "joint_velocity": torch.full((batch, 9), 2.0),
        "tool_position": torch.full((batch, 3), 3.0),
        "tool_quaternion": torch.tensor([[-1.0, 0.0, 0.0, 0.0]]).repeat(batch, 1),
        "cube_position": torch.full((batch, 3), 4.0),
        "cube_linear_velocity": torch.full((batch, 3), 5.0),
        "goal_position": torch.full((batch, 3), 6.0),
        "previous_action": torch.full((batch, 7), 7.0),
        "phase_one_hot": torch.nn.functional.one_hot(
            torch.tensor([0, 4]), num_classes=5
        ).float(),
    }

    observation = assemble_state_observation(**values)

    assert observation.shape == (2, 52)
    assert torch.all(observation[:, :9] == 1.0)
    assert torch.all(observation[:, 18:21] == 3.0)
    assert torch.all(observation[:, 21] == 1.0)
    assert torch.all(observation[:, 34:37] == 1.0)
    assert torch.all(observation[:, 37:40] == 2.0)
    assert torch.all(observation[:, 40:47] == 7.0)


def test_phase_is_monotonic_and_advances_at_most_once_per_call() -> None:
    import torch

    from embodied_ai.contracts.rl import PickPlaceRlPhase
    from embodied_ai.sim.tasks.franka_pick_place.rl_terms import advance_phase

    phase = torch.tensor([int(PickPlaceRlPhase.REACH)])
    common = {
        "tool_to_cube_distance_m": torch.tensor([0.01]),
        "bilateral_contact": torch.tensor([True]),
        "cube_height_m": torch.tensor([0.12]),
        "cube_to_goal_distance_m": torch.tensor([0.01]),
        "gripper_open": torch.tensor([True]),
        "reach_distance_m": 0.065,
        "lift_height_m": 0.09,
        "place_distance_m": 0.08,
    }

    expected = (
        PickPlaceRlPhase.GRASP,
        PickPlaceRlPhase.LIFT,
        PickPlaceRlPhase.PLACE,
        PickPlaceRlPhase.RELEASE,
    )
    for expected_phase in expected:
        phase = advance_phase(phase, **common)
        assert phase.item() == int(expected_phase)

    phase = advance_phase(
        phase,
        **{**common, "bilateral_contact": torch.tensor([False])},
    )
    assert phase.item() == int(PickPlaceRlPhase.RELEASE)


def test_reward_terms_are_gated_and_penalties_are_non_negative() -> None:
    import torch

    from embodied_ai.contracts.rl import PickPlaceRlPhase
    from embodied_ai.sim.tasks.franka_pick_place import rl_terms

    phase = torch.tensor(
        [
            int(PickPlaceRlPhase.REACH),
            int(PickPlaceRlPhase.GRASP),
            int(PickPlaceRlPhase.LIFT),
        ]
    )
    distance = torch.tensor([0.05, 0.05, 0.05])

    reach = rl_terms.reach_reward(distance, phase, std_m=0.1)
    grasp = rl_terms.grasp_reward(torch.ones(3, dtype=torch.bool), phase)
    lift = rl_terms.lift_reward(
        torch.full((3,), 0.12),
        phase,
        resting_height_m=0.03,
        target_height_m=0.15,
    )
    place = rl_terms.place_reward(distance, phase, std_m=0.1)
    assert reach[0] > 0.0 and torch.all(reach[1:] == 0.0)
    assert grasp[1] > 0.0 and grasp[0] == 0.0 and grasp[2] == 0.0
    assert lift[1] > 0.0 and lift[0] == 0.0 and lift[2] == 0.0
    assert torch.all(place[:2] == 0.0) and place[2] > 0.0

    current = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    previous = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    assert rl_terms.action_magnitude_penalty(current).item() > 0.0
    assert rl_terms.action_rate_penalty(current, previous).item() > 0.0
    assert rl_terms.gripper_toggle_penalty(current, previous).item() == 1.0
