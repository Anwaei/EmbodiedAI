#!/usr/bin/env python3
"""Reset, evaluate, step, and record one bounded Franka dummy episode."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
default_dataset_root = Path(
    os.environ.get("EMBODIEDAI_DATASETS", "/root/autodl-tmp/EmbodiedAI/datasets")
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=default_dataset_root / "stage6-smoke",
)
parser.add_argument("--episode_id", default="episode-stage6-smoke-000001")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv

import embodied_ai.sim.tasks  # noqa: F401
from embodied_ai.contracts import EpisodeOutcome, EpisodeProvenance
from embodied_ai.contracts.tasks.franka_pick_place import (
    CONTROL_HZ,
    CUBE_RESET_POSITION_ENV_M,
    FRANKA_PICK_PLACE_ACTION_SCHEMA,
    FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
)
from embodied_ai.sim.recording import NpyEpisodeRecorder, validate_npy_episode
from embodied_ai.sim.tasks.franka_pick_place import TASK_ID
from embodied_ai.sim.tasks.franka_pick_place.env_cfg import (
    CONTRACT_OBSERVATION_TERM_MAP,
    FrankaPickPlaceEnvCfg,
)
from embodied_ai.sim.tasks.franka_pick_place.evaluation import (
    PickPlaceEvaluation,
    evaluate_pick_place,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_revision(repository_root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return f"{revision}-dirty" if dirty else revision


def _configuration_revision(repository_root: Path) -> str:
    digest = hashlib.sha256()
    relative_paths = (
        "src/embodied_ai/contracts/tasks/franka_pick_place.py",
        "src/embodied_ai/sim/tasks/franka_pick_place/env_cfg.py",
        "src/embodied_ai/sim/tasks/franka_pick_place/evaluation.py",
        "src/embodied_ai/sim/tasks/franka_pick_place/mdp.py",
    )
    for relative_path in relative_paths:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((repository_root / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _policy_terms(observations: dict[str, object]) -> dict[str, torch.Tensor]:
    policy = observations.get("policy")
    if not isinstance(policy, dict):
        raise RuntimeError("expected unconcatenated policy observations")
    if not all(isinstance(value, torch.Tensor) for value in policy.values()):
        raise RuntimeError("policy observations must be torch tensors")
    return policy


def _contract_observation(
    observations: dict[str, object], env_index: int = 0
) -> dict[str, np.ndarray]:
    policy = _policy_terms(observations)
    return {
        field.key: (
            policy[CONTRACT_OBSERVATION_TERM_MAP[field.key]][env_index]
            .detach()
            .cpu()
            .numpy()
        )
        for field in FRANKA_PICK_PLACE_OBSERVATION_SCHEMA.fields
    }


def _assert_configured_reset(
    env: ManagerBasedEnv, observations: dict[str, object]
) -> None:
    policy = _policy_terms(observations)
    robot = env.scene["robot"]
    expected_values = {
        "robot.joint_position": robot.data.default_joint_pos,
        "robot.joint_velocity": robot.data.default_joint_vel,
        "object.cube.position": policy["cube_position"].new_tensor(
            CUBE_RESET_POSITION_ENV_M
        ).expand(1, -1),
    }
    for key, expected in expected_values.items():
        actual = policy[CONTRACT_OBSERVATION_TERM_MAP[key]]
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)


def _assert_non_terminal_reset(env: ManagerBasedEnv) -> PickPlaceEvaluation:
    evaluation = evaluate_pick_place(env)
    if evaluation.success.any() or evaluation.failure.any():
        raise RuntimeError("fixed reset must begin in a non-terminal state")
    return evaluation


def _make_provenance(repository_root: Path) -> EpisodeProvenance:
    return EpisodeProvenance(
        simulator_name="Isaac Sim",
        simulator_version=version("isaacsim"),
        repository_revision=_repository_revision(repository_root),
        configuration_revision=_configuration_revision(repository_root),
        environment_lock_sha256=_sha256(repository_root / "env/isaac/uv.lock"),
    )


def main() -> None:
    if args_cli.steps < 1:
        raise ValueError("--steps must be positive")
    if args_cli.seed < 0:
        raise ValueError("--seed must be non-negative")

    repository_root = Path(__file__).resolve().parents[2]
    env_cfg = FrankaPickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env = gym.make(TASK_ID, cfg=env_cfg)
    print("STAGE6_EPISODE_SMOKE_PHASE environment-created", flush=True)
    try:
        observations, _ = env.reset(seed=args_cli.seed)
        print("STAGE6_EPISODE_SMOKE_PHASE first-reset", flush=True)
        _assert_configured_reset(env.unwrapped, observations)
        initial_evaluation = _assert_non_terminal_reset(env.unwrapped)
        print("STAGE6_EPISODE_SMOKE_PHASE reset-validated", flush=True)

        actions = torch.zeros(
            env.action_space.shape,
            dtype=torch.float32,
            device=env.unwrapped.device,
        )

        recorder = NpyEpisodeRecorder(
            output_root=args_cli.output_root,
            episode_id=args_cli.episode_id,
            task="franka-pick-place",
            robot="franka-panda",
            scene="table-cube-goal-v1",
            random_seed=args_cli.seed,
            observation_schema=FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
            action_schema=FRANKA_PICK_PLACE_ACTION_SCHEMA,
            provenance=_make_provenance(repository_root),
        )
        print("STAGE6_EPISODE_SMOKE_PHASE recorder-created", flush=True)
        control_period_ns = round(1_000_000_000 / CONTROL_HZ)
        for step_index in range(args_cli.steps):
            recorder.append(
                _contract_observation(observations),
                actions[0].detach().cpu().numpy(),
                step_index * control_period_ns,
            )
            with torch.inference_mode():
                observations, _, terminated, truncated, _ = env.step(actions)
            if terminated.any() or truncated.any():
                raise RuntimeError("bounded dummy rollout terminated unexpectedly")

        final_evaluation = evaluate_pick_place(env.unwrapped)
        recorded = recorder.finalize(
            EpisodeOutcome.TRUNCATED,
            termination_reason="smoke-test-step-limit",
        )
        print("STAGE6_EPISODE_SMOKE_PHASE episode-published", flush=True)
        validated = validate_npy_episode(recorded.directory)
        if validated != recorded.metadata:
            raise RuntimeError("validated metadata differs from the recorded manifest")

        print(
            "STAGE6_EPISODE_SMOKE_OK",
            f"episode={recorded.directory}",
            f"steps={recorded.metadata.step_count}",
            f"reset_seed={args_cli.seed}",
            f"initial_goal_error_m={initial_evaluation.position_error_m[0].item():.6f}",
            f"final_goal_error_m={final_evaluation.position_error_m[0].item():.6f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
