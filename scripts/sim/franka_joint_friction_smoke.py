#!/usr/bin/env python3
"""Verify the reviewed joint-friction profile in a live Isaac Sim articulation."""

# Isaac Lab must start AppLauncher before simulator-dependent imports.
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher


artifacts_root = (
    Path(os.environ.get("EMBODIEDAI_ARTIFACTS", "/root/autodl-tmp/EmbodiedAI/artifacts"))
    .expanduser()
    .resolve()
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument(
    "--artifact_path",
    type=Path,
    default=artifacts_root / "sim-dynamics" / "franka_joint_friction_smoke.json",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import embodied_ai.sim.tasks  # noqa: F401
from embodied_ai.sim.tasks.franka_pick_place import RL_TASK_ID
from embodied_ai.sim.tasks.franka_pick_place.dynamics_cfg import (
    FRANKA_ARM_JOINT_NAMES,
    franka_joint_friction_identity,
    load_reviewed_franka_joint_friction_config,
)
from embodied_ai.sim.tasks.franka_pick_place.rl_env_cfg import FrankaPickPlacePPOEnvCfg


def _artifact_path(requested: Path) -> Path:
    candidate = requested.expanduser()
    if not candidate.is_absolute():
        candidate = artifacts_root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(artifacts_root):
        raise ValueError("--artifact_path must resolve under EMBODIEDAI_ARTIFACTS")
    return resolved


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _as_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().tolist()]


def main() -> None:
    if args_cli.num_envs < 1 or args_cli.steps < 1:
        raise ValueError("--num_envs and --steps must be positive")

    config = load_reviewed_franka_joint_friction_config()
    env_cfg = FrankaPickPlacePPOEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = 0
    env = gym.make(RL_TASK_ID, cfg=env_cfg)
    try:
        observations, _ = env.reset(seed=0)
        robot = env.unwrapped.scene["robot"]
        joint_ids, joint_names = robot.find_joints(
            list(FRANKA_ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(joint_names) != FRANKA_ARM_JOINT_NAMES:
            raise RuntimeError(f"unexpected arm joint order: {joint_names}")

        expected_coulomb = torch.tensor(
            config.arm.coulomb, device=env.unwrapped.device, dtype=torch.float32
        )
        expected_viscous = torch.tensor(
            config.arm.viscous, device=env.unwrapped.device, dtype=torch.float32
        )
        static = robot.data.joint_friction_coeff[0, joint_ids]
        dynamic = robot.data.joint_dynamic_friction_coeff[0, joint_ids]
        viscous = robot.data.joint_viscous_friction_coeff[0, joint_ids]
        torch.testing.assert_close(static, expected_coulomb, rtol=0.0, atol=1.0e-6)
        torch.testing.assert_close(dynamic, expected_coulomb, rtol=0.0, atol=1.0e-6)
        torch.testing.assert_close(viscous, expected_viscous, rtol=0.0, atol=1.0e-6)

        actions = torch.zeros(
            env.action_space.shape,
            dtype=torch.float32,
            device=env.unwrapped.device,
        )
        for _ in range(args_cli.steps):
            with torch.inference_mode():
                observations, rewards, _, _, _ = env.step(actions)
            if not torch.isfinite(observations["policy"]).all():
                raise RuntimeError("state observation became non-finite")
            if not torch.isfinite(rewards).all():
                raise RuntimeError("reward became non-finite")

        artifact_path = _artifact_path(args_cli.artifact_path)
        report = {
            "schema_version": "embodied-ai.franka-joint-friction-smoke/v1",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "passed",
            "task_id": RL_TASK_ID,
            "device": args_cli.device,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "versions": {
                "isaac_sim": version("isaacsim"),
                "isaac_lab": version("isaaclab"),
                "torch": version("torch"),
            },
            "joint_names": joint_names,
            "observed": {
                "static_coulomb_effort": _as_list(static),
                "dynamic_coulomb_effort": _as_list(dynamic),
                "viscous_coefficient": _as_list(viscous),
            },
            "friction_profile": franka_joint_friction_identity(),
        }
        _write_json_atomic(artifact_path, report)
        print(
            "FRANKA_JOINT_FRICTION_SMOKE_OK",
            f"profile={config.profile}",
            f"num_envs={args_cli.num_envs}",
            f"steps={args_cli.steps}",
            f"artifact={artifact_path}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
