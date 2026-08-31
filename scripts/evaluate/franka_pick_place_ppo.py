#!/usr/bin/env python3
"""Evaluate Stage 9 PPO checkpoints and controls on frozen held-out scenarios."""

# Isaac Lab must start AppLauncher before simulator-dependent imports.
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path

from isaaclab.app import AppLauncher

repository_root = Path(
    os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI")
).resolve()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--controller",
    action="append",
    choices=("zero", "random", "checkpoint"),
    dest="controllers",
)
parser.add_argument("--checkpoint", action="append", type=Path, dest="checkpoints")
parser.add_argument("--output_name", required=True)
parser.add_argument("--random_seed", type=int, default=20260901)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import mdp as isaac_mdp
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import embodied_ai.sim.tasks  # noqa: F401
from embodied_ai.evaluation.config import load_scenarios
from embodied_ai.sim.tasks.franka_pick_place import RL_TASK_ID
from embodied_ai.sim.tasks.franka_pick_place.agents.rsl_rl_ppo_cfg import (
    FrankaPickPlacePPORunnerCfg,
)
from embodied_ai.sim.tasks.franka_pick_place.rl_env_cfg import (
    STAGE9_STANDALONE_CONFIG,
    FrankaPickPlacePPOEnvCfg,
)
from embodied_ai.sim.tasks.franka_pick_place.rl_mdp import (
    set_franka_pick_place_rl_geometry,
)

REPORT_SCHEMA_VERSION = "embodied-ai.stage9-ppo-evaluation/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _resolve_checkpoints(paths: list[Path]) -> tuple[Path, ...]:
    runs_root = Path(
        os.environ.get("EMBODIEDAI_RUNS", "/root/autodl-tmp/EmbodiedAI/runs")
    ).resolve()
    resolved: list[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve()
        if not candidate.is_relative_to(runs_root) or not candidate.is_file():
            raise ValueError("all checkpoints must be existing files under EMBODIEDAI_RUNS")
        resolved.append(candidate)
    return tuple(resolved)


def _reset_exact_scenarios(wrapped_env, scenarios) -> object:
    wrapped_env.reset()
    base = wrapped_env.unwrapped
    env_ids = torch.arange(base.num_envs, device=base.device, dtype=torch.long)
    isaac_mdp.reset_scene_to_default(base, env_ids)
    cube = torch.tensor(
        [scenario.cube_position_env_m for scenario in scenarios],
        dtype=torch.float32,
        device=base.device,
    )
    goal = torch.tensor(
        [scenario.goal_position_env_m for scenario in scenarios],
        dtype=torch.float32,
        device=base.device,
    )
    set_franka_pick_place_rl_geometry(
        base,
        env_ids,
        cube_position_env_m=cube,
        goal_position_env_m=goal,
    )
    base.episode_length_buf[:] = 0
    base.sim.forward()
    base.scene.update(base.physics_dt)
    return wrapped_env.get_observations()


def _controller_specs(
    controllers: tuple[str, ...], checkpoints: tuple[Path, ...]
) -> tuple[tuple[str, Path | None], ...]:
    specs: list[tuple[str, Path | None]] = []
    for controller in controllers:
        if controller == "checkpoint":
            for checkpoint in checkpoints:
                specs.append((f"checkpoint:{checkpoint.parent.name}/{checkpoint.stem}", checkpoint))
        else:
            specs.append((controller, None))
    return tuple(specs)


def _evaluate_controller(
    wrapped_env,
    scenarios,
    *,
    controller_id: str,
    checkpoint: Path | None,
    random_seed: int,
) -> dict[str, object]:
    base = wrapped_env.unwrapped
    obs = _reset_exact_scenarios(wrapped_env, scenarios)
    runner = None
    policy = None
    policy_nn = None
    checkpoint_record = None
    if checkpoint is not None:
        agent_cfg = FrankaPickPlacePPORunnerCfg()
        agent_cfg.device = base.device
        runner = OnPolicyRunner(
            wrapped_env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(str(checkpoint), load_optimizer=False, map_location=base.device)
        policy = runner.get_inference_policy(device=base.device)
        policy_nn = runner.alg.policy
        checkpoint_record = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "iteration": int(checkpoint.stem.removeprefix("model_")),
        }

    generator = torch.Generator(device=base.device)
    generator.manual_seed(random_seed)
    active = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
    episode_return = torch.zeros(base.num_envs, device=base.device)
    episode_steps = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
    max_phase = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
    saturated_values = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
    action_values = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
    outcomes = ["timeout"] * base.num_envs

    for _ in range(STAGE9_STANDALONE_CONFIG.max_episode_steps):
        if not torch.any(active):
            break
        phase = obs["policy"][:, -5:].argmax(dim=-1)
        max_phase = torch.maximum(max_phase, phase)
        with torch.inference_mode():
            if policy is not None:
                actions = policy(obs)
            elif controller_id == "random":
                actions = 2.0 * torch.rand(
                    (base.num_envs, 7),
                    generator=generator,
                    device=base.device,
                ) - 1.0
            else:
                actions = torch.zeros((base.num_envs, 7), device=base.device)
        saturated_values += ((torch.abs(actions) >= 1.0) & active[:, None]).sum(dim=-1)
        action_values += active.to(dtype=torch.long) * actions.shape[1]
        active_before = active.clone()
        obs, rewards, dones, _ = wrapped_env.step(actions)
        episode_return += rewards * active_before
        episode_steps += active_before.to(dtype=torch.long)
        success = base.termination_manager.get_term("success")
        failure = base.termination_manager.get_term("failure")
        invalid = base.termination_manager.get_term("invalid_state")
        timed_out = base.termination_manager.get_term("time_out")
        newly_done = active_before & dones.to(dtype=torch.bool)
        for env_index in torch.nonzero(newly_done, as_tuple=False).flatten().tolist():
            if bool(success[env_index]):
                outcomes[env_index] = "success"
            elif bool(invalid[env_index]):
                outcomes[env_index] = "invalid_state"
            elif bool(failure[env_index]):
                outcomes[env_index] = "failure"
            elif bool(timed_out[env_index]):
                outcomes[env_index] = "timeout"
            else:
                outcomes[env_index] = "unknown_termination"
        active &= ~newly_done
        if policy_nn is not None:
            policy_nn.reset(dones)

    rows: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios):
        denominator = max(int(action_values[index].item()), 1)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "simulation_seed": scenario.simulation_seed,
                "outcome": outcomes[index],
                "steps": int(episode_steps[index].item()),
                "return": float(episode_return[index].item()),
                "max_phase": int(max_phase[index].item()),
                "raw_action_saturation_rate": (
                    float(saturated_values[index].item()) / denominator
                ),
            }
        )
    success_count = sum(row["outcome"] == "success" for row in rows)
    return {
        "controller_id": controller_id,
        "checkpoint": checkpoint_record,
        "success_count": success_count,
        "success_rate": success_count / len(rows),
        "failure_count": sum(row["outcome"] == "failure" for row in rows),
        "timeout_count": sum(row["outcome"] == "timeout" for row in rows),
        "invalid_state_count": sum(row["outcome"] == "invalid_state" for row in rows),
        "mean_return": sum(float(row["return"]) for row in rows) / len(rows),
        "mean_steps": sum(int(row["steps"]) for row in rows) / len(rows),
        "scenarios": rows,
    }


def main() -> None:
    controllers = tuple(args_cli.controllers or ("zero", "random", "checkpoint"))
    checkpoints = _resolve_checkpoints(args_cli.checkpoints or [])
    if "checkpoint" in controllers and not checkpoints:
        raise ValueError("at least one --checkpoint is required for checkpoint evaluation")
    if "checkpoint" not in controllers and checkpoints:
        raise ValueError("--checkpoint was supplied without --controller checkpoint")

    scenarios = load_scenarios(
        STAGE9_STANDALONE_CONFIG.evaluation_scenarios_path,
        expected_count=10,
    )
    output_dir = (
        STAGE9_STANDALONE_CONFIG.output_dir / "evaluations" / args_cli.output_name
    ).resolve()
    if not output_dir.is_relative_to(STAGE9_STANDALONE_CONFIG.output_dir.resolve()):
        raise ValueError("--output_name escapes the reviewed output directory")
    report_path = output_dir / "evaluation.json"
    if report_path.exists():
        raise FileExistsError(f"evaluation already exists: {report_path}")

    env_cfg = FrankaPickPlacePPOEnvCfg()
    env_cfg.scene.num_envs = len(scenarios)
    env_cfg.seed = args_cli.random_seed
    env_cfg.sim.device = args_cli.device
    env_cfg.log_dir = str(output_dir)
    env = gym.make(RL_TASK_ID, cfg=env_cfg)
    try:
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=1.0)
        results = []
        for controller_id, checkpoint in _controller_specs(controllers, checkpoints):
            print("STAGE9_PPO_EVALUATION_START", controller_id, flush=True)
            result = _evaluate_controller(
                wrapped_env,
                scenarios,
                controller_id=controller_id,
                checkpoint=checkpoint,
                random_seed=args_cli.random_seed,
            )
            results.append(result)
            print(
                "STAGE9_PPO_EVALUATION_CONTROLLER_OK",
                controller_id,
                f"success_rate={result['success_rate']:.3f}",
                f"mean_return={result['mean_return']:.3f}",
                flush=True,
            )
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "task_id": RL_TASK_ID,
            "device": args_cli.device,
            "random_seed": args_cli.random_seed,
            "scenario_manifest": {
                "path": str(STAGE9_STANDALONE_CONFIG.evaluation_scenarios_path),
                "sha256": _sha256(STAGE9_STANDALONE_CONFIG.evaluation_scenarios_path),
                "count": len(scenarios),
            },
            "results": results,
        }
        _write_json_atomic(report_path, report)
        print("STAGE9_PPO_EVALUATION_OK", f"report={report_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
