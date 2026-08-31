#!/usr/bin/env python3
"""Train or resume the Stage 9 state-based Franka PPO baseline."""

# Isaac Lab must start AppLauncher before simulator-dependent imports.
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

repository_root = Path(
    os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI")
).resolve()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run_name", required=True)
parser.add_argument("--num_envs", type=int)
parser.add_argument("--iterations", type=int)
parser.add_argument("--save_interval", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument(
    "--geometry_mode",
    choices=("fixed", "distribution"),
    default="distribution",
)
parser.add_argument("--resume_checkpoint", type=Path)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from embodied_ai.rl.config import reviewed_stage9_config_path
import embodied_ai.sim.tasks  # noqa: F401
from embodied_ai.sim.tasks.franka_pick_place import RL_TASK_ID
from embodied_ai.sim.tasks.franka_pick_place.agents.rsl_rl_ppo_cfg import (
    FrankaPickPlacePPORunnerCfg,
)
from embodied_ai.sim.tasks.franka_pick_place.rl_env_cfg import (
    STAGE9_STANDALONE_CONFIG,
    FrankaPickPlacePPOEnvCfg,
)

MANIFEST_SCHEMA_VERSION = "embodied-ai.stage9-ppo-training-run/v1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def _git_identity() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def _checkpoint_records(run_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in run_dir.glob("model_*.pt"):
        try:
            iteration = int(path.stem.removeprefix("model_"))
        except ValueError:
            continue
        records.append(
            {
                "iteration": iteration,
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return sorted(records, key=lambda item: int(item["iteration"]))


def _resolve_checkpoint(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    runs_root = Path(
        os.environ.get("EMBODIEDAI_RUNS", "/root/autodl-tmp/EmbodiedAI/runs")
    ).resolve()
    if not resolved.is_relative_to(runs_root) or not resolved.is_file():
        raise ValueError("--resume_checkpoint must be an existing file under EMBODIEDAI_RUNS")
    return resolved


def _apply_fixed_geometry(env_cfg: FrankaPickPlacePPOEnvCfg) -> None:
    params = env_cfg.events.reset_scene.params
    # A point range is accepted by the runtime sampler and is deliberately kept
    # out of the reviewed distribution contract, whose ranges must be non-empty.
    params["cube_x_m"] = (0.50, 0.50)
    params["cube_y_m"] = (0.00, 0.00)
    params["goal_x_m"] = (0.65, 0.65)
    params["goal_y_m"] = (-0.20, -0.20)


def main() -> None:
    num_envs = args_cli.num_envs or STAGE9_STANDALONE_CONFIG.num_envs
    iterations = args_cli.iterations or STAGE9_STANDALONE_CONFIG.ppo.max_iterations
    save_interval = args_cli.save_interval or STAGE9_STANDALONE_CONFIG.ppo.save_interval
    seed = args_cli.seed if args_cli.seed is not None else STAGE9_STANDALONE_CONFIG.identity.seed
    if min(num_envs, iterations, save_interval) <= 0:
        raise ValueError("environment count, iterations, and save interval must be positive")
    unsafe_run_part = any(
        part in {"", ".", ".."} for part in Path(args_cli.run_name).parts
    )
    if not args_cli.run_name or unsafe_run_part:
        raise ValueError("--run_name must be a safe non-empty relative path")

    run_dir = (STAGE9_STANDALONE_CONFIG.output_dir / args_cli.run_name).resolve()
    if not run_dir.is_relative_to(STAGE9_STANDALONE_CONFIG.output_dir.resolve()):
        raise ValueError("--run_name escapes the reviewed Stage 9 output directory")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"training output already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    resume_checkpoint = _resolve_checkpoint(args_cli.resume_checkpoint)
    reviewed_config_path = reviewed_stage9_config_path(repository_root)
    loaded_iteration: int | None = None
    started_at = _utc_now()
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "run_name": args_cli.run_name,
        "run_dir": str(run_dir),
        "started_at": started_at,
        "task_id": RL_TASK_ID,
        "geometry_mode": args_cli.geometry_mode,
        "num_envs": num_envs,
        "iterations_requested": iterations,
        "save_interval": save_interval,
        "seed": seed,
        "device": args_cli.device,
        "parent_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "git": _git_identity(),
        "versions": {
            "isaac_sim": version("isaacsim"),
            "isaac_lab": version("isaaclab"),
            "isaaclab_rl": version("isaaclab-rl"),
            "rsl_rl": version("rsl-rl-lib"),
            "torch": version("torch"),
        },
        "reviewed_config": {
            "path": str(reviewed_config_path),
            "sha256": _sha256(reviewed_config_path),
        },
    }
    _write_json_atomic(manifest_path, manifest)

    env = None
    runner = None
    try:
        env_cfg = FrankaPickPlacePPOEnvCfg()
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = seed
        env_cfg.sim.device = args_cli.device
        env_cfg.log_dir = str(run_dir)
        if args_cli.geometry_mode == "fixed":
            _apply_fixed_geometry(env_cfg)

        agent_cfg = FrankaPickPlacePPORunnerCfg()
        agent_cfg.seed = seed
        agent_cfg.device = args_cli.device
        agent_cfg.max_iterations = iterations
        agent_cfg.save_interval = save_interval
        agent_cfg.run_name = args_cli.run_name

        env = gym.make(RL_TASK_ID, cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped_env,
            agent_cfg.to_dict(),
            log_dir=str(run_dir),
            device=agent_cfg.device,
        )
        runner.add_git_repo_to_log(__file__)
        if resume_checkpoint is not None:
            runner.load(str(resume_checkpoint))
            loaded_iteration = runner.current_learning_iteration
            manifest["loaded_iteration"] = loaded_iteration
            _write_json_atomic(manifest_path, manifest)

        dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()

        checkpoints = _checkpoint_records(run_dir)
        if not checkpoints:
            raise RuntimeError("RSL-RL completed without publishing a checkpoint")
        final_iteration = int(checkpoints[-1]["iteration"])
        if loaded_iteration is not None and final_iteration <= loaded_iteration:
            raise RuntimeError("resume job did not advance beyond the loaded checkpoint")
        manifest.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "final_iteration": final_iteration,
                "final_checkpoint": checkpoints[-1],
                "checkpoints": checkpoints,
            }
        )
        _write_json_atomic(manifest_path, manifest)
        print(
            "STAGE9_PPO_TRAINING_OK",
            f"run={args_cli.run_name}",
            f"iteration={final_iteration}",
            f"checkpoint={checkpoints[-1]['path']}",
            flush=True,
        )
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "completed_at": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "checkpoints": _checkpoint_records(run_dir),
            }
        )
        _write_json_atomic(manifest_path, manifest)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
