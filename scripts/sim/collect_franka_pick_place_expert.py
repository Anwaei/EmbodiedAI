#!/usr/bin/env python3
"""Collect one instruction-bearing Franka pick-and-place expert episode."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

from embodied_ai.contracts.tasks.franka_pick_place import (
    DEFAULT_INSTRUCTION,
    DEFAULT_INSTRUCTION_ID,
    DEFAULT_INSTRUCTION_LANGUAGE,
    TASK_NAME,
)


repository_root = Path(__file__).resolve().parents[2]
artifacts_root = Path(
    os.environ.get("EMBODIEDAI_ARTIFACTS", "/root/autodl-tmp/EmbodiedAI/artifacts")
).expanduser().resolve()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--output_root",
    type=Path,
    default=Path(
        os.environ.get("EMBODIEDAI_DATASETS", "/root/autodl-tmp/EmbodiedAI/datasets")
    )
    / "stage6-expert",
)
parser.add_argument("--episode_id", default="episode-stage6-expert-000001")
parser.add_argument("--task", default=TASK_NAME)
parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
parser.add_argument("--instruction_id", default=DEFAULT_INSTRUCTION_ID)
parser.add_argument("--instruction_language", default=DEFAULT_INSTRUCTION_LANGUAGE)
parser.add_argument(
    "--video_path",
    type=Path,
    default=None,
    help="optional MP4 path under EMBODIEDAI_ARTIFACTS",
)
parser.add_argument(
    "--expert_config",
    type=Path,
    default=repository_root
    / "configs"
    / "sim"
    / "franka_pick_place"
    / "state_machine_expert.toml",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import embodied_ai.sim.tasks  # noqa: F401
from embodied_ai.contracts import EpisodeOutcome, EpisodeProvenance
from embodied_ai.contracts.tasks.franka_pick_place import (
    CONTROL_HZ,
    FRANKA_PICK_PLACE_ACTION_SCHEMA,
    FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
    GOAL_POSITION_ENV_M,
    ROBOT_NAME,
    SCENE_NAME,
)
from embodied_ai.data.episode_video import encode_episode_camera_video
from embodied_ai.sim.collection import collect_expert_episode
from embodied_ai.sim.experts import (
    ExpertTaskContext,
    FrankaPickPlaceStateMachineExpert,
    load_state_machine_config,
)
from embodied_ai.sim.recording import NpyEpisodeRecorder, validate_npy_episode
from embodied_ai.sim.tasks.franka_pick_place import TASK_ID
from embodied_ai.sim.tasks.franka_pick_place.env_cfg import (
    CONTRACT_OBSERVATION_TERM_MAP,
    FrankaPickPlaceEnvCfg,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return f"{revision}-dirty" if dirty else revision


def _configuration_revision(root: Path, expert_config: Path) -> str:
    digest = hashlib.sha256()
    paths = (
        root / "src/embodied_ai/contracts/tasks/franka_pick_place.py",
        root / "src/embodied_ai/sim/tasks/franka_pick_place/env_cfg.py",
        root / "src/embodied_ai/sim/tasks/franka_pick_place/evaluation.py",
        root / "src/embodied_ai/sim/experts/franka_pick_place_state_machine.py",
        expert_config.resolve(),
    )
    for path in paths:
        # Include names as delimiters so equal byte concatenations cannot alias revisions.
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _provenance(expert_config: Path) -> EpisodeProvenance:
    return EpisodeProvenance(
        simulator_name="Isaac Sim",
        simulator_version=version("isaacsim"),
        repository_revision=_repository_revision(repository_root),
        configuration_revision=_configuration_revision(repository_root, expert_config),
        environment_lock_sha256=_sha256(repository_root / "env/isaac/uv.lock"),
    )


def _artifact_video_path(requested_path: Path) -> Path:
    candidate = requested_path.expanduser()
    if not candidate.is_absolute():
        candidate = artifacts_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(artifacts_root):
        raise ValueError("--video_path must resolve under EMBODIEDAI_ARTIFACTS")
    return candidate


def main() -> None:
    if args_cli.steps < 1:
        raise ValueError("--steps must be positive")
    if args_cli.seed < 0:
        raise ValueError("--seed must be non-negative")

    config, config_revision = load_state_machine_config(args_cli.expert_config)
    task_context = ExpertTaskContext(
        task=args_cli.task,
        instruction=args_cli.instruction,
        instruction_id=args_cli.instruction_id,
        instruction_language=args_cli.instruction_language,
        goal_position_env_m=GOAL_POSITION_ENV_M,
    )
    env_cfg = FrankaPickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    # Keep the environment timeout aligned with the collector's explicit bounded horizon.
    env_cfg.episode_length_s = args_cli.steps / CONTROL_HZ
    env = gym.make(TASK_ID, cfg=env_cfg)
    print("STAGE6_EXPERT_PHASE environment-created", flush=True)
    try:
        expert = FrankaPickPlaceStateMachineExpert(
            config=config,
            configuration_revision=config_revision,
            task_context=task_context,
            num_envs=env.unwrapped.num_envs,
            device=env.unwrapped.device,
        )
        recorder = NpyEpisodeRecorder(
            output_root=args_cli.output_root,
            episode_id=args_cli.episode_id,
            task=task_context.task,
            robot=ROBOT_NAME,
            scene=SCENE_NAME,
            random_seed=args_cli.seed,
            observation_schema=FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
            action_schema=FRANKA_PICK_PLACE_ACTION_SCHEMA,
            provenance=_provenance(args_cli.expert_config),
            instruction=task_context.instruction,
            instruction_id=task_context.instruction_id,
            instruction_language=task_context.instruction_language,
            expert=expert.metadata,
        )

        # The reusable collector owns the only explicit reset in this environment lifecycle.
        result = collect_expert_episode(
            env=env,
            expert=expert,
            recorder=recorder,
            observation_schema=FRANKA_PICK_PLACE_OBSERVATION_SCHEMA,
            observation_term_map=CONTRACT_OBSERVATION_TERM_MAP,
            control_period_ns=round(1_000_000_000 / CONTROL_HZ),
            max_steps=args_cli.steps,
            seed=args_cli.seed,
        )
        validated = validate_npy_episode(result.recorded.directory)
        if validated != result.recorded.metadata:
            raise RuntimeError("validated expert metadata differs from the recorded manifest")

        print(
            "STAGE6_EXPERT_RESULT",
            f"episode={result.recorded.directory}",
            f"outcome={result.outcome.value}",
            f"steps={result.recorded.metadata.step_count}",
            f"phases={dict(result.phase_counts)}",
            flush=True,
        )
        if result.outcome is not EpisodeOutcome.SUCCESS:
            raise RuntimeError(
                f"expert rollout did not succeed: {result.termination_reason}"
            )
        if args_cli.video_path is not None:
            video = encode_episode_camera_video(
                result.recorded.directory,
                _artifact_video_path(args_cli.video_path),
            )
            print(
                "STAGE6_EXPERT_VIDEO",
                f"path={video.path}",
                f"frames={video.frame_count}",
                f"size={video.width}x{video.height}",
                f"fps={video.fps:g}",
                flush=True,
            )
        print("STAGE6_EXPERT_OK", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
