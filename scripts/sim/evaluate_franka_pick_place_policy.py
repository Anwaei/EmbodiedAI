#!/usr/bin/env python3
"""Run the reviewed Stage 8 closed-loop Franka policy evaluation matrix."""

# Isaac Lab must start AppLauncher before importing simulator-dependent modules.
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import traceback
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

repository_root = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI"))
data_root = Path(os.environ.get("EMBODIEDAI_DATA", "/root/autodl-tmp/EmbodiedAI"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--config",
    type=Path,
    default=repository_root / "configs/evaluation/stage8_franka_pick_place_v1.toml",
)
parser.add_argument(
    "--policy_kind", choices=("scripted_expert", "base", "peft_adapter"), required=True
)
parser.add_argument(
    "--scenario_id",
    help="run one reviewed scenario; process-per-scenario avoids Kit scene reuse",
)
parser.add_argument("--no_video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import embodied_ai.sim.tasks  # noqa: F401
from embodied_ai.contracts.evaluation import (
    PolicyDescriptor,
    PolicyKind,
    RolloutOutcome,
)
from embodied_ai.contracts.policy_rpc import (
    PolicyInferenceRequest,
    PolicyResetRequest,
    canonical_json_sha256,
)
from embodied_ai.contracts.tasks.franka_pick_place import (
    CONTROL_HZ,
    TASK_NAME,
    FrankaPickPlaceEpisodeParameters,
)
from embodied_ai.evaluation.config import Stage8RunConfig, load_scenarios
from embodied_ai.evaluation.http_transport import PolicyHttpClient
from embodied_ai.evaluation.recording import EvaluationRolloutRecorder
from embodied_ai.evaluation.scheduler import RecedingHorizonScheduler
from embodied_ai.sim.experts import (
    ExpertTaskContext,
    FrankaPickPlaceStateMachineExpert,
    load_state_machine_config,
)
from embodied_ai.sim.evaluation.robot_client import (
    current_task_state,
    extract_live_observation,
    step_environment,
)
from embodied_ai.sim.tasks.franka_pick_place import TASK_ID
from embodied_ai.sim.tasks.franka_pick_place.env_cfg import (
    FrankaPickPlaceEnvCfg,
    apply_episode_parameters,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return f"{revision}-dirty" if dirty else revision


def _learned_descriptor(client: PolicyHttpClient, kind: str) -> PolicyDescriptor:
    health = client.health()
    if health.identity.kind != kind:
        raise RuntimeError("connected policy kind differs from requested policy")
    return PolicyDescriptor(
        kind=PolicyKind(kind),
        identifier=health.identity.identifier,
        revision=health.identity.revision,
        identity_sha256=health.identity.identity_sha256,
    )


def _expert_descriptor(config_path: Path) -> tuple[PolicyDescriptor, object, str]:
    config, revision = load_state_machine_config(config_path)
    descriptor = PolicyDescriptor(
        kind=PolicyKind.SCRIPTED_EXPERT,
        identifier=config.identifier,
        revision=config.revision,
        identity_sha256=canonical_json_sha256(
            {"kind": "scripted_expert", "configuration_sha256": revision}
        ),
    )
    return descriptor, config, revision


def main() -> None:
    config = Stage8RunConfig.from_toml(
        args_cli.config, repository_root=repository_root, data_root=data_root
    )
    scenarios = load_scenarios(
        config.scenarios_path,
        expected_count=config.expected_scenario_count,
    )
    if args_cli.scenario_id is not None:
        scenarios = tuple(
            scenario for scenario in scenarios if scenario.scenario_id == args_cli.scenario_id
        )
        if not scenarios:
            raise ValueError(f"unknown --scenario_id: {args_cli.scenario_id}")
    output_root = data_root / "runs/stage8" / config.run_id / args_cli.policy_kind / "rollouts"
    output_root.mkdir(parents=True, exist_ok=True)
    client = None
    expert_config = None
    expert_revision = None
    if args_cli.policy_kind == "scripted_expert":
        descriptor, expert_config, expert_revision = _expert_descriptor(config.expert_config_path)
    else:
        client = PolicyHttpClient(
            config.host, config.port, timeout_s=config.timeout_s,
            max_payload_bytes=config.max_payload_bytes,
        )
        descriptor = _learned_descriptor(client, args_cli.policy_kind)

    provenance = {
        "repository_revision": _git_revision(),
        "evaluation_config_sha256": _sha256(args_cli.config),
        "scenarios_sha256": _sha256(config.scenarios_path),
        "isaac_lock_sha256": _sha256(repository_root / "env/isaac/uv.lock"),
        "isaac_sim_version": version("isaacsim"),
        "isaac_lab_version": version("isaaclab"),
        "device": args_cli.device,
    }
    completed = 0
    for scenario in scenarios:
        parameters = FrankaPickPlaceEpisodeParameters(
            scenario.cube_position_env_m, scenario.goal_position_env_m
        )
        env_cfg = FrankaPickPlaceEnvCfg()
        apply_episode_parameters(env_cfg, parameters)
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args_cli.device
        env_cfg.episode_length_s = config.max_steps / CONTROL_HZ
        # Manual terminal checks preserve the post-step state instead of receiving Gym's
        # automatic reset observation on the success frame.
        env_cfg.terminations.success = None
        env_cfg.terminations.failure = None
        env_cfg.terminations.time_out = None
        env = gym.make(TASK_ID, cfg=env_cfg)
        try:
            for noise_seed in config.noise_seeds:
                rollout_id = f"{args_cli.policy_kind}-{scenario.scenario_id}-noise-{noise_seed}"
                rollout_dir = output_root / rollout_id
                if rollout_dir.exists():
                    print("STAGE8_ROLLOUT_EXISTS", rollout_dir, flush=True)
                    completed += 1
                    continue
                observations, _ = env.reset(seed=scenario.simulation_seed)
                recorder = EvaluationRolloutRecorder(rollout_dir)
                scheduler = RecedingHorizonScheduler(
                    hard_action_limit=config.hard_action_limit
                )
                expert = None
                if client is not None:
                    reset_id = f"reset-{rollout_id}"
                    reset_response = client.reset(
                        PolicyResetRequest(
                            reset_id, rollout_id, scenario.instruction, noise_seed
                        )
                    )
                    if reset_response.identity_sha256 != descriptor.identity_sha256:
                        raise RuntimeError("policy identity changed after health check")
                else:
                    context = ExpertTaskContext(
                        task=TASK_NAME,
                        instruction=scenario.instruction,
                        instruction_id=scenario.scenario_id,
                        instruction_language="en",
                        goal_position_env_m=scenario.goal_position_env_m,
                    )
                    expert = FrankaPickPlaceStateMachineExpert(
                        config=expert_config,
                        configuration_revision=expert_revision,
                        task_context=context,
                        num_envs=1,
                        device=env.unwrapped.device,
                    )
                    expert.reset()

                outcome = RolloutOutcome.TRUNCATED
                reason = "evaluation-step-limit"
                for step_index in range(config.max_steps):
                    safety_abort = False
                    wire, contract = extract_live_observation(observations)
                    if expert is not None:
                        expert_step = expert.act(env.unwrapped, observations)
                        raw_action = expert_step.actions[0].detach().cpu().numpy()
                        executed_action = raw_action.copy()
                        if bool(expert_step.failed[0]):
                            outcome = RolloutOutcome.FAILURE
                            reason = expert_step.failure_reasons[0] or "expert-failed"
                            break
                    else:
                        if scheduler.needs_prediction:
                            request_id = f"infer-{rollout_id}-{step_index:04d}"
                            started = time.perf_counter()
                            response = client.infer(
                                PolicyInferenceRequest(
                                    request_id=request_id,
                                    episode_id=rollout_id,
                                    step_index=step_index,
                                    instruction=scenario.instruction,
                                    noise_seed=noise_seed,
                                    observation=wire,
                                )
                            )
                            round_trip_ms = (time.perf_counter() - started) * 1000.0
                            if response.identity_sha256 != descriptor.identity_sha256:
                                raise RuntimeError("policy identity changed during rollout")
                            inference_record = {
                                "request_id": request_id,
                                "step_index": step_index,
                                "noise_seed": noise_seed + step_index,
                                "inference_latency_ms": response.inference_latency_ms,
                                "round_trip_latency_ms": round_trip_ms,
                                "diagnostics": dict(response.diagnostics),
                            }
                            try:
                                scheduler.accept(response.action_chunk)
                            except ValueError as error:
                                # Never execute a chunk that violates the reviewed hard bound.
                                # A safe zero action creates one auditable terminal frame.
                                safety_abort = True
                                outcome = RolloutOutcome.ERROR
                                reason = f"unsafe-policy-output: {error}"
                                inference_record["safety_error"] = str(error)
                                raw_action = np.asarray(
                                    response.action_chunk[0], dtype=np.float32
                                )
                                executed_action = np.zeros(7, dtype=np.float32)
                            recorder.add_inference_record(inference_record)
                        if not safety_abort:
                            scheduled = scheduler.pop()
                            raw_action = np.asarray(scheduled.raw, dtype=np.float32)
                            executed_action = np.asarray(
                                scheduled.executed, dtype=np.float32
                            )

                    observations = step_environment(env, executed_action)
                    state = current_task_state(
                        env.unwrapped, goal_position_env_m=scenario.goal_position_env_m
                    )
                    recorder.append(
                        joint_position=contract["robot.joint_position"],
                        camera_rgb=contract["camera.front.rgb"],
                        raw_action=raw_action,
                        executed_action=executed_action,
                        cube_position=state["cube_position_env_m"],
                        goal_error_m=state["goal_error_m"],
                        cube_speed_m_s=state["cube_speed_m_s"],
                        gripper_open=state["gripper_open"],
                        timestamp_ns=round((step_index + 1) * 1_000_000_000 / CONTROL_HZ),
                    )
                    if safety_abort:
                        break
                    if state["success"]:
                        outcome = RolloutOutcome.SUCCESS
                        reason = "task-success"
                        break
                    if state["failure"]:
                        outcome = RolloutOutcome.FAILURE
                        reason = "task-workspace-failure"
                        break
                    if expert is not None and bool(expert_step.done[0]):
                        outcome = RolloutOutcome.FAILURE
                        reason = "expert-done-without-success"
                        break

                manifest = recorder.finalize(
                    rollout_id=rollout_id,
                    run_id=config.run_id,
                    scenario=scenario,
                    policy=descriptor,
                    noise_seed=noise_seed,
                    outcome=outcome,
                    termination_reason=reason,
                    provenance=provenance,
                    record_video=config.record_video and not args_cli.no_video,
                )
                completed += 1
                print(
                    "STAGE8_ROLLOUT_OK",
                    f"rollout={rollout_id}", f"outcome={outcome.value}",
                    f"manifest={manifest}", flush=True,
                )
        finally:
            env.close()
    print(
        "STAGE8_POLICY_EVALUATION_OK",
        f"policy={args_cli.policy_kind}", f"rollouts={completed}",
        f"output={output_root}", flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac's shutdown may terminate logging before Python emits an unhandled traceback.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
