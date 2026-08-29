#!/usr/bin/env python3
"""Run a reviewed Franka expert collection plan with one Isaac process per episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from embodied_ai.contracts import (
    EpisodeMetadata,
    EpisodeOutcome,
    ExpertCollectionEpisodeSpec,
    ExpertCollectionPlan,
)
from embodied_ai.sim.recording import validate_npy_episode

SUMMARY_SCHEMA_VERSION = "embodied-ai.expert-collection-summary/v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs/sim/franka_pick_place/expert_collection_v1.toml"
)


def _parser() -> argparse.ArgumentParser:
    datasets_root = Path(
        os.environ.get("EMBODIEDAI_DATASETS", "/root/autodl-tmp/EmbodiedAI/datasets")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=datasets_root / "stage6-expert-batch-v1-20260827",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode_timeout_s", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_under(path: Path, root: Path, name: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"{name} must resolve under {root.resolve()}")
    return candidate


def _expected_parameters(
    spec: ExpertCollectionEpisodeSpec,
) -> tuple[dict[str, object], dict[str, object]]:
    parameters = spec.parameters
    return parameters.task_parameters(), parameters.reset_parameters()


def _validated_record(
    *,
    spec: ExpertCollectionEpisodeSpec,
    metadata: EpisodeMetadata,
    plan: ExpertCollectionPlan,
    manifest_path: Path,
    log_path: Path,
    exit_code: int,
    disposition: str,
) -> dict[str, object]:
    expected_task_parameters, expected_reset_parameters = _expected_parameters(spec)
    expected = {
        "episode_id": spec.episode_id,
        "task": plan.task,
        "random_seed": spec.seed,
        "instruction": spec.instruction,
        "instruction_id": spec.instruction_id,
        "instruction_language": spec.instruction_language,
        "task_parameters": expected_task_parameters,
        "reset_parameters": expected_reset_parameters,
    }
    actual = {
        "episode_id": metadata.episode_id,
        "task": metadata.task,
        "random_seed": metadata.random_seed,
        "instruction": metadata.instruction,
        "instruction_id": metadata.instruction_id,
        "instruction_language": metadata.instruction_language,
        "task_parameters": metadata.task_parameters,
        "reset_parameters": metadata.reset_parameters,
    }
    if actual != expected:
        raise ValueError(
            f"episode manifest does not match collection plan: actual={actual!r} expected={expected!r}"
        )
    if metadata.expert is None or metadata.expert.identifier != plan.expert_identifier:
        raise ValueError("episode expert identity does not match collection plan")
    if exit_code == 0 and metadata.outcome is not EpisodeOutcome.SUCCESS:
        raise ValueError("successful collector exit published a non-success episode")
    if exit_code != 0 and metadata.outcome is EpisodeOutcome.SUCCESS:
        raise ValueError("collector failed after publishing a success episode")
    return {
        "episode_id": metadata.episode_id,
        "disposition": disposition,
        "exit_code": exit_code,
        "outcome": metadata.outcome.value,
        "termination_reason": metadata.termination_reason,
        "step_count": metadata.step_count,
        "seed": metadata.random_seed,
        "instruction": metadata.instruction,
        "instruction_id": metadata.instruction_id,
        "instruction_language": metadata.instruction_language,
        "task_parameters": metadata.task_parameters,
        "reset_parameters": metadata.reset_parameters,
        "expert": metadata.expert.to_dict(),
        "manifest_sha256": _sha256(manifest_path),
        "log_path": str(log_path),
    }


def _write_summary(
    path: Path,
    *,
    plan: ExpertCollectionPlan,
    plan_path: Path,
    records: list[dict[str, object]],
) -> None:
    success_count = sum(record.get("outcome") == EpisodeOutcome.SUCCESS.value for record in records)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "collection_id": plan.collection_id,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "task": plan.task,
        "expert_identifier": plan.expert_identifier,
        "requested_episode_count": len(plan.episodes),
        "processed_episode_count": len(records),
        "success_count": success_count,
        "failure_count": len(records) - success_count,
        "complete": len(records) == len(plan.episodes),
        "episodes": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _child_command(
    spec: ExpertCollectionEpisodeSpec,
    plan: ExpertCollectionPlan,
    output_root: Path,
    expert_config: Path,
    device: str,
) -> list[str]:
    parameters = spec.parameters
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/sim/collect_franka_pick_place_expert.py"),
        "--headless",
        "--enable_cameras",
        "--device",
        device,
        "--steps",
        str(plan.max_steps),
        "--seed",
        str(spec.seed),
        "--output_root",
        str(output_root),
        "--episode_id",
        spec.episode_id,
        "--task",
        plan.task,
        "--instruction",
        spec.instruction,
        "--instruction_id",
        spec.instruction_id,
        "--instruction_language",
        spec.instruction_language,
        "--cube_reset_position_env_m",
        *(f"{value:.9g}" for value in parameters.cube_reset_position_env_m),
        "--goal_position_env_m",
        *(f"{value:.9g}" for value in parameters.goal_position_env_m),
        "--expert_config",
        str(expert_config),
    ]


def main() -> None:
    args = _parser().parse_args()
    if args.episode_timeout_s <= 0:
        raise ValueError("--episode_timeout_s must be positive")

    datasets_root = Path(
        os.environ.get("EMBODIEDAI_DATASETS", "/root/autodl-tmp/EmbodiedAI/datasets")
    ).expanduser().resolve()
    runs_root = Path(
        os.environ.get("EMBODIEDAI_RUNS", "/root/autodl-tmp/EmbodiedAI/runs")
    ).expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()
    if not plan_path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("--plan must be a reviewed file inside the repository")
    plan = ExpertCollectionPlan.from_toml(plan_path)
    expert_config = (REPOSITORY_ROOT / plan.expert_config).resolve()
    if not expert_config.is_relative_to(REPOSITORY_ROOT) or not expert_config.is_file():
        raise ValueError("plan expert_config does not resolve to a repository file")
    output_root = _resolve_under(args.output_root, datasets_root, "--output_root")
    summary_path = output_root / "collection_summary.json"
    log_root = runs_root / "stage6-expert-batch" / plan.collection_id

    if args.dry_run:
        print(
            "STAGE6_EXPERT_BATCH_PLAN_OK",
            f"collection={plan.collection_id}",
            f"episodes={len(plan.episodes)}",
            f"plan_sha256={_sha256(plan_path)}",
            flush=True,
        )
        return
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"collection output already exists; use --resume after review: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    child_environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    existing_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else source_root
    )

    for index, spec in enumerate(plan.episodes, start=1):
        episode_root = output_root / spec.episode_id
        manifest_path = episode_root / "manifest.json"
        log_path = log_root / f"{spec.episode_id}.log"
        print(
            "STAGE6_EXPERT_BATCH_EPISODE_START",
            f"index={index}/{len(plan.episodes)}",
            f"episode={spec.episode_id}",
            flush=True,
        )
        exit_code = 0
        disposition = "existing"
        error: str | None = None
        if episode_root.exists():
            if not args.resume:
                raise FileExistsError(f"episode already exists: {episode_root}")
        else:
            disposition = "collected"
            command = _child_command(spec, plan, output_root, expert_config, args.device)
            try:
                with log_path.open("w", encoding="utf-8", newline="\n") as log:
                    result = subprocess.run(
                        command,
                        cwd=REPOSITORY_ROOT,
                        env=child_environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.episode_timeout_s,
                        check=False,
                        text=True,
                    )
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
                error = f"collector exceeded {args.episode_timeout_s} seconds"

        try:
            metadata = validate_npy_episode(episode_root)
            record = _validated_record(
                spec=spec,
                metadata=metadata,
                plan=plan,
                manifest_path=manifest_path,
                log_path=log_path,
                exit_code=exit_code,
                disposition=disposition,
            )
            if error is not None:
                record["error"] = error
        except Exception as validation_error:
            record = {
                "episode_id": spec.episode_id,
                "disposition": disposition,
                "exit_code": exit_code,
                "outcome": "invalid",
                "error": error or str(validation_error),
                "log_path": str(log_path),
            }
        records.append(record)
        _write_summary(
            summary_path,
            plan=plan,
            plan_path=plan_path,
            records=records,
        )
        print(
            "STAGE6_EXPERT_BATCH_EPISODE_RESULT",
            f"episode={spec.episode_id}",
            f"outcome={record['outcome']}",
            f"steps={record.get('step_count', 0)}",
            f"exit_code={exit_code}",
            flush=True,
        )

    success_count = sum(record.get("outcome") == "success" for record in records)
    print(
        "STAGE6_EXPERT_BATCH_RESULT",
        f"collection={plan.collection_id}",
        f"success={success_count}",
        f"total={len(records)}",
        f"summary={summary_path}",
        flush=True,
    )
    if success_count != len(plan.episodes):
        raise RuntimeError("one or more collection episodes did not succeed")


if __name__ == "__main__":
    main()
