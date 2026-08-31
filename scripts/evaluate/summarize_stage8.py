#!/usr/bin/env python3
"""Validate and summarize the complete Stage 8 three-policy rollout matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

from embodied_ai.evaluation.config import Stage8RunConfig, load_scenarios
from embodied_ai.evaluation.metrics import aggregate_rollouts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_policy(root: Path, policy_kind: str, expected: set[tuple[str, int]]) -> list[dict]:
    records: list[dict] = []
    observed: set[tuple[str, int]] = set()
    for manifest_path in sorted((root / policy_kind / "rollouts").glob("*/manifest.json")):
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        if record.get("schema_version") != "embodied-ai.stage8-rollout/v1":
            raise ValueError(f"unsupported rollout manifest: {manifest_path}")
        if record["policy"]["kind"] != policy_kind:
            raise ValueError(f"policy directory/manifest mismatch: {manifest_path}")
        key = (record["scenario"]["scenario_id"], int(record["noise_seed"]))
        if key in observed:
            raise ValueError(f"duplicate policy/scenario/seed rollout: {key}")
        observed.add(key)
        rollout_dir = manifest_path.parent
        for payload in record["payloads"]:
            path = rollout_dir / payload["path"]
            if not path.is_file() or path.stat().st_size != payload["byte_size"]:
                raise ValueError(f"missing or size-mismatched payload: {path}")
            if _sha256(path) != payload["sha256"]:
                raise ValueError(f"payload checksum mismatch: {path}")
        records.append(
            {
                "scenario_id": key[0],
                "noise_seed": key[1],
                "outcome": record["outcome"],
                "metrics": record["metrics"],
                "manifest": manifest_path.as_posix(),
            }
        )
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"incomplete rollout matrix for {policy_kind}: "
            f"missing={missing} extra={extra}"
        )
    return records


def _markdown(report: dict[str, object]) -> str:
    comparisons = report["comparisons"]
    base_expert = comparisons["base_vs_expert"]["success_rate_delta"]
    peft_expert = comparisons["peft_adapter_vs_expert"]["success_rate_delta"]
    peft_base = comparisons["peft_adapter_vs_base"]["success_rate_delta"]
    lines = [
        "# Stage 8 Closed-loop Policy Evaluation", "",
        f"Run: `{report['run_id']}`", "",
        "| Policy | Rollouts | Success | Success rate | Mean final error (m) | Mean steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in ("scripted_expert", "base", "peft_adapter"):
        item = report["policies"][kind]  # type: ignore[index]
        lines.append(
            f"| {kind} | {item['rollout_count']} | {item['success_count']} | "
            f"{item['success_rate']:.3f} | {item['mean_final_goal_error_m']:.4f} | "
            f"{item['mean_steps']:.1f} |"
        )
    lines.extend(
        [
            "", "## Acceptance", "",
            "The complete matrix and artifact-integrity checks passed. Policy acceptance "
            "below is derived from the recorded outcomes; any learned-policy error keeps that "
            "policy out of Stage 9 until separately reviewed.",
            "", "## Baseline comparison", "",
            "The scripted expert is the task sanity baseline. Base and PEFT are compared on "
            f"the identical {report['scenario_count']} scenarios and three policy-noise seeds.", "",
            f"- Base success-rate delta vs expert: {base_expert:+.3f}",
            f"- PEFT success-rate delta vs expert: {peft_expert:+.3f}",
            f"- PEFT success-rate delta vs base: {peft_base:+.3f}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    repository = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI"))
    data_root = Path(os.environ.get("EMBODIEDAI_DATA", "/root/autodl-tmp/EmbodiedAI"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=repository / "configs/evaluation/stage8_franka_pick_place_v1.toml",
    )
    args = parser.parse_args()
    config = Stage8RunConfig.from_toml(
        args.config, repository_root=repository, data_root=data_root
    )
    scenarios = load_scenarios(
        config.scenarios_path,
        expected_count=config.expected_scenario_count,
    )
    expected = {
        (scenario.scenario_id, seed)
        for scenario in scenarios
        for seed in config.noise_seeds
    }
    run_root = data_root / "runs/stage8" / config.run_id
    policy_records = {
        kind: _load_policy(run_root, kind, expected)
        for kind in ("scripted_expert", "base", "peft_adapter")
    }
    policies = {
        kind: aggregate_rollouts(records) for kind, records in policy_records.items()
    }
    policy_identities = {}
    for kind in ("base", "peft_adapter"):
        identity_path = run_root / kind / "policy_identity.json"
        if not identity_path.is_file():
            raise FileNotFoundError(f"policy identity artifact is missing: {identity_path}")
        policy_identities[kind] = json.loads(identity_path.read_text(encoding="utf-8"))

    def acceptance(kind: str) -> str:
        outcome_counts = policies[kind]["outcome_counts"]
        if int(outcome_counts.get("error", 0)) > 0:
            return "failed_safety_or_runtime"
        if kind == "scripted_expert" and float(policies[kind]["success_rate"]) < 1.0:
            return "failed_task_sanity"
        return "passed_evaluation"

    def compare(left: str, right: str) -> dict[str, float]:
        return {
            "success_rate_delta": float(policies[left]["success_rate"])
            - float(policies[right]["success_rate"]),
            "mean_final_goal_error_m_delta": float(
                policies[left]["mean_final_goal_error_m"]
            ) - float(policies[right]["mean_final_goal_error_m"]),
        }

    report: dict[str, object] = {
        "schema_version": "embodied-ai.stage8-comparison/v1",
        "status": "completed",
        "run_id": config.run_id,
        "scenario_count": len(scenarios),
        "expected_rollouts_per_policy": len(expected),
        "scheduler": {"prediction_horizon": 50, "execute_horizon": 5},
        "configuration": {
            "evaluation_config": args.config.resolve().as_posix(),
            "evaluation_config_sha256": _sha256(args.config),
            "scenarios": config.scenarios_path.as_posix(),
            "scenarios_sha256": _sha256(config.scenarios_path),
        },
        "policy_identities": policy_identities,
        "policies": policies,
        "policy_acceptance": {
            kind: acceptance(kind)
            for kind in ("scripted_expert", "base", "peft_adapter")
        },
        "comparisons": {
            "base_vs_expert": compare("base", "scripted_expert"),
            "peft_adapter_vs_expert": compare("peft_adapter", "scripted_expert"),
            "peft_adapter_vs_base": compare("peft_adapter", "base"),
        },
        "rollouts": policy_records,
        "reproduction": {
            "policy_server": (
                "PYTHONPATH=src $EMBODIEDAI_ENVS/vla/bin/python "
                "scripts/policy/serve_smolvla.py --policy_kind <base|peft_adapter>"
            ),
            "robot_client": (
                "PYTHONPATH=src $EMBODIEDAI_ENVS/isaac/bin/python "
                "scripts/sim/evaluate_franka_pick_place_policy.py --policy_kind "
                "<scripted_expert|base|peft_adapter> --scenario_id <scenario> "
                "--headless --device cuda:0"
            ),
            "summary": (
                "PYTHONPATH=src $EMBODIEDAI_ENVS/vla/bin/python "
                "scripts/evaluate/summarize_stage8.py"
            ),
        },
    }
    report_dir = run_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("comparison.json", json.dumps(report, indent=2, sort_keys=True) + "\n"),
        ("comparison.md", _markdown(report)),
    ):
        path = report_dir / name
        temporary = report_dir / f".{name}.{uuid.uuid4().hex}.partial"
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    print(
        "STAGE8_SUMMARY_OK",
        f"rollouts={len(expected) * 3}",
        f"report={report_dir / 'comparison.json'}", flush=True,
    )


if __name__ == "__main__":
    main()
