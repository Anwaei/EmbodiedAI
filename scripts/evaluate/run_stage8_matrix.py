#!/usr/bin/env python3
"""Run each Stage 8 scenario in a fresh Isaac process for one policy kind."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from embodied_ai.evaluation.config import Stage8RunConfig, load_scenarios


def main() -> None:
    repository = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI"))
    data_root = Path(os.environ.get("EMBODIEDAI_DATA", "/root/autodl-tmp/EmbodiedAI"))
    environments = Path(
        os.environ.get("EMBODIEDAI_ENVS", "/root/autodl-tmp/EmbodiedAI/envs")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository / "configs/evaluation/stage8_franka_pick_place_v1.toml",
    )
    parser.add_argument(
        "--policy_kind",
        choices=("scripted_expert", "base", "peft_adapter"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    config = Stage8RunConfig.from_toml(
        args.config,
        repository_root=repository,
        data_root=data_root,
    )
    scenarios = load_scenarios(
        config.scenarios_path,
        expected_count=config.expected_scenario_count,
    )
    evaluator = repository / "scripts/sim/evaluate_franka_pick_place_policy.py"
    python = environments / "isaac/bin/python"
    environment = {**os.environ, "PYTHONPATH": str(repository / "src")}
    for index, scenario in enumerate(scenarios, start=1):
        command = [
            str(python),
            str(evaluator),
            "--config",
            str(args.config),
            "--policy_kind",
            args.policy_kind,
            "--scenario_id",
            scenario.scenario_id,
            "--headless",
            "--device",
            args.device,
        ]
        if args.no_video:
            command.append("--no_video")
        print(
            "STAGE8_MATRIX_SCENARIO_START",
            f"policy={args.policy_kind}",
            f"scenario={scenario.scenario_id}",
            f"index={index}/{len(scenarios)}",
            flush=True,
        )
        # A fresh Kit process prevents simulator scene state from leaking between
        # geometrically distinct held-out scenarios.
        subprocess.run(command, cwd=repository, env=environment, check=True)
    print(
        "STAGE8_MATRIX_POLICY_OK",
        f"policy={args.policy_kind}",
        f"scenarios={len(scenarios)}",
        f"rollouts={len(scenarios) * len(config.noise_seeds)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
