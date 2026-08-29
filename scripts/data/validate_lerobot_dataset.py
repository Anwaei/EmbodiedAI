#!/usr/bin/env python3
"""Validate one local LeRobotDataset against its immutable source collection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from embodied_ai.data.lerobot_validation import (
    validate_lerobot_dataset,
    write_validation_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument(
        "--report_path",
        type=Path,
        help="defaults to EMBODIEDAI_RUNS/stage7-validation/<dataset-name>.json",
    )
    return parser


def _resolve_under(path: Path, allowed_root: Path, argument: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = allowed_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(allowed_root):
        raise ValueError(f"{argument} must resolve under {allowed_root}")
    return candidate


def main() -> None:
    args = _parser().parse_args()
    datasets_root = Path(
        os.environ.get("EMBODIEDAI_DATASETS", "/root/autodl-tmp/EmbodiedAI/datasets")
    ).expanduser().resolve()
    runs_root = Path(
        os.environ.get("EMBODIEDAI_RUNS", "/root/autodl-tmp/EmbodiedAI/runs")
    ).expanduser().resolve()
    dataset_root = _resolve_under(args.dataset_root, datasets_root, "--dataset_root")
    source_root = _resolve_under(args.source_root, datasets_root, "--source_root")
    report_path = args.report_path or (
        runs_root / "stage7-validation" / f"{dataset_root.name}.json"
    )
    report_path = _resolve_under(report_path, runs_root, "--report_path")

    source_episodes = tuple(
        sorted(
            path
            for path in source_root.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        )
    )
    report = validate_lerobot_dataset(
        dataset_root,
        source_episodes,
        repo_id=args.repo_id,
    )
    written = write_validation_report(report, report_path)
    print(
        "STAGE7_VALIDATION_OK",
        f"dataset={report.dataset_root}",
        f"repo_id={report.repo_id}",
        f"episodes={report.episode_count}",
        f"frames={report.frame_count}",
        f"tasks={report.task_count}",
        f"decoded_image_samples={report.decoded_image_samples}",
        f"report={written}",
        flush=True,
    )


if __name__ == "__main__":
    main()
