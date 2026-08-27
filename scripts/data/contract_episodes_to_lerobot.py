#!/usr/bin/env python3
"""Convert validated Stage 6 contract episodes into a local LeRobotDataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from embodied_ai.data.lerobot_converter import convert_contract_episodes_to_lerobot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_directories", nargs="+", type=Path)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument(
        "--storage",
        choices=("images", "videos"),
        default="videos",
        help="LeRobot visual storage; images is the bounded low-resource smoke mode",
    )
    return parser


def _resolve_under_datasets(path: Path, datasets_root: Path, argument: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = datasets_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(datasets_root):
        raise ValueError(f"{argument} must resolve under EMBODIEDAI_DATASETS")
    return candidate


def main() -> None:
    args = _parser().parse_args()
    datasets_root = Path(
        os.environ.get("EMBODIEDAI_DATASETS", "/root/autodl-tmp/EmbodiedAI/datasets")
    ).expanduser().resolve()
    output_root = _resolve_under_datasets(args.output_root, datasets_root, "--output_root")
    episode_directories = tuple(
        _resolve_under_datasets(path, datasets_root, "episode directory")
        for path in args.episode_directories
    )

    converted = convert_contract_episodes_to_lerobot(
        episode_directories,
        output_root,
        repo_id=args.repo_id,
        use_videos=args.storage == "videos",
    )
    print(
        "STAGE7_CONVERSION_OK",
        f"root={converted.root}",
        f"repo_id={converted.repo_id}",
        f"episodes={converted.episode_count}",
        f"frames={converted.frame_count}",
        f"fps={converted.fps}",
        f"storage={'videos' if converted.use_videos else 'images'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
