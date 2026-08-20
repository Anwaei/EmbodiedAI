#!/usr/bin/env python3
"""Create, finalize, reload, and validate a tiny local LeRobot dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    datasets_root = Path(
        os.environ.get("EMBODIEDAI_DATASETS", "/root/autodl-tmp/EmbodiedAI/datasets")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=datasets_root / "stage5" / "smoke_roundtrip_script",
    )
    parser.add_argument("--repo-id", default="embodiedai/stage5-smoke-script")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {args.root}")
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["s0", "s1", "s2", "s3"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["a0", "a1"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=10,
        root=args.root,
        robot_type="stage5_synthetic",
        features=features,
        use_videos=False,
    )
    for index in range(3):
        dataset.add_frame(
            {
                "observation.state": np.array(
                    [index, index + 1, index + 2, index + 3],
                    dtype=np.float32,
                ),
                "action": np.array([index / 10, -index / 10], dtype=np.float32),
                "task": "stage5 round trip",
            }
        )
    dataset.save_episode()
    # LeRobot 0.6.0 buffers metadata writers. Finalize before re-opening the
    # dataset in the same process so the Parquet footer and metadata are durable.
    dataset.finalize()

    reloaded = LeRobotDataset(args.repo_id, root=args.root)
    if len(reloaded) != 3 or reloaded.num_episodes != 1:
        raise RuntimeError(
            f"unexpected dataset shape: frames={len(reloaded)}, episodes={reloaded.num_episodes}"
        )
    sample = reloaded[2]
    np.testing.assert_allclose(sample["observation.state"].numpy(), [2, 3, 4, 5])
    np.testing.assert_allclose(sample["action"].numpy(), [0.2, -0.2])
    if sample["task"] != "stage5 round trip":
        raise RuntimeError(f"unexpected task: {sample['task']!r}")
    print("root", args.root)
    print("frames", len(reloaded), "episodes", reloaded.num_episodes)
    print("STAGE5_DATASET_ROUNDTRIP_OK")


if __name__ == "__main__":
    main()
