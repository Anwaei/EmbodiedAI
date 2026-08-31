#!/usr/bin/env python3
"""Validate project SmolVLA processors on the real Stage 7 dataset."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import torch

from embodied_ai.policies.smolvla.config import Stage7RunConfig
from embodied_ai.policies.smolvla.dataset import (
    open_dataset,
    policy_input_from_sample,
    validate_dataset_and_split,
)
from embodied_ai.policies.smolvla.processing import (
    atomic_write_json,
    build_project_config,
    build_project_processors,
    compute_policy_statistics,
    load_project_processors,
    sha256_file,
)
from embodied_ai.policies.smolvla.profile import franka_pick_place_smolvla_profile
from embodied_ai.policies.smolvla.runtime import LocalSmolVLAAssets, runtime_identity
from embodied_ai.policies.smolvla.split import Stage7EpisodeSplit


def _parser() -> argparse.ArgumentParser:
    repository = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI"))
    artifacts = Path(
        os.environ.get("EMBODIEDAI_ARTIFACTS", "/root/autodl-tmp/EmbodiedAI/artifacts")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository / "configs/policy/smolvla_franka_pick_place_v1.toml",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=(
            artifacts
            / "stage7"
            / "processors"
            / "franka-pick-place-smolvla-v1-train-split-v1"
        ),
    )
    return parser


def _under(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} must resolve under {root}")
    return resolved


def _git_revision(repository: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return f"{revision}-dirty" if dirty else revision


def main() -> None:
    args = _parser().parse_args()
    repository = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI")).resolve()
    artifacts_root = Path(
        os.environ.get("EMBODIEDAI_ARTIFACTS", "/root/autodl-tmp/EmbodiedAI/artifacts")
    ).resolve()
    output_dir = _under(args.output_dir, artifacts_root, "--output_dir")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Step 4 artifacts: {output_dir}")

    run_config = Stage7RunConfig.from_toml(args.config, repository_root=repository)
    split = Stage7EpisodeSplit.from_json(run_config.split_path)
    dataset_root, _ = validate_dataset_and_split(run_config, split)
    assets = LocalSmolVLAAssets.from_run_config(run_config)
    asset_identity = assets.verify()
    profile = franka_pick_place_smolvla_profile(run_config.dataset_repo_id)

    train_dataset = open_dataset(
        run_config,
        episodes=split.train_episode_indices,
        profile=profile,
    )
    statistics = compute_policy_statistics(train_dataset, profile)
    expected_train_frames = sum(
        int(row["length"])
        for row in train_dataset.meta.episodes
        if int(row["episode_index"]) in split.train_episode_indices
    )
    if len(train_dataset) != expected_train_frames:
        raise RuntimeError("selected training dataset frame count is inconsistent")

    policy_config = build_project_config(
        assets.model_dir,
        assets.vlm_dir,
        device=run_config.device,
        profile=profile,
    )
    processors = build_project_processors(
        policy_config,
        statistics,
        statistics_scope=run_config.statistics_scope,
        profile=profile,
    )

    full_dataset = open_dataset(run_config, profile=profile)
    raw_sample = full_dataset[0]
    inference = policy_input_from_sample(raw_sample, profile)
    processed_inference = processors.preprocess_inference(inference)

    horizon_dataset = open_dataset(
        run_config,
        episodes=split.train_episode_indices,
        policy_config=policy_config,
        action_horizon=True,
        profile=profile,
    )
    training_batch = next(iter(torch.utils.data.DataLoader(horizon_dataset, batch_size=1)))
    processed_training = processors.preprocess_training(training_batch)
    if processed_training[profile.action_key].shape != (1, profile.chunk_size, 7):
        raise RuntimeError("real training batch did not retain the 50x7 action horizon")

    roundtrip_raw = {
        **inference,
        profile.action_key: raw_sample[profile.action_key],
    }
    normalized = processors.preprocessor(roundtrip_raw)[profile.action_key]
    reconstructed, diagnostics = processors.postprocess(normalized)
    expected_action = raw_sample[profile.action_key].unsqueeze(0)
    maximum_roundtrip_error = float(
        (reconstructed - expected_action).abs().max().item()
    )
    if maximum_roundtrip_error > 1e-6:
        raise RuntimeError(
            "action processor round trip exceeded tolerance: "
            f"{maximum_roundtrip_error}"
        )

    provenance = {
        "repository_revision": _git_revision(repository),
        "run_config": run_config.to_dict(),
        "run_config_sha256": sha256_file(run_config.path),
        "split": split.to_dict(),
        "split_sha256": sha256_file(run_config.split_path),
        "dataset_root": dataset_root.as_posix(),
        "conversion_manifest_sha256": sha256_file(
            dataset_root / "meta/embodied_ai_conversion.json"
        ),
        "asset_identity": asset_identity,
        "runtime": runtime_identity(),
    }
    processors.save(output_dir, provenance=provenance)

    # Reload from disk to ensure processor JSON and safetensor state are sufficient.
    reloaded = load_project_processors(output_dir, profile=profile)
    reloaded_inference = reloaded.preprocess_inference(inference)
    for key in (
        profile.state_key,
        profile.image_key,
        "observation.language.tokens",
        "observation.language.attention_mask",
    ):
        if not torch.equal(processed_inference[key], reloaded_inference[key]):
            raise RuntimeError(f"reloaded processor changed {key!r}")
    reloaded_normalized = reloaded.preprocessor(roundtrip_raw)[profile.action_key]
    reloaded_action, _ = reloaded.postprocess(reloaded_normalized)
    if not torch.equal(reloaded_action, reconstructed):
        raise RuntimeError("reloaded postprocessor changed the reconstructed action")

    report = {
        "schema_version": "embodied-ai.stage7-step4-report/v1",
        "status": "passed",
        "profile": profile.to_dict(),
        "statistics_scope": run_config.statistics_scope,
        "statistics_sha256": processors.statistics_sha256,
        "training_episode_indices": list(split.train_episode_indices),
        "training_episode_count": len(split.train_episode_indices),
        "training_frame_count": len(train_dataset),
        "processed_inference_shapes": {
            key: list(value.shape)
            for key, value in processed_inference.items()
            if isinstance(value, torch.Tensor)
        },
        "processed_training_shapes": {
            key: list(value.shape)
            for key, value in processed_training.items()
            if isinstance(value, torch.Tensor)
        },
        "action_roundtrip_max_abs_error": maximum_roundtrip_error,
        "action_roundtrip_diagnostics": diagnostics,
        "serialized_reload_equal": True,
        "provenance": provenance,
    }
    report_path = atomic_write_json(output_dir / "step4_report.json", report)
    print(
        "STAGE7_STEP4_OK",
        f"train_episodes={len(split.train_episode_indices)}",
        f"train_frames={len(train_dataset)}",
        f"statistics_sha256={processors.statistics_sha256}",
        f"roundtrip_max_abs_error={maximum_roundtrip_error:.3e}",
        f"report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
