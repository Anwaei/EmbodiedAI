#!/usr/bin/env python3
"""Run deterministic project-data SmolVLA base or adapter offline inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np

from embodied_ai.policies.smolvla.config import Stage7RunConfig
from embodied_ai.policies.smolvla.dataset import (
    offline_anchors,
    open_dataset,
    validate_dataset_and_split,
)
from embodied_ai.policies.smolvla.offline import run_offline_inference
from embodied_ai.policies.smolvla.processing import (
    PROCESSOR_MANIFEST_NAME,
    atomic_write_json,
    load_json_object,
    load_project_processors,
    sha256_file,
)
from embodied_ai.policies.smolvla.runtime import (
    LocalSmolVLAAssets,
    load_adapter_policy,
    load_base_policy,
    runtime_identity,
)
from embodied_ai.policies.smolvla.split import Stage7EpisodeSplit


def _parser() -> argparse.ArgumentParser:
    repository = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI"))
    data = Path(os.environ.get("EMBODIEDAI_DATA", "/root/autodl-tmp/EmbodiedAI"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository / "configs/policy/smolvla_franka_pick_place_v1.toml",
    )
    parser.add_argument(
        "--processor_dir",
        type=Path,
        default=(
            data
            / "artifacts/stage7/processors"
            / "franka-pick-place-smolvla-v1-train-split-v1"
        ),
    )
    parser.add_argument("--adapter_dir", type=Path)
    parser.add_argument("--baseline_report", type=Path)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=data / "runs/stage7-step5/smolvla-base-offline-v1",
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


def _write_npz(path: Path, result: object, anchors: tuple[object, ...]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            predictions=result.predictions,
            targets=result.targets,
            action_chunks=result.chunks,
            episode_index=np.asarray([anchor.episode_index for anchor in anchors]),
            frame_index=np.asarray([anchor.frame_index for anchor in anchors]),
            dataset_index=np.asarray([anchor.dataset_index for anchor in anchors]),
            latency_seconds=np.asarray(result.latency_seconds, dtype=np.float64),
        )


def _comparison(baseline_path: Path, metrics: dict[str, object]) -> dict[str, object]:
    baseline = load_json_object(baseline_path)
    baseline_metrics = baseline.get("metrics")
    if not isinstance(baseline_metrics, dict):
        raise ValueError("baseline report lacks metrics")
    result: dict[str, object] = {"baseline_report": baseline_path.as_posix()}
    for key in ("mae", "rmse", "gripper_sign_agreement", "bound_violation_rate"):
        before = float(baseline_metrics[key])
        after = float(metrics[key])
        result[key] = {"base": before, "adapter": after, "delta": after - before}
    return result


def main() -> None:
    args = _parser().parse_args()
    repository = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI")).resolve()
    data_root = Path(
        os.environ.get("EMBODIEDAI_DATA", "/root/autodl-tmp/EmbodiedAI")
    ).resolve()
    artifacts_root = (data_root / "artifacts").resolve()
    runs_root = (data_root / "runs").resolve()
    checkpoints_root = (data_root / "checkpoints").resolve()
    output_dir = _under(args.output_dir, runs_root, "--output_dir")
    processor_dir = _under(args.processor_dir, data_root, "--processor_dir")
    if not processor_dir.is_relative_to(artifacts_root) and not processor_dir.is_relative_to(
        checkpoints_root
    ):
        raise ValueError("processor artifacts must be under artifacts or checkpoints")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite offline inference output: {output_dir}")

    run_config = Stage7RunConfig.from_toml(args.config, repository_root=repository)
    split = Stage7EpisodeSplit.from_json(run_config.split_path)
    dataset_root, _ = validate_dataset_and_split(run_config, split)
    processors = load_project_processors(processor_dir)
    if processors.statistics_scope != "train_split":
        raise ValueError("Step 5/6A offline inference requires training-split statistics")
    processor_manifest = load_json_object(processor_dir / PROCESSOR_MANIFEST_NAME)
    processor_provenance = processor_manifest.get("provenance")
    if not isinstance(processor_provenance, dict):
        raise ValueError("processor provenance is missing")
    if processor_provenance.get("split_sha256") != sha256_file(run_config.split_path):
        raise ValueError("processor artifacts were not built from the reviewed split")

    assets = LocalSmolVLAAssets.from_run_config(run_config)
    asset_identity = assets.verify()
    adapter_dir = None
    if args.adapter_dir is None:
        policy, policy_config = load_base_policy(assets, device=run_config.device)
        policy_kind = "base"
    else:
        adapter_dir = _under(args.adapter_dir, checkpoints_root, "--adapter_dir")
        policy, policy_config = load_adapter_policy(
            assets,
            adapter_dir,
            device=run_config.device,
        )
        policy_kind = "peft_adapter"

    dataset = open_dataset(run_config)
    anchors = offline_anchors(dataset, split.validation_episode_indices)
    result = run_offline_inference(
        policy,
        policy_config,
        processors,
        dataset,
        anchors,
        noise_seed=run_config.noise_seed,
        warmup_samples=run_config.warmup_samples,
    )
    repeated = run_offline_inference(
        policy,
        policy_config,
        processors,
        dataset,
        anchors[:1],
        noise_seed=run_config.noise_seed,
        warmup_samples=0,
    )
    repeat_max_abs_error = float(
        np.abs(repeated.predictions[0] - result.predictions[0]).max()
    )
    if repeat_max_abs_error > 1e-6:
        raise RuntimeError(f"offline inference is not repeatable: {repeat_max_abs_error}")

    validation_report = (
        runs_root / "stage7-validation/stage7-franka-pick-place-batch-v1.json"
    )
    if not validation_report.is_file():
        raise FileNotFoundError(f"Stage 7 validation report is missing: {validation_report}")
    report: dict[str, object] = {
        "schema_version": "embodied-ai.stage7-offline-inference/v1",
        "status": "passed",
        "policy_kind": policy_kind,
        "adapter_dir": None if adapter_dir is None else adapter_dir.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "validation_episode_indices": list(split.validation_episode_indices),
        "anchors": [record for record in result.records],
        "metrics": result.metrics,
        "repeatability_max_abs_error": repeat_max_abs_error,
        "peak_allocated_bytes": result.peak_allocated_bytes,
        "peak_reserved_bytes": result.peak_reserved_bytes,
        "provenance": {
            "repository_revision": _git_revision(repository),
            "run_config": run_config.to_dict(),
            "run_config_sha256": sha256_file(run_config.path),
            "split": split.to_dict(),
            "split_sha256": sha256_file(run_config.split_path),
            "dataset_conversion_sha256": sha256_file(
                dataset_root / "meta/embodied_ai_conversion.json"
            ),
            "dataset_validation_report": validation_report.as_posix(),
            "dataset_validation_report_sha256": sha256_file(validation_report),
            "processor_dir": processor_dir.as_posix(),
            "processor_manifest_sha256": sha256_file(
                processor_dir / PROCESSOR_MANIFEST_NAME
            ),
            "statistics_sha256": processors.statistics_sha256,
            "vla_lock_sha256": sha256_file(repository / "env/vla/uv.lock"),
            "assets": asset_identity,
            "runtime": runtime_identity(),
        },
    }
    if args.baseline_report is not None:
        baseline_report = _under(args.baseline_report, runs_root, "--baseline_report")
        report["base_adapter_comparison"] = _comparison(baseline_report, result.metrics)

    partial = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.partial"
    partial.mkdir(parents=True, exist_ok=False)
    try:
        _write_npz(partial / "predictions.npz", result, anchors)
        atomic_write_json(partial / "report.json", report)
        os.replace(partial, output_dir)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    print(
        "STAGE7_OFFLINE_INFERENCE_OK",
        f"policy={policy_kind}",
        f"samples={result.metrics['sample_count']}",
        f"mae={result.metrics['mae']:.6f}",
        f"rmse={result.metrics['rmse']:.6f}",
        f"repeat_max_abs_error={repeat_max_abs_error:.3e}",
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
