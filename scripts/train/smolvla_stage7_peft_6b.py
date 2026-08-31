#!/usr/bin/env python3
"""Train and publish the formal five-epoch Stage 7 Step 6B LoRA adapter."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import os
import random
import shutil
import subprocess
import time
import uuid
from enum import Enum
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from embodied_ai.policies.smolvla.config import Stage7Step6BConfig
from embodied_ai.policies.smolvla.dataset import open_dataset, validate_dataset_and_split
from embodied_ai.policies.smolvla.processing import (
    PROCESSOR_MANIFEST_NAME,
    atomic_write_json,
    load_json_object,
    load_project_processors,
    sha256_file,
)
from embodied_ai.policies.smolvla.profile import franka_pick_place_smolvla_profile
from embodied_ai.policies.smolvla.runtime import (
    LocalSmolVLAAssets,
    load_base_policy,
    runtime_identity,
)
from embodied_ai.policies.smolvla.split import (
    STAGE7_FORMAL_SPLIT_SCHEMA_VERSION,
    Stage7EpisodeSplit,
)
from embodied_ai.policies.smolvla.training import (
    TrainingResult,
    optimizer_for_policy,
    train_bounded,
    validation_loss,
    wrap_lora,
)


def _parser() -> argparse.ArgumentParser:
    repository = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI"))
    data = Path(os.environ.get("EMBODIEDAI_DATA", "/root/autodl-tmp/EmbodiedAI"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            repository
            / "configs/policy/smolvla_franka_pick_place_v2_100_step6b.toml"
        ),
    )
    parser.add_argument(
        "--processor_dir",
        type=Path,
        default=(
            data
            / "artifacts/stage7/processors"
            / "franka-pick-place-smolvla-v2-100-train80-v1"
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=(
            data
            / "checkpoints/stage7-step6b"
            / "smolvla-lora-r8-100ep-5epochs-v1"
        ),
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=(
            data / "runs/stage7-step6b/smolvla-lora-r8-100ep-5epochs-v1"
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


def _selected_validation_batches(
    dataset: Any,
    episode_indices: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    """Use one deterministic middle-frame anchor from every validation episode."""

    rows = {int(row["episode_index"]): row for row in dataset.meta.episodes}
    mapping = dataset.absolute_to_relative_idx
    relative_indices: list[int] = []
    for episode_index in episode_indices:
        row = rows[episode_index]
        absolute = int(row["dataset_from_index"]) + int(row["length"]) // 2
        relative_indices.append(absolute if mapping is None else mapping[absolute])
    subset = torch.utils.data.Subset(dataset, relative_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    return tuple(loader)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"SmolVLA config contains unsupported value {type(value).__name__}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _result_summary(result: TrainingResult) -> dict[str, object]:
    return {
        "optimizer_steps": len(result.step_losses),
        "first_loss": result.step_losses[0],
        "final_loss": result.step_losses[-1],
        "mean_loss": fmean(result.step_losses),
        "minimum_loss": min(result.step_losses),
        "maximum_loss": max(result.step_losses),
        "mean_gradient_norm": fmean(result.gradient_norms),
        "maximum_gradient_norm": max(result.gradient_norms),
        "changed_trainable_tensors": result.changed_trainable_tensors,
        "peak_allocated_bytes": result.peak_allocated_bytes,
        "peak_reserved_bytes": result.peak_reserved_bytes,
    }


def main() -> None:
    args = _parser().parse_args()
    repository = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI")).resolve()
    data_root = Path(
        os.environ.get("EMBODIEDAI_DATA", "/root/autodl-tmp/EmbodiedAI")
    ).resolve()
    processor_dir = _under(args.processor_dir, data_root / "artifacts", "--processor_dir")
    checkpoint_dir = _under(
        args.checkpoint_dir,
        data_root / "checkpoints",
        "--checkpoint_dir",
    )
    run_dir = _under(args.run_dir, data_root / "runs", "--run_dir")
    for path in (checkpoint_dir, run_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Step 6B output: {path}")

    formal_config = Stage7Step6BConfig.from_toml(
        args.config,
        repository_root=repository,
    )
    run_config = formal_config.run
    split = Stage7EpisodeSplit.from_json(run_config.split_path)
    if split.schema_version != STAGE7_FORMAL_SPLIT_SCHEMA_VERSION:
        raise ValueError("Step 6B requires the reviewed v2 80/10/10 split")
    profile = franka_pick_place_smolvla_profile(run_config.dataset_repo_id)
    dataset_root, _ = validate_dataset_and_split(run_config, split, profile=profile)
    processors = load_project_processors(processor_dir, profile=profile)
    if processors.statistics_scope != "train_split":
        raise ValueError("Step 6B requires training-split statistics")
    processor_manifest = load_json_object(processor_dir / PROCESSOR_MANIFEST_NAME)
    provenance = processor_manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("processor provenance is missing")
    if provenance.get("split_sha256") != sha256_file(run_config.split_path):
        raise ValueError("processor artifacts were not built from the reviewed split")

    base_report = (
        data_root
        / "runs/stage7-step5"
        / "smolvla-base-offline-100ep-validation-v1/report.json"
    )
    if not base_report.is_file():
        raise FileNotFoundError("expanded-dataset base report must exist before Step 6B")

    _seed_everything(run_config.seed)
    assets = LocalSmolVLAAssets.from_run_config(run_config)
    asset_identity = assets.verify()
    base_policy, policy_config = load_base_policy(
        assets,
        device=run_config.device,
        profile=profile,
    )
    train_dataset = open_dataset(
        run_config,
        episodes=split.train_episode_indices,
        policy_config=policy_config,
        action_horizon=True,
        profile=profile,
    )
    validation_dataset = open_dataset(
        run_config,
        episodes=split.validation_episode_indices,
        policy_config=policy_config,
        action_horizon=True,
        profile=profile,
    )
    validation_batches = _selected_validation_batches(
        validation_dataset,
        split.validation_episode_indices,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(run_config.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=run_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    policy, peft_summary = wrap_lora(
        base_policy,
        rank=run_config.peft_rank,
        alpha=run_config.peft_alpha,
        dropout=run_config.peft_dropout,
    )
    optimizer = optimizer_for_policy(
        policy,
        learning_rate=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    initial_validation_loss = validation_loss(policy, processors, validation_batches)
    checkpoint_partial = (
        checkpoint_dir.parent / f".{checkpoint_dir.name}.{uuid.uuid4().hex}.partial"
    )
    checkpoint_partial.mkdir(parents=True, exist_ok=False)
    epoch_reports: list[dict[str, object]] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    started = time.perf_counter()
    try:
        for epoch in range(1, formal_config.epochs + 1):
            epoch_started = time.perf_counter()
            result = train_bounded(
                policy,
                processors,
                train_loader,
                optimizer,
                optimizer_steps=len(train_loader),
                gradient_accumulation_steps=run_config.gradient_accumulation_steps,
                gradient_clip_norm=run_config.gradient_clip_norm,
                log_every_steps=run_config.log_every_steps,
                log_prefix=f"STAGE7_STEP6B_EPOCH_{epoch}",
            )
            if epoch % formal_config.validation_every_epochs == 0:
                current_validation_loss = validation_loss(
                    policy,
                    processors,
                    validation_batches,
                )
            else:
                current_validation_loss = float("nan")
            epoch_dir = checkpoint_partial / "epochs" / f"epoch-{epoch:03d}"
            if formal_config.keep_every_epoch_adapter:
                policy.save_pretrained(epoch_dir)
            if current_validation_loss < best_validation_loss:
                best_epoch = epoch
                best_validation_loss = current_validation_loss
            summary = {
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "validation_loss": current_validation_loss,
                "training": _result_summary(result),
            }
            epoch_reports.append(summary)
            print(
                "STAGE7_STEP6B_EPOCH_OK",
                f"epoch={epoch}/{formal_config.epochs}",
                f"train_loss={summary['training']['mean_loss']:.6f}",
                f"validation_loss={current_validation_loss:.6f}",
                f"elapsed_seconds={summary['elapsed_seconds']:.1f}",
                flush=True,
            )

        best_dir = checkpoint_partial / "epochs" / f"epoch-{best_epoch:03d}"
        if not best_dir.is_dir():
            raise RuntimeError("best epoch adapter was not serialized")
        # Publish the best validation epoch at the checkpoint root while retaining
        # all epoch adapters for auditability and later ablation.
        for source in best_dir.iterdir():
            if source.is_file():
                shutil.copy2(source, checkpoint_partial / source.name)

        common_provenance = {
            "repository_revision": _git_revision(repository),
            "formal_config": formal_config.to_dict(),
            "run_config_sha256": sha256_file(run_config.path),
            "split": split.to_dict(),
            "split_sha256": sha256_file(run_config.split_path),
            "dataset_root": dataset_root.as_posix(),
            "dataset_conversion_sha256": sha256_file(
                dataset_root / "meta/embodied_ai_conversion.json"
            ),
            "processor_source": processor_dir.as_posix(),
            "processor_manifest_sha256": sha256_file(
                processor_dir / PROCESSOR_MANIFEST_NAME
            ),
            "statistics_sha256": processors.statistics_sha256,
            "expanded_base_report": base_report.as_posix(),
            "expanded_base_report_sha256": sha256_file(base_report),
            "vla_lock_sha256": sha256_file(repository / "env/vla/uv.lock"),
            "assets": asset_identity,
            "runtime": runtime_identity(),
        }
        processors.save(
            checkpoint_partial / "processors",
            provenance={**common_provenance, "artifact_role": "step6b_checkpoint"},
        )
        policy_config_value = _jsonable(policy_config)
        if not isinstance(policy_config_value, dict):
            raise TypeError("SmolVLA policy config did not serialize to an object")
        atomic_write_json(
            checkpoint_partial / "project_policy_config.json",
            policy_config_value,
        )
        report = {
            "schema_version": "embodied-ai.stage7-step6b-training/v1",
            "status": "training_passed_reload_evaluation_pending",
            "epochs": formal_config.epochs,
            "optimizer_steps_per_epoch": len(train_loader),
            "optimizer_steps_total": len(train_loader) * formal_config.epochs,
            "training_episode_indices": list(split.train_episode_indices),
            "validation_episode_indices": list(split.validation_episode_indices),
            "test_episode_indices": list(split.test_episode_indices),
            "training_frames": len(train_dataset),
            "validation_frames": len(validation_dataset),
            "validation_anchor_count": len(validation_batches),
            "peft": peft_summary.to_dict(),
            "initial_validation_loss": initial_validation_loss,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "elapsed_seconds": time.perf_counter() - started,
            "epoch_reports": epoch_reports,
            "provenance": common_provenance,
        }
        atomic_write_json(checkpoint_partial / "training_report.json", report)
        os.replace(checkpoint_partial, checkpoint_dir)
    finally:
        if checkpoint_partial.exists():
            shutil.rmtree(checkpoint_partial)

    run_partial = run_dir.parent / f".{run_dir.name}.{uuid.uuid4().hex}.partial"
    run_partial.mkdir(parents=True, exist_ok=False)
    try:
        atomic_write_json(
            run_partial / "report.json",
            {
                **report,
                "status": "passed",
                "checkpoint_dir": checkpoint_dir.as_posix(),
                "checkpoint_adapter_sha256": sha256_file(
                    checkpoint_dir / "adapter_model.safetensors"
                ),
            },
        )
        os.replace(run_partial, run_dir)
    finally:
        if run_partial.exists():
            shutil.rmtree(run_partial)

    del optimizer, policy, base_policy
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "STAGE7_STEP6B_TRAIN_OK",
        f"epochs={formal_config.epochs}",
        f"best_epoch={best_epoch}",
        f"best_validation_loss={best_validation_loss:.6f}",
        f"checkpoint={checkpoint_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
