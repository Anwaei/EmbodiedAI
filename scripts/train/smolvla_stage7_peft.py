#!/usr/bin/env python3
"""Run Stage 7 Step 6A micro-overfit and bounded SmolVLA LoRA training."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import random
import shutil
import subprocess
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from embodied_ai.policies.smolvla.config import Stage7RunConfig
from embodied_ai.policies.smolvla.dataset import (
    open_dataset,
    validate_dataset_and_split,
)
from embodied_ai.policies.smolvla.processing import (
    PROCESSOR_MANIFEST_NAME,
    atomic_write_json,
    load_json_object,
    load_project_processors,
    sha256_file,
)
from embodied_ai.policies.smolvla.runtime import (
    LocalSmolVLAAssets,
    load_base_policy,
    runtime_identity,
)
from embodied_ai.policies.smolvla.split import Stage7EpisodeSplit
from embodied_ai.policies.smolvla.training import (
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
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=data / "checkpoints/stage7-step6a/smolvla-lora-r2-steps50-v1",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=data / "runs/stage7-step6a/smolvla-lora-r2-steps50-v1",
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
    rows = {int(row["episode_index"]): row for row in dataset.meta.episodes}
    mapping = dataset.absolute_to_relative_idx
    indices: list[int] = []
    for episode_index in episode_indices:
        row = rows[episode_index]
        absolute = int(row["dataset_from_index"]) + int(row["length"]) // 2
        relative = absolute if mapping is None else mapping[absolute]
        indices.append(relative)
    subset = torch.utils.data.Subset(dataset, indices)
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


def _policy_config_dict(config: Any) -> dict[str, object]:
    value = _jsonable(config)
    if not isinstance(value, dict):
        raise TypeError("SmolVLA config serialization did not return an object")
    return value


def _seed_everything(seed: int) -> None:
    """Reset every RNG used by this bounded single-process training entry point."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = _parser().parse_args()
    repository = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI")).resolve()
    data_root = Path(
        os.environ.get("EMBODIEDAI_DATA", "/root/autodl-tmp/EmbodiedAI")
    ).resolve()
    artifacts_root = (data_root / "artifacts").resolve()
    checkpoints_root = (data_root / "checkpoints").resolve()
    runs_root = (data_root / "runs").resolve()
    processor_dir = _under(args.processor_dir, artifacts_root, "--processor_dir")
    checkpoint_dir = _under(args.checkpoint_dir, checkpoints_root, "--checkpoint_dir")
    run_dir = _under(args.run_dir, runs_root, "--run_dir")
    for path in (checkpoint_dir, run_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Step 6A output: {path}")

    run_config = Stage7RunConfig.from_toml(args.config, repository_root=repository)
    _seed_everything(run_config.seed)
    split = Stage7EpisodeSplit.from_json(run_config.split_path)
    dataset_root, _ = validate_dataset_and_split(run_config, split)
    processors = load_project_processors(processor_dir)
    if processors.statistics_scope != "train_split":
        raise ValueError("Step 6A requires training-split statistics")
    processor_manifest = load_json_object(processor_dir / PROCESSOR_MANIFEST_NAME)
    processor_provenance = processor_manifest.get("provenance")
    if not isinstance(processor_provenance, dict):
        raise ValueError("processor provenance is missing")
    if processor_provenance.get("split_sha256") != sha256_file(run_config.split_path):
        raise ValueError("processor artifacts were not built from the reviewed split")

    base_report = runs_root / "stage7-step5/smolvla-base-offline-v1/report.json"
    if not base_report.is_file():
        raise FileNotFoundError("the reviewed Step 5 base report must exist before Step 6A")

    assets = LocalSmolVLAAssets.from_run_config(run_config)
    asset_identity = assets.verify()
    dataset_base, dataset_policy_config = load_base_policy(
        assets,
        device=run_config.device,
    )
    train_dataset = open_dataset(
        run_config,
        episodes=split.train_episode_indices,
        policy_config=dataset_policy_config,
        action_horizon=True,
    )
    validation_dataset = open_dataset(
        run_config,
        episodes=split.validation_episode_indices,
        policy_config=dataset_policy_config,
        action_horizon=True,
    )
    del dataset_policy_config, dataset_base
    gc.collect()
    torch.cuda.empty_cache()

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
    validation_batches = _selected_validation_batches(
        validation_dataset,
        split.validation_episode_indices,
    )
    micro_batch = next(iter(train_loader))

    # The disposable micro-overfit model checks optimizer mechanics without biasing
    # the adapter that will be published by the bounded training run.
    micro_seed = run_config.seed + 1
    _seed_everything(micro_seed)
    micro_base, _ = load_base_policy(assets, device=run_config.device)
    micro_policy, micro_summary = wrap_lora(
        micro_base,
        rank=run_config.peft_rank,
        alpha=run_config.peft_alpha,
        dropout=run_config.peft_dropout,
    )
    micro_optimizer = optimizer_for_policy(
        micro_policy,
        learning_rate=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    micro_initial_loss = validation_loss(micro_policy, processors, (micro_batch,))
    micro_result = train_bounded(
        micro_policy,
        processors,
        (micro_batch,),
        micro_optimizer,
        optimizer_steps=run_config.micro_overfit_steps,
        gradient_accumulation_steps=1,
        gradient_clip_norm=run_config.gradient_clip_norm,
        log_every_steps=1,
        log_prefix="STAGE7_MICRO_OVERFIT",
        fixed_flow_input_step=0,
    )
    micro_final_loss = validation_loss(micro_policy, processors, (micro_batch,))
    if micro_final_loss >= micro_initial_loss:
        raise RuntimeError(
            "micro-overfit check failed: fixed-input loss did not decrease "
            f"({micro_initial_loss:.6f} -> {micro_final_loss:.6f})"
        )
    del micro_optimizer, micro_policy, micro_base
    gc.collect()
    torch.cuda.empty_cache()

    published_adapter_seed = run_config.seed + 2
    _seed_everything(published_adapter_seed)
    base_policy, policy_config = load_base_policy(assets, device=run_config.device)
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
    training_result = train_bounded(
        policy,
        processors,
        train_loader,
        optimizer,
        optimizer_steps=run_config.max_optimizer_steps,
        gradient_accumulation_steps=run_config.gradient_accumulation_steps,
        gradient_clip_norm=run_config.gradient_clip_norm,
        log_every_steps=run_config.log_every_steps,
        log_prefix="STAGE7_STEP6A_TRAIN",
    )
    final_validation_loss = validation_loss(policy, processors, validation_batches)

    common_provenance = {
        "repository_revision": _git_revision(repository),
        "run_config": run_config.to_dict(),
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
        "step5_base_report": base_report.as_posix(),
        "step5_base_report_sha256": sha256_file(base_report),
        "vla_lock_sha256": sha256_file(repository / "env/vla/uv.lock"),
        "assets": asset_identity,
        "runtime": runtime_identity(),
    }
    training_report = {
        "schema_version": "embodied-ai.stage7-step6a-training/v1",
        "status": "training_passed_reload_evaluation_pending",
        "training_episode_indices": list(split.train_episode_indices),
        "validation_episode_indices": list(split.validation_episode_indices),
        "training_frames": len(train_dataset),
        "validation_frames": len(validation_dataset),
        "validation_anchor_count": len(validation_batches),
        "peft": peft_summary.to_dict(),
        "micro_overfit": {
            "seed": micro_seed,
            "peft": micro_summary.to_dict(),
            "steps": run_config.micro_overfit_steps,
            "fixed_flow_input_step": 0,
            "initial_loss": micro_initial_loss,
            "final_loss": micro_final_loss,
            "losses": list(micro_result.step_losses),
            "gradient_norms": list(micro_result.gradient_norms),
            "changed_trainable_tensors": micro_result.changed_trainable_tensors,
        },
        "bounded_training": {
            "seed": published_adapter_seed,
            "optimizer_steps": run_config.max_optimizer_steps,
            "gradient_accumulation_steps": run_config.gradient_accumulation_steps,
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "losses": list(training_result.step_losses),
            "gradient_norms": list(training_result.gradient_norms),
            "changed_trainable_tensors": training_result.changed_trainable_tensors,
            "peak_allocated_bytes": training_result.peak_allocated_bytes,
            "peak_reserved_bytes": training_result.peak_reserved_bytes,
        },
        "provenance": common_provenance,
    }

    checkpoint_partial = (
        checkpoint_dir.parent / f".{checkpoint_dir.name}.{uuid.uuid4().hex}.partial"
    )
    checkpoint_partial.mkdir(parents=True, exist_ok=False)
    try:
        policy.save_pretrained(checkpoint_partial)
        processors.save(
            checkpoint_partial / "processors",
            provenance={**common_provenance, "artifact_role": "step6a_checkpoint"},
        )
        atomic_write_json(
            checkpoint_partial / "project_policy_config.json",
            _policy_config_dict(policy_config),
        )
        atomic_write_json(checkpoint_partial / "training_report.json", training_report)
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
                **training_report,
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

    print(
        "STAGE7_STEP6A_TRAIN_OK",
        f"steps={run_config.max_optimizer_steps}",
        f"trainable={peft_summary.trainable_parameters}",
        f"initial_val_loss={initial_validation_loss:.6f}",
        f"final_val_loss={final_validation_loss:.6f}",
        f"checkpoint={checkpoint_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
