"""Validated TOML configuration for Stage 7 SmolVLA runs."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SMOLVLA_RUN_CONFIG_SCHEMA_VERSION = "embodied-ai.smolvla-run-config/v1"


def _table(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a TOML table")
    return value


@dataclass(frozen=True, slots=True)
class Stage7RunConfig:
    """Small reviewed configuration shared by processor, inference, and PEFT jobs."""

    path: Path
    seed: int
    device: str
    dataset_root_name: str
    dataset_repo_id: str
    mapping_profile: str
    split_path: Path
    video_backend: str
    model_revision: str
    model_sha256: str
    vlm_revision: str
    statistics_scope: str
    chunk_size: int
    tokenizer_max_length: int
    anchor_positions: tuple[str, ...]
    noise_seed: int
    warmup_samples: int
    peft_rank: int
    peft_alpha: int
    peft_dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_accumulation_steps: int
    micro_overfit_steps: int
    max_optimizer_steps: int
    gradient_clip_norm: float
    log_every_steps: int
    schema_version: str = SMOLVLA_RUN_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SMOLVLA_RUN_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported SmolVLA run config schema version")
        if self.device not in ("cuda", "cuda:0"):
            raise ValueError("the accepted Stage 7 runtime must use logical CUDA device 0")
        if self.statistics_scope != "train_split":
            raise ValueError("Stage 7 Step 5/6A must use training-split statistics")
        if self.chunk_size != 50 or self.tokenizer_max_length != 48:
            raise ValueError("SmolVLA chunk/tokenizer settings differ from the reviewed baseline")
        if self.anchor_positions != ("first", "middle", "last"):
            raise ValueError("offline anchors must remain first/middle/last")
        positive_ints = (
            self.seed,
            self.noise_seed,
            self.warmup_samples,
            self.peft_rank,
            self.peft_alpha,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.micro_overfit_steps,
            self.max_optimizer_steps,
            self.log_every_steps,
        )
        if any(isinstance(value, bool) or value <= 0 for value in positive_ints):
            raise ValueError("integer run settings must be positive")
        if self.max_optimizer_steps > 200:
            raise ValueError("Step 6A may not exceed the reviewed 200-step ceiling")
        if not 0.0 <= self.peft_dropout < 1.0:
            raise ValueError("PEFT dropout must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("learning rate and gradient clipping must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight decay must be non-negative")

    @classmethod
    def from_toml(cls, path: Path, *, repository_root: Path) -> Stage7RunConfig:
        resolved = path.expanduser().resolve()
        with resolved.open("rb") as stream:
            value: Any = tomllib.load(stream)
        if not isinstance(value, dict):
            raise ValueError("run configuration must be a TOML object")
        dataset = _table(value.get("dataset"), "dataset")
        model = _table(value.get("model"), "model")
        processor = _table(value.get("processor"), "processor")
        offline = _table(value.get("offline_inference"), "offline_inference")
        peft = _table(value.get("peft"), "peft")
        if peft.get("method") != "lora":
            raise ValueError("Stage 7 Step 6A supports only LoRA")
        anchors = offline.get("anchor_positions")
        if not isinstance(anchors, list) or not all(isinstance(item, str) for item in anchors):
            raise ValueError("offline anchor_positions must be a string list")
        split_path = (repository_root / dataset.get("split_path", "")).resolve()
        if not split_path.is_relative_to(repository_root.resolve()):
            raise ValueError("split path must remain inside the Git repository")
        return cls(
            path=resolved,
            seed=value.get("seed"),
            device=value.get("device"),
            dataset_root_name=dataset.get("root_name"),
            dataset_repo_id=dataset.get("repo_id"),
            mapping_profile=dataset.get("mapping_profile"),
            split_path=split_path,
            video_backend=dataset.get("video_backend"),
            model_revision=model.get("revision"),
            model_sha256=model.get("sha256"),
            vlm_revision=model.get("vlm_revision"),
            statistics_scope=processor.get("statistics_scope"),
            chunk_size=processor.get("chunk_size"),
            tokenizer_max_length=processor.get("tokenizer_max_length"),
            anchor_positions=tuple(anchors),
            noise_seed=offline.get("noise_seed"),
            warmup_samples=offline.get("warmup_samples"),
            peft_rank=peft.get("rank"),
            peft_alpha=peft.get("alpha"),
            peft_dropout=peft.get("dropout"),
            learning_rate=peft.get("learning_rate"),
            weight_decay=peft.get("weight_decay"),
            batch_size=peft.get("batch_size"),
            gradient_accumulation_steps=peft.get("gradient_accumulation_steps"),
            micro_overfit_steps=peft.get("micro_overfit_steps"),
            max_optimizer_steps=peft.get("max_optimizer_steps"),
            gradient_clip_norm=peft.get("gradient_clip_norm"),
            log_every_steps=peft.get("log_every_steps"),
            schema_version=value.get("schema_version"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": self.path.as_posix(),
            "seed": self.seed,
            "device": self.device,
            "dataset": {
                "root_name": self.dataset_root_name,
                "repo_id": self.dataset_repo_id,
                "mapping_profile": self.mapping_profile,
                "split_path": self.split_path.as_posix(),
                "video_backend": self.video_backend,
            },
            "model": {
                "revision": self.model_revision,
                "sha256": self.model_sha256,
                "vlm_revision": self.vlm_revision,
            },
            "processor": {
                "statistics_scope": self.statistics_scope,
                "chunk_size": self.chunk_size,
                "tokenizer_max_length": self.tokenizer_max_length,
            },
            "offline_inference": {
                "anchor_positions": list(self.anchor_positions),
                "noise_seed": self.noise_seed,
                "warmup_samples": self.warmup_samples,
            },
            "peft": {
                "rank": self.peft_rank,
                "alpha": self.peft_alpha,
                "dropout": self.peft_dropout,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "micro_overfit_steps": self.micro_overfit_steps,
                "max_optimizer_steps": self.max_optimizer_steps,
                "gradient_clip_norm": self.gradient_clip_norm,
                "log_every_steps": self.log_every_steps,
            },
        }


__all__ = ["SMOLVLA_RUN_CONFIG_SCHEMA_VERSION", "Stage7RunConfig"]
