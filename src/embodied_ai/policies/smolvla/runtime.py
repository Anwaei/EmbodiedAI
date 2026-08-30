"""Pinned local SmolVLA asset verification and project-bound model loading."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import Stage7RunConfig
from .processing import build_project_config, sha256_file
from .profile import FRANKA_PICK_PLACE_SMOLVLA_PROFILE, SmolVLAProjectProfile


@dataclass(frozen=True, slots=True)
class LocalSmolVLAAssets:
    model_dir: Path
    vlm_dir: Path
    model_revision: str
    model_sha256: str
    vlm_revision: str

    @classmethod
    def from_run_config(cls, config: Stage7RunConfig) -> LocalSmolVLAAssets:
        models_root = Path(
            os.environ.get("EMBODIEDAI_MODELS", "/root/autodl-tmp/EmbodiedAI/models")
        ).expanduser()
        return cls(
            model_dir=models_root / "lerobot" / "smolvla_base" / config.model_revision,
            vlm_dir=(
                models_root
                / "huggingfacetb"
                / "SmolVLM2-500M-Video-Instruct"
                / config.vlm_revision
            ),
            model_revision=config.model_revision,
            model_sha256=config.model_sha256,
            vlm_revision=config.vlm_revision,
        )

    def verify(self) -> dict[str, object]:
        if self.model_dir.name != self.model_revision:
            raise ValueError("model path does not end in the reviewed immutable revision")
        if self.vlm_dir.name != self.vlm_revision:
            raise ValueError("VLM path does not end in the reviewed immutable revision")
        required_model_files = (
            self.model_dir / "config.json",
            self.model_dir / "model.safetensors",
        )
        for path in (*required_model_files, self.vlm_dir / "config.json"):
            if not path.is_file():
                raise FileNotFoundError(f"required local model asset is missing: {path}")
        observed_sha256 = sha256_file(self.model_dir / "model.safetensors")
        if observed_sha256 != self.model_sha256:
            raise ValueError("pinned SmolVLA model checksum does not match")
        return {
            "model_dir": self.model_dir.as_posix(),
            "model_revision": self.model_revision,
            "model_sha256": observed_sha256,
            "vlm_dir": self.vlm_dir.as_posix(),
            "vlm_revision": self.vlm_revision,
        }


def load_base_policy(
    assets: LocalSmolVLAAssets,
    *,
    device: str,
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE,
) -> tuple[Any, Any]:
    """Load full pinned weights with project 9D/one-camera/7D feature metadata."""

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = build_project_config(
        assets.model_dir,
        assets.vlm_dir,
        device=device,
        profile=profile,
    )
    policy = SmolVLAPolicy.from_pretrained(
        assets.model_dir,
        config=config,
        local_files_only=True,
    )
    policy.config.pretrained_path = str(assets.model_dir)
    policy.to(device)
    return policy, config


def load_adapter_policy(
    assets: LocalSmolVLAAssets,
    adapter_dir: Path,
    *,
    device: str,
    profile: SmolVLAProjectProfile = FRANKA_PICK_PLACE_SMOLVLA_PROFILE,
) -> tuple[Any, Any]:
    """Load a saved PEFT adapter over the same verified project-bound base."""

    from peft import PeftConfig, PeftModel

    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_weights_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_config_path.is_file() or not adapter_weights_path.is_file():
        raise FileNotFoundError(f"adapter files are incomplete: {adapter_dir}")
    with adapter_config_path.open(encoding="utf-8") as stream:
        serialized = json.load(stream)
    if serialized.get("base_model_name_or_path") != str(assets.model_dir):
        raise ValueError("adapter base model identity differs from the pinned local checkpoint")
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base, config = load_base_policy(assets, device=device, profile=profile)
    policy = PeftModel.from_pretrained(
        base,
        adapter_dir,
        config=peft_config,
        is_trainable=False,
    )
    policy.to(device)
    return policy, config


def fixed_noise(config: Any, *, seed: int, device: str) -> torch.Tensor:
    """Generate reproducible per-sample flow noise without consuming global RNG state."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(
        (1, config.chunk_size, config.max_action_dim),
        generator=generator,
        dtype=torch.float32,
    )
    return noise.to(device)


def runtime_identity() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 7 model execution requires a CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    return {
        "python_environment": os.environ.get("VIRTUAL_ENV"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_total_memory_bytes": int(properties.total_memory),
    }


__all__ = [
    "LocalSmolVLAAssets",
    "fixed_noise",
    "load_adapter_policy",
    "load_base_policy",
    "runtime_identity",
]
