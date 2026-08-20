#!/usr/bin/env python3
"""Run the bounded, offline SmolVLA inference and LoRA Stage 5 checks."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

MODEL_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
MODEL_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def parse_args() -> argparse.Namespace:
    models_root = Path(os.environ.get("EMBODIEDAI_MODELS", "/root/autodl-tmp/EmbodiedAI/models"))
    checkpoints_root = Path(
        os.environ.get("EMBODIEDAI_CHECKPOINTS", "/root/autodl-tmp/EmbodiedAI/checkpoints")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inference", "lora", "both"), default="both")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=models_root / "lerobot" / "smolvla_base" / MODEL_REVISION,
    )
    parser.add_argument(
        "--vlm-dir",
        type=Path,
        default=(
            models_root
            / "huggingfacetb"
            / "SmolVLM2-500M-Video-Instruct"
            / VLM_REVISION
        ),
    )
    parser.add_argument(
        "--adapter-output",
        type=Path,
        default=checkpoints_root / "stage5" / "smolvla_lora_smoke_script",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_local_policy(
    args: argparse.Namespace,
) -> tuple[SmolVLAPolicy, SmolVLAConfig, object, object]:
    if args.model_dir.name != MODEL_REVISION or args.vlm_dir.name != VLM_REVISION:
        raise ValueError("model and VLM paths must end in the reviewed immutable revisions")
    weight_path = args.model_dir / "model.safetensors"
    if sha256(weight_path) != MODEL_SHA256:
        raise ValueError(f"unexpected checkpoint checksum: {weight_path}")

    config = PreTrainedConfig.from_pretrained(args.model_dir, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"expected SmolVLAConfig, got {type(config).__name__}")

    # Build the exact architecture from the pinned local VLM configuration. The
    # full SmolVLA checkpoint below supplies every model weight, so the separate
    # 2 GB foundation-model weights are neither downloaded nor loaded.
    config.load_vlm_weights = False
    config.vlm_model_name = str(args.vlm_dir)
    config.device = args.device
    policy = SmolVLAPolicy.from_pretrained(
        args.model_dir,
        config=config,
        local_files_only=True,
    )
    policy.config.pretrained_path = str(args.model_dir)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.model_dir),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(args.vlm_dir)},
            "device_processor": {"device": args.device},
        },
    )
    return policy, config, preprocessor, postprocessor


def run_inference(
    policy: SmolVLAPolicy,
    config: SmolVLAConfig,
    preprocessor: object,
    postprocessor: object,
) -> None:
    raw = {
        "observation.state": torch.zeros(6, dtype=torch.float32),
        "observation.images.camera1": torch.zeros(3, 256, 256, dtype=torch.float32),
        "observation.images.camera2": torch.full((3, 256, 256), 0.25, dtype=torch.float32),
        "observation.images.camera3": torch.full((3, 256, 256), 0.5, dtype=torch.float32),
        "task": "pick up the cube",
    }
    batch = preprocessor(raw)
    noise = torch.zeros(
        (1, config.chunk_size, config.max_action_dim),
        device=config.device,
    )
    policy.reset()
    policy.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        action = postprocessor(policy.select_action(batch, noise=noise))
    torch.cuda.synchronize()
    if tuple(action.shape) != (1, 6) or not torch.isfinite(action).all():
        raise RuntimeError(
            f"invalid action: shape={tuple(action.shape)}, "
            f"finite={torch.isfinite(action).all()}"
        )
    print("inference_seconds", round(time.monotonic() - started, 3))
    print("action", action.detach().cpu().tolist())
    print("inference_peak_allocated_mib", round(torch.cuda.max_memory_allocated() / 1024**2, 2))
    print("STAGE5_SMOLVLA_INFERENCE_OK")


def run_lora_step(
    policy: SmolVLAPolicy,
    config: SmolVLAConfig,
    preprocessor: object,
    output_dir: Path,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty adapter output: {output_dir}")
    raw = {
        "observation.state": torch.linspace(-0.25, 0.25, 6, dtype=torch.float32),
        "observation.images.camera1": torch.full((3, 256, 256), 0.25, dtype=torch.float32),
        "action": torch.linspace(-0.1, 0.1, 50 * 6, dtype=torch.float32).reshape(50, 6),
        "task": "pick up the cube",
    }
    batch = preprocessor(raw)
    policy.config.load_vlm_weights = True
    peft_policy = policy.wrap_with_peft(
        peft_cli_overrides={
            "method_type": "lora",
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
        }
    )
    peft_policy.train()
    trainable = [
        (name, parameter)
        for name, parameter in peft_policy.named_parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    total_count = sum(parameter.numel() for parameter in peft_policy.parameters())
    if not 0 < trainable_count < total_count:
        raise RuntimeError("PEFT did not isolate a strict trainable parameter subset")

    before = {name: parameter.detach().clone() for name, parameter in trainable}
    optimizer = torch.optim.AdamW((parameter for _, parameter in trainable), lr=1e-4)
    noise = torch.zeros((1, config.chunk_size, config.max_action_dim), device=config.device)
    sample_time = torch.full((1,), 0.5, device=config.device)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    loss, loss_dict = peft_policy(batch, noise=noise, time=sample_time)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss}")
    loss.backward()
    gradients = [parameter.grad for _, parameter in trainable if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("LoRA gradients are missing or non-finite")
    optimizer.step()
    torch.cuda.synchronize()
    changed = [
        name
        for name, parameter in trainable
        if not torch.equal(before[name], parameter.detach())
    ]
    if not changed:
        raise RuntimeError("optimizer step did not update an adapter tensor")
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_policy.save_pretrained(output_dir)
    print("loss", float(loss.detach()), loss_dict)
    print("trainable_parameters", trainable_count, "of", total_count)
    print("changed_trainable_tensors", len(changed), "of", len(trainable))
    print("training_step_seconds", round(time.monotonic() - started, 3))
    print("training_peak_allocated_mib", round(torch.cuda.max_memory_allocated() / 1024**2, 2))
    print("adapter_output", output_dir)
    print("STAGE5_SMOLVLA_LORA_TRAIN_OK")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5 smoke test requires a CUDA GPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy, config, preprocessor, postprocessor = load_local_policy(args)
    print("model_revision", MODEL_REVISION)
    print("vlm_revision", VLM_REVISION)
    print("device", next(policy.parameters()).device)
    if args.mode in {"inference", "both"}:
        run_inference(policy, config, preprocessor, postprocessor)
    if args.mode in {"lora", "both"}:
        run_lora_step(policy, config, preprocessor, args.adapter_output)


if __name__ == "__main__":
    main()
