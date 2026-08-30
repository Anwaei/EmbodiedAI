"""Bounded PEFT training mechanics for Stage 7 Step 6A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .processing import ProjectProcessors


@dataclass(frozen=True, slots=True)
class PeftSummary:
    trainable_parameters: int
    total_parameters: int
    trainable_tensors: int
    target_modules: str

    def to_dict(self) -> dict[str, object]:
        return {
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_fraction": self.trainable_parameters / self.total_parameters,
            "trainable_tensors": self.trainable_tensors,
            "target_modules": self.target_modules,
        }


@dataclass(frozen=True, slots=True)
class TrainingResult:
    step_losses: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    changed_trainable_tensors: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def wrap_lora(
    base_policy: Any,
    *,
    rank: int,
    alpha: int,
    dropout: float,
) -> tuple[Any, PeftSummary]:
    """Apply only SmolVLA's reviewed default LoRA targets."""

    base_policy.config.load_vlm_weights = True
    policy = base_policy.wrap_with_peft(
        peft_cli_overrides={
            "method_type": "lora",
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
        }
    )
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in policy.parameters())
    if not 0 < trainable_count < total_count:
        raise RuntimeError("PEFT did not isolate a strict trainable parameter subset")
    peft_config = next(iter(policy.peft_config.values()))
    return policy, PeftSummary(
        trainable_parameters=trainable_count,
        total_parameters=total_count,
        trainable_tensors=len(trainable),
        target_modules=str(peft_config.target_modules),
    )


def optimizer_for_policy(
    policy: Any,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def _next_batch(iterator: Any, dataloader: Any) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def _fixed_training_inputs(
    config: Any,
    batch_size: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260830 + step)
    noise = torch.randn(
        (batch_size, config.chunk_size, config.max_action_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(config.device)
    # Sweep the flow interval deterministically while avoiding its singular endpoints.
    fraction = (step % 10 + 0.5) / 10.0
    sample_time = torch.full(
        (batch_size,),
        fraction,
        dtype=torch.float32,
        device=config.device,
    )
    return noise, sample_time


def validation_loss(
    policy: Any,
    processors: ProjectProcessors,
    batches: tuple[dict[str, Any], ...],
    *,
    seed_offset: int = 0,
) -> float:
    was_training = policy.training
    policy.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for index, raw in enumerate(batches):
            batch = processors.preprocess_training(raw)
            batch_size = int(batch[processors.profile.action_key].shape[0])
            noise, sample_time = _fixed_training_inputs(
                policy.config,
                batch_size,
                seed_offset + index,
            )
            loss, _ = policy(batch, noise=noise, time=sample_time)
            if not torch.isfinite(loss):
                raise RuntimeError("validation loss is non-finite")
            losses.append(float(loss.detach().item()))
    policy.train(was_training)
    return sum(losses) / len(losses)


def train_bounded(
    policy: Any,
    processors: ProjectProcessors,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    *,
    optimizer_steps: int,
    gradient_accumulation_steps: int,
    gradient_clip_norm: float,
    log_every_steps: int,
    log_prefix: str,
    fixed_flow_input_step: int | None = None,
) -> TrainingResult:
    policy.train()
    trainable = {
        name: parameter
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
    iterator = iter(dataloader)
    losses: list[float] = []
    gradient_norms: list[float] = []
    global_micro_step = 0
    torch.cuda.reset_peak_memory_stats()
    for optimizer_step in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            raw, iterator = _next_batch(iterator, dataloader)
            batch = processors.preprocess_training(raw)
            batch_size = int(batch[processors.profile.action_key].shape[0])
            noise, sample_time = _fixed_training_inputs(
                policy.config,
                batch_size,
                (
                    global_micro_step
                    if fixed_flow_input_step is None
                    else fixed_flow_input_step
                ),
            )
            loss, _ = policy(batch, noise=noise, time=sample_time)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {optimizer_step + 1}")
            accumulated_loss += float(loss.detach().item())
            (loss / gradient_accumulation_steps).backward()
            global_micro_step += 1
        gradients = [
            parameter.grad for parameter in trainable.values() if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"missing or non-finite PEFT gradients at step {optimizer_step + 1}")
        norm = torch.nn.utils.clip_grad_norm_(
            list(trainable.values()),
            gradient_clip_norm,
        )
        if not torch.isfinite(norm):
            raise RuntimeError(f"non-finite gradient norm at step {optimizer_step + 1}")
        optimizer.step()
        loss_value = accumulated_loss / gradient_accumulation_steps
        losses.append(loss_value)
        gradient_norms.append(float(norm.detach().item()))
        if (optimizer_step + 1) % log_every_steps == 0 or optimizer_step == 0:
            print(
                log_prefix,
                f"step={optimizer_step + 1}/{optimizer_steps}",
                f"loss={loss_value:.6f}",
                f"grad_norm={gradient_norms[-1]:.6f}",
                flush=True,
            )
    torch.cuda.synchronize()
    changed = sum(
        not torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in trainable.items()
    )
    if changed == 0:
        raise RuntimeError("bounded training did not update any trainable tensor")
    return TrainingResult(
        step_losses=tuple(losses),
        gradient_norms=tuple(gradient_norms),
        changed_trainable_tensors=changed,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    )


__all__ = [
    "PeftSummary",
    "TrainingResult",
    "optimizer_for_policy",
    "train_bounded",
    "validation_loss",
    "wrap_lora",
]
