"""SmolVLA online inference engine and loopback HTTP service for Stage 8."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from embodied_ai.contracts.policy_rpc import (
    POLICY_RPC_HEALTH_PATH,
    POLICY_RPC_INFERENCE_PATH,
    POLICY_RPC_RESET_PATH,
    PolicyHealthResponse,
    PolicyIdentity,
    PolicyInferenceRequest,
    PolicyInferenceResponse,
    PolicyResetRequest,
    PolicyResetResponse,
    decode_json,
    encode_json,
    error_response,
)

from .config import Stage7RunConfig
from .processing import PROCESSOR_MANIFEST_NAME, load_project_processors, sha256_file
from .runtime import (
    LocalSmolVLAAssets,
    fixed_noise,
    load_adapter_policy,
    load_base_policy,
    runtime_identity,
)


class SmolVLAOnlineEngine:
    """Own one CUDA policy and preserve episode reset state across HTTP requests."""

    def __init__(
        self,
        *,
        run_config: Stage7RunConfig,
        processor_dir: Path,
        policy_kind: str,
        adapter_dir: Path | None,
    ) -> None:
        assets = LocalSmolVLAAssets.from_run_config(run_config)
        assets.verify()
        self.processors = load_project_processors(processor_dir)
        if policy_kind == "base":
            self.policy, self.policy_config = load_base_policy(
                assets, device=run_config.device
            )
            identifier = "lerobot/smolvla_base"
            revision = run_config.model_revision
            model_sha256 = run_config.model_sha256
        elif policy_kind == "peft_adapter" and adapter_dir is not None:
            self.policy, self.policy_config = load_adapter_policy(
                assets, adapter_dir, device=run_config.device
            )
            identifier = "embodiedai/franka-pick-place-smolvla-lora"
            revision = adapter_dir.name
            model_sha256 = sha256_file(adapter_dir / "adapter_model.safetensors")
        else:
            raise ValueError("policy_kind must be base or peft_adapter with an adapter")
        self.policy.eval()
        self.identity = PolicyIdentity(
            kind=policy_kind,
            identifier=identifier,
            revision=revision,
            model_sha256=model_sha256,
            processor_sha256=sha256_file(processor_dir / PROCESSOR_MANIFEST_NAME),
            profile=self.processors.profile.name,
        )
        self.runtime = runtime_identity()
        self._episode_id: str | None = None
        self._instruction: str | None = None
        self._lock = threading.Lock()

    def health(self) -> PolicyHealthResponse:
        return PolicyHealthResponse("ready", self.identity, self.runtime)

    def reset(self, request: PolicyResetRequest) -> PolicyResetResponse:
        with self._lock:
            reset = getattr(self.policy, "reset", None)
            if not callable(reset):
                raise TypeError("loaded policy does not expose reset()")
            reset()
            self._episode_id = request.episode_id
            self._instruction = request.instruction
        return PolicyResetResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            identity_sha256=self.identity.identity_sha256,
        )

    def infer(self, request: PolicyInferenceRequest) -> PolicyInferenceResponse:
        with self._lock:
            if request.episode_id != self._episode_id or request.instruction != self._instruction:
                raise ValueError("inference request does not match the active reset context")
            image = np.frombuffer(request.observation.rgb_bytes(), dtype=np.uint8).reshape(
                3, 224, 224
            )
            sample = {
                self.processors.profile.state_key: torch.tensor(
                    request.observation.joint_position, dtype=torch.float32
                ),
                self.processors.profile.image_key: (
                    torch.from_numpy(image.copy()).float().div_(255.0)
                ),
                "task": request.instruction,
            }
            batch = self.processors.preprocess_inference(sample)
            # Derive each receding-horizon sample deterministically from episode seed and step.
            noise = fixed_noise(
                self.policy_config,
                seed=request.noise_seed + request.step_index,
                device=self.policy_config.device,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                normalized = self.policy.predict_action_chunk(batch, noise=noise)
                action_chunk, diagnostics = self.processors.postprocess(normalized)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            chunk = tuple(
                tuple(float(value) for value in action)
                for action in action_chunk[0].numpy()
            )
        return PolicyInferenceResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            step_index=request.step_index,
            identity_sha256=self.identity.identity_sha256,
            action_chunk=chunk,
            inference_latency_ms=latency_ms,
            diagnostics=diagnostics,
        )


def serve(
    engine: SmolVLAOnlineEngine,
    *,
    host: str,
    port: int,
    max_payload_bytes: int,
) -> None:
    """Serve one engine; the server is loopback-only by construction."""

    if host not in ("127.0.0.1", "::1"):
        raise ValueError("Stage 8 policy server may bind only to loopback")

    class Handler(BaseHTTPRequestHandler):
        server_version = "EmbodiedAIStage8/1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write(self, status: int, value: dict[str, object]) -> None:
            body = encode_json(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != POLICY_RPC_HEALTH_PATH:
                self._write(404, error_response("not_found", "unknown endpoint"))
                return
            self._write(200, engine.health().to_dict())

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > max_payload_bytes:
                    raise ValueError("request body length is outside configured limits")
                source = decode_json(self.rfile.read(length), max_bytes=max_payload_bytes)
                if self.path == POLICY_RPC_RESET_PATH:
                    response = engine.reset(PolicyResetRequest.from_dict(source)).to_dict()
                elif self.path == POLICY_RPC_INFERENCE_PATH:
                    response = engine.infer(PolicyInferenceRequest.from_dict(source)).to_dict()
                else:
                    self._write(404, error_response("not_found", "unknown endpoint"))
                    return
                self._write(200, response)
            except (ValueError, TypeError, RuntimeError, KeyError, json.JSONDecodeError) as error:
                self._write(400, error_response(type(error).__name__, str(error)))

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


__all__ = ["SmolVLAOnlineEngine", "serve"]
