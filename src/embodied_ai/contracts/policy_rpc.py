"""Dependency-light Stage 8 Robot Client / Policy Server wire contracts."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any

from .action import ACTION_SCHEMA_VERSION
from .observation import OBSERVATION_SCHEMA_VERSION

POLICY_RPC_SCHEMA_VERSION = "embodied-ai.policy-rpc/v1"
POLICY_RPC_HEALTH_PATH = "/health"
POLICY_RPC_RESET_PATH = "/v1/reset"
POLICY_RPC_INFERENCE_PATH = "/v1/action-chunk"
POLICY_STATE_DIMENSION = 9
POLICY_ACTION_DIMENSION = 7
POLICY_ACTION_HORIZON = 50
POLICY_RGB_SHAPE = (3, 224, 224)
POLICY_RGB_BYTE_COUNT = 3 * 224 * 224


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def encode_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def decode_json(payload: bytes, *, max_bytes: int) -> dict[str, Any]:
    if not payload or len(payload) > max_bytes:
        raise ValueError("RPC payload is empty or exceeds the configured byte limit")
    try:
        value: Any = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RPC payload is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("RPC payload must contain a JSON object")
    return value


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _float_tuple(value: object, length: int, label: str) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, (Iterable, Sized))
    ):
        raise ValueError(f"{label} must be a sequence")
    if len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))


def _schema(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != POLICY_RPC_SCHEMA_VERSION:
        raise ValueError("unsupported policy RPC schema version")


@dataclass(frozen=True, slots=True)
class PolicyIdentity:
    """Immutable policy and processor identity advertised by one server process."""

    kind: str
    identifier: str
    revision: str
    model_sha256: str
    processor_sha256: str
    profile: str
    control_hz: int = 20
    action_horizon: int = POLICY_ACTION_HORIZON
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind not in ("base", "peft_adapter"):
            raise ValueError("policy kind must be base or peft_adapter")
        for label, value in (
            ("identifier", self.identifier),
            ("revision", self.revision),
            ("profile", self.profile),
        ):
            _string(value, label)
        for label, value in (
            ("model_sha256", self.model_sha256),
            ("processor_sha256", self.processor_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{label} must be a lowercase SHA-256")
        if self.control_hz != 20 or self.action_horizon != POLICY_ACTION_HORIZON:
            raise ValueError("Stage 8 requires 20 Hz and a 50-step prediction horizon")
        if self.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("policy observation schema version mismatch")
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError("policy action schema version mismatch")

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "revision": self.revision,
            "model_sha256": self.model_sha256,
            "processor_sha256": self.processor_sha256,
            "profile": self.profile,
            "control_hz": self.control_hz,
            "action_horizon": self.action_horizon,
            "observation_schema_version": self.observation_schema_version,
            "action_schema_version": self.action_schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyIdentity:
        source = _object(value, "policy identity")
        return cls(
            kind=source.get("kind"),
            identifier=source.get("identifier"),
            revision=source.get("revision"),
            model_sha256=source.get("model_sha256"),
            processor_sha256=source.get("processor_sha256"),
            profile=source.get("profile"),
            control_hz=source.get("control_hz"),
            action_horizon=source.get("action_horizon"),
            observation_schema_version=source.get("observation_schema_version"),
            action_schema_version=source.get("action_schema_version"),
        )


@dataclass(frozen=True, slots=True)
class PolicyHealthResponse:
    status: str
    identity: PolicyIdentity
    runtime: Mapping[str, object]
    schema_version: str = POLICY_RPC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status != "ready":
            raise ValueError("healthy policy server status must be ready")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "identity": self.identity.to_dict(),
            "identity_sha256": self.identity.identity_sha256,
            "runtime": dict(self.runtime),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyHealthResponse:
        source = _object(value, "health response")
        _schema(source)
        identity = PolicyIdentity.from_dict(_object(source.get("identity"), "identity"))
        if source.get("identity_sha256") != identity.identity_sha256:
            raise ValueError("health response identity hash mismatch")
        return cls(
            status=source.get("status"),
            identity=identity,
            runtime=dict(_object(source.get("runtime"), "runtime")),
            schema_version=source.get("schema_version"),
        )


@dataclass(frozen=True, slots=True)
class PolicyResetRequest:
    request_id: str
    episode_id: str
    instruction: str
    noise_seed: int
    schema_version: str = POLICY_RPC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _string(self.request_id, "request_id")
        _string(self.episode_id, "episode_id")
        _string(self.instruction, "instruction")
        _integer(self.noise_seed, "noise_seed")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "noise_seed": self.noise_seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyResetRequest:
        source = _object(value, "reset request")
        _schema(source)
        return cls(
            request_id=source.get("request_id"),
            episode_id=source.get("episode_id"),
            instruction=source.get("instruction"),
            noise_seed=source.get("noise_seed"),
            schema_version=source.get("schema_version"),
        )


@dataclass(frozen=True, slots=True)
class PolicyResetResponse:
    request_id: str
    episode_id: str
    identity_sha256: str
    status: str = "ok"
    schema_version: str = POLICY_RPC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status != "ok":
            raise ValueError("reset response status must be ok")
        _string(self.request_id, "request_id")
        _string(self.episode_id, "episode_id")
        if len(self.identity_sha256) != 64:
            raise ValueError("identity_sha256 must be a SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyResetResponse:
        source = _object(value, "reset response")
        _schema(source)
        return cls(
            request_id=source.get("request_id"),
            episode_id=source.get("episode_id"),
            identity_sha256=source.get("identity_sha256"),
            status=source.get("status"),
            schema_version=source.get("schema_version"),
        )


@dataclass(frozen=True, slots=True)
class LivePolicyObservation:
    """Portable 9D state and base64 CHW uint8 RGB used on the wire."""

    joint_position: tuple[float, ...]
    front_rgb_base64: str

    def __post_init__(self) -> None:
        if len(self.joint_position) != POLICY_STATE_DIMENSION:
            raise ValueError("live joint position must have dimension 9")
        if not all(math.isfinite(value) for value in self.joint_position):
            raise ValueError("live joint position contains non-finite values")
        self.rgb_bytes()

    def rgb_bytes(self) -> bytes:
        try:
            result = base64.b64decode(self.front_rgb_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("front RGB is not valid base64") from error
        if len(result) != POLICY_RGB_BYTE_COUNT:
            raise ValueError("front RGB byte count does not match 3x224x224 uint8")
        return result

    @classmethod
    def from_rgb_bytes(
        cls,
        joint_position: Sequence[float],
        rgb_bytes: bytes,
    ) -> LivePolicyObservation:
        if len(rgb_bytes) != POLICY_RGB_BYTE_COUNT:
            raise ValueError("front RGB byte count does not match 3x224x224 uint8")
        return cls(
            joint_position=_float_tuple(joint_position, POLICY_STATE_DIMENSION, "joint_position"),
            front_rgb_base64=base64.b64encode(rgb_bytes).decode("ascii"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "robot.joint_position": list(self.joint_position),
            "camera.front.rgb": {
                "dtype": "uint8",
                "shape": list(POLICY_RGB_SHAPE),
                "encoding": "base64",
                "data": self.front_rgb_base64,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LivePolicyObservation:
        source = _object(value, "observation")
        image = _object(source.get("camera.front.rgb"), "camera.front.rgb")
        if image.get("dtype") != "uint8" or image.get("encoding") != "base64":
            raise ValueError("front RGB must use uint8/base64")
        if tuple(image.get("shape", ())) != POLICY_RGB_SHAPE:
            raise ValueError("front RGB shape must be 3x224x224")
        return cls(
            joint_position=_float_tuple(
                source.get("robot.joint_position"),
                POLICY_STATE_DIMENSION,
                "robot.joint_position",
            ),
            front_rgb_base64=_string(image.get("data"), "camera.front.rgb.data"),
        )


@dataclass(frozen=True, slots=True)
class PolicyInferenceRequest:
    request_id: str
    episode_id: str
    step_index: int
    instruction: str
    noise_seed: int
    observation: LivePolicyObservation
    schema_version: str = POLICY_RPC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _string(self.request_id, "request_id")
        _string(self.episode_id, "episode_id")
        _integer(self.step_index, "step_index")
        _string(self.instruction, "instruction")
        _integer(self.noise_seed, "noise_seed")
        if not isinstance(self.observation, LivePolicyObservation):
            raise ValueError("observation must be LivePolicyObservation")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "instruction": self.instruction,
            "noise_seed": self.noise_seed,
            "observation": self.observation.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyInferenceRequest:
        source = _object(value, "inference request")
        _schema(source)
        return cls(
            request_id=source.get("request_id"),
            episode_id=source.get("episode_id"),
            step_index=source.get("step_index"),
            instruction=source.get("instruction"),
            noise_seed=source.get("noise_seed"),
            observation=LivePolicyObservation.from_dict(
                _object(source.get("observation"), "observation")
            ),
            schema_version=source.get("schema_version"),
        )


@dataclass(frozen=True, slots=True)
class PolicyInferenceResponse:
    request_id: str
    episode_id: str
    step_index: int
    identity_sha256: str
    action_chunk: tuple[tuple[float, ...], ...]
    inference_latency_ms: float
    diagnostics: Mapping[str, object]
    status: str = "ok"
    schema_version: str = POLICY_RPC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status != "ok":
            raise ValueError("inference response status must be ok")
        _string(self.request_id, "request_id")
        _string(self.episode_id, "episode_id")
        _integer(self.step_index, "step_index")
        if len(self.identity_sha256) != 64:
            raise ValueError("identity_sha256 must be a SHA-256")
        if len(self.action_chunk) != POLICY_ACTION_HORIZON:
            raise ValueError("policy response must contain 50 actions")
        for index, action in enumerate(self.action_chunk):
            if len(action) != POLICY_ACTION_DIMENSION or not all(
                math.isfinite(component) for component in action
            ):
                raise ValueError(f"action_chunk[{index}] must contain seven finite values")
        if _finite(self.inference_latency_ms, "inference_latency_ms") < 0.0:
            raise ValueError("inference latency must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "identity_sha256": self.identity_sha256,
            "action_chunk": [list(action) for action in self.action_chunk],
            "inference_latency_ms": self.inference_latency_ms,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyInferenceResponse:
        source = _object(value, "inference response")
        _schema(source)
        chunk = source.get("action_chunk")
        if not isinstance(chunk, Sequence) or isinstance(chunk, (str, bytes, bytearray)):
            raise ValueError("action_chunk must be a sequence")
        return cls(
            request_id=source.get("request_id"),
            episode_id=source.get("episode_id"),
            step_index=source.get("step_index"),
            identity_sha256=source.get("identity_sha256"),
            action_chunk=tuple(
                _float_tuple(action, POLICY_ACTION_DIMENSION, f"action_chunk[{index}]")
                for index, action in enumerate(chunk)
            ),
            inference_latency_ms=source.get("inference_latency_ms"),
            diagnostics=dict(_object(source.get("diagnostics", {}), "diagnostics")),
            status=source.get("status"),
            schema_version=source.get("schema_version"),
        )


def error_response(error_type: str, message: str) -> dict[str, object]:
    return {
        "schema_version": POLICY_RPC_SCHEMA_VERSION,
        "status": "error",
        "error_type": _string(error_type, "error_type"),
        "message": _string(message, "message"),
    }


__all__ = [
    "LivePolicyObservation",
    "POLICY_ACTION_DIMENSION",
    "POLICY_ACTION_HORIZON",
    "POLICY_RGB_BYTE_COUNT",
    "POLICY_RGB_SHAPE",
    "POLICY_RPC_HEALTH_PATH",
    "POLICY_RPC_INFERENCE_PATH",
    "POLICY_RPC_RESET_PATH",
    "POLICY_RPC_SCHEMA_VERSION",
    "PolicyHealthResponse",
    "PolicyIdentity",
    "PolicyInferenceRequest",
    "PolicyInferenceResponse",
    "PolicyResetRequest",
    "PolicyResetResponse",
    "canonical_json_sha256",
    "decode_json",
    "encode_json",
    "error_response",
]
