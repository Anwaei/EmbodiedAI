"""Small synchronous stdlib HTTP client for the loopback policy service."""

from __future__ import annotations

import http.client
import json
from collections.abc import Mapping

from embodied_ai.contracts.policy_rpc import (
    POLICY_RPC_HEALTH_PATH,
    POLICY_RPC_INFERENCE_PATH,
    POLICY_RPC_RESET_PATH,
    PolicyHealthResponse,
    PolicyInferenceRequest,
    PolicyInferenceResponse,
    PolicyResetRequest,
    PolicyResetResponse,
    decode_json,
    encode_json,
)


class PolicyTransportError(RuntimeError):
    pass


class PolicyHttpClient:
    def __init__(self, host: str, port: int, *, timeout_s: float, max_payload_bytes: int) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.max_payload_bytes = max_payload_bytes

    def _exchange(self, method: str, path: str, payload: Mapping[str, object] | None) -> dict:
        body = None if payload is None else encode_json(payload)
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            length = response.getheader("Content-Length")
            if length is not None and int(length) > self.max_payload_bytes:
                raise PolicyTransportError("policy response exceeds payload limit")
            response_body = response.read(self.max_payload_bytes + 1)
            if len(response_body) > self.max_payload_bytes:
                raise PolicyTransportError("policy response exceeds payload limit")
            decoded = decode_json(response_body, max_bytes=self.max_payload_bytes)
            if response.status != 200 or decoded.get("status") == "error":
                raise PolicyTransportError(
                    f"policy service returned HTTP {response.status}: "
                    f"{decoded.get('message', decoded)}"
                )
            return decoded
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            raise PolicyTransportError(f"policy service request failed: {error}") from error
        finally:
            connection.close()

    def health(self) -> PolicyHealthResponse:
        return PolicyHealthResponse.from_dict(self._exchange("GET", POLICY_RPC_HEALTH_PATH, None))

    def reset(self, request: PolicyResetRequest) -> PolicyResetResponse:
        response = PolicyResetResponse.from_dict(
            self._exchange("POST", POLICY_RPC_RESET_PATH, request.to_dict())
        )
        if response.request_id != request.request_id or response.episode_id != request.episode_id:
            raise PolicyTransportError("reset response correlation mismatch")
        return response

    def infer(self, request: PolicyInferenceRequest) -> PolicyInferenceResponse:
        response = PolicyInferenceResponse.from_dict(
            self._exchange("POST", POLICY_RPC_INFERENCE_PATH, request.to_dict())
        )
        if (
            response.request_id != request.request_id
            or response.episode_id != request.episode_id
            or response.step_index != request.step_index
        ):
            raise PolicyTransportError("inference response correlation mismatch")
        return response


__all__ = ["PolicyHttpClient", "PolicyTransportError"]
