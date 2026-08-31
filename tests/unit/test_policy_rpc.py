import json

import pytest

from embodied_ai.contracts.policy_rpc import (
    POLICY_RGB_BYTE_COUNT,
    LivePolicyObservation,
    PolicyInferenceRequest,
    PolicyInferenceResponse,
)


def test_policy_rpc_round_trip_and_exact_chunk() -> None:
    observation = LivePolicyObservation.from_rgb_bytes([0.0] * 9, bytes(POLICY_RGB_BYTE_COUNT))
    request = PolicyInferenceRequest("request-1", "episode-1", 0, "Do task.", 7, observation)
    assert PolicyInferenceRequest.from_dict(json.loads(json.dumps(request.to_dict()))) == request
    chunk = tuple((0.0,) * 7 for _ in range(50))
    response = PolicyInferenceResponse("request-1", "episode-1", 0, "a" * 64, chunk, 1.0, {})
    assert PolicyInferenceResponse.from_dict(response.to_dict()) == response


def test_policy_rpc_rejects_short_chunk() -> None:
    with pytest.raises(ValueError, match="50 actions"):
        PolicyInferenceResponse("r", "e", 0, "a" * 64, ((0.0,) * 7,), 1.0, {})
