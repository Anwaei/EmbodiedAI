#!/usr/bin/env python3
"""Start the Stage 8 SmolVLA policy server on loopback."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from embodied_ai.evaluation.config import Stage8RunConfig
from embodied_ai.policies.smolvla.config import Stage7RunConfig
from embodied_ai.policies.smolvla.online import SmolVLAOnlineEngine, serve


def main() -> None:
    repository = Path(os.environ.get("EMBODIEDAI_REPO", "/root/projects/EmbodiedAI"))
    data_root = Path(os.environ.get("EMBODIEDAI_DATA", "/root/autodl-tmp/EmbodiedAI"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation_config",
        type=Path,
        default=repository / "configs/evaluation/stage8_franka_pick_place_v1.toml",
    )
    parser.add_argument("--policy_kind", choices=("base", "peft_adapter"), required=True)
    args = parser.parse_args()

    evaluation = Stage8RunConfig.from_toml(
        args.evaluation_config, repository_root=repository, data_root=data_root
    )
    stage7 = Stage7RunConfig.from_toml(
        evaluation.stage7_config_path, repository_root=repository
    )
    engine = SmolVLAOnlineEngine(
        run_config=stage7,
        processor_dir=evaluation.processor_dir,
        policy_kind=args.policy_kind,
        adapter_dir=evaluation.adapter_dir if args.policy_kind == "peft_adapter" else None,
    )
    identity_path = (
        data_root / "runs/stage8" / evaluation.run_id / args.policy_kind
        / "policy_identity.json"
    )
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": "embodied-ai.stage8-policy-identity/v1",
        "identity": engine.identity.to_dict(),
        "identity_sha256": engine.identity.identity_sha256,
        "runtime": dict(engine.runtime),
    }
    temporary = identity_path.parent / f".{identity_path.name}.{uuid.uuid4().hex}.partial"
    temporary.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, identity_path)
    print(
        "STAGE8_POLICY_SERVER_READY",
        f"kind={args.policy_kind}",
        f"identity={engine.identity.identity_sha256}",
        f"address={evaluation.host}:{evaluation.port}",
        f"identity_artifact={identity_path}",
        flush=True,
    )
    serve(
        engine,
        host=evaluation.host,
        port=evaluation.port,
        max_payload_bytes=evaluation.max_payload_bytes,
    )


if __name__ == "__main__":
    main()
