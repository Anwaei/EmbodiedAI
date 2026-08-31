"""Atomic Stage 8 rollout recorder with integrity metadata and optional MP4."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from embodied_ai.contracts.episode import PayloadFile
from embodied_ai.contracts.evaluation import (
    EvaluationRolloutMetadata,
    EvaluationScenario,
    PolicyDescriptor,
    RolloutOutcome,
)

from .metrics import rollout_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(path: Path, root: Path, media_type: str) -> PayloadFile:
    return PayloadFile(
        path=path.relative_to(root).as_posix(),
        media_type=media_type,
        byte_size=path.stat().st_size,
        sha256=_sha256(path),
    )


class EvaluationRolloutRecorder:
    def __init__(self, output_dir: Path, *, control_hz: int = 20) -> None:
        if control_hz != 20:
            raise ValueError("Stage 8 recording requires 20 Hz")
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite rollout: {output_dir}")
        self.output_dir = output_dir
        self.control_hz = control_hz
        self.joints: list[np.ndarray] = []
        self.images: list[np.ndarray] = []
        self.raw_actions: list[np.ndarray] = []
        self.executed_actions: list[np.ndarray] = []
        self.cube_positions: list[np.ndarray] = []
        self.goal_errors: list[float] = []
        self.cube_speeds: list[float] = []
        self.gripper_open: list[bool] = []
        self.timestamps_ns: list[int] = []
        self.inference_records: list[dict[str, object]] = []

    def append(
        self,
        *,
        joint_position: np.ndarray,
        camera_rgb: np.ndarray,
        raw_action: Sequence[float],
        executed_action: Sequence[float],
        cube_position: Sequence[float],
        goal_error_m: float,
        cube_speed_m_s: float,
        gripper_open: bool,
        timestamp_ns: int,
    ) -> None:
        joint = np.asarray(joint_position, dtype=np.float32)
        image = np.asarray(camera_rgb, dtype=np.uint8)
        raw = np.asarray(raw_action, dtype=np.float32)
        executed = np.asarray(executed_action, dtype=np.float32)
        cube = np.asarray(cube_position, dtype=np.float32)
        if joint.shape != (9,) or image.shape != (3, 224, 224):
            raise ValueError("rollout observation has an invalid contract shape")
        if raw.shape != (7,) or executed.shape != (7,) or cube.shape != (3,):
            raise ValueError("rollout action or cube state has an invalid shape")
        if self.timestamps_ns and timestamp_ns <= self.timestamps_ns[-1]:
            raise ValueError("rollout timestamps must be strictly increasing")
        self.joints.append(joint.copy())
        self.images.append(image.copy())
        self.raw_actions.append(raw.copy())
        self.executed_actions.append(executed.copy())
        self.cube_positions.append(cube.copy())
        self.goal_errors.append(float(goal_error_m))
        self.cube_speeds.append(float(cube_speed_m_s))
        self.gripper_open.append(bool(gripper_open))
        self.timestamps_ns.append(int(timestamp_ns))

    def add_inference_record(self, value: Mapping[str, object]) -> None:
        self.inference_records.append(dict(value))

    def finalize(
        self,
        *,
        rollout_id: str,
        run_id: str,
        scenario: EvaluationScenario,
        policy: PolicyDescriptor,
        noise_seed: int,
        outcome: RolloutOutcome,
        termination_reason: str,
        provenance: Mapping[str, object],
        record_video: bool,
    ) -> Path:
        if not self.joints:
            raise RuntimeError("cannot finalize an empty rollout")
        partial = self.output_dir.parent / f".{self.output_dir.name}.{uuid.uuid4().hex}.partial"
        partial.mkdir(parents=True, exist_ok=False)
        try:
            arrays = {
                "joint_position.npy": np.stack(self.joints),
                "camera_front_rgb.npy": np.stack(self.images),
                "raw_action.npy": np.stack(self.raw_actions),
                "executed_action.npy": np.stack(self.executed_actions),
                "cube_position_env_m.npy": np.stack(self.cube_positions),
                "goal_error_m.npy": np.asarray(self.goal_errors, dtype=np.float32),
                "cube_speed_m_s.npy": np.asarray(self.cube_speeds, dtype=np.float32),
                "gripper_open.npy": np.asarray(self.gripper_open, dtype=np.bool_),
                "timestamp_ns.npy": np.asarray(self.timestamps_ns, dtype=np.int64),
            }
            payloads: list[PayloadFile] = []
            for name, value in arrays.items():
                path = partial / name
                with path.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                payloads.append(_payload(path, partial, "application/x-npy"))
            inference_path = partial / "inference_requests.jsonl"
            with inference_path.open("x", encoding="utf-8", newline="\n") as stream:
                for record in self.inference_records:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            payloads.append(_payload(inference_path, partial, "application/x-ndjson"))
            if record_video:
                video_path = partial / "camera_front.mp4"
                self._encode_video(video_path, arrays["camera_front_rgb.npy"])
                payloads.append(_payload(video_path, partial, "video/mp4"))

            metrics = rollout_metrics(
                cube_positions=arrays["cube_position_env_m.npy"],
                goal_errors=arrays["goal_error_m.npy"],
                executed_actions=arrays["executed_action.npy"],
                inference_latencies_ms=(
                    float(record["inference_latency_ms"])
                    for record in self.inference_records
                ),
                success=outcome is RolloutOutcome.SUCCESS,
            )
            manifest = EvaluationRolloutMetadata(
                rollout_id=rollout_id,
                run_id=run_id,
                scenario=scenario,
                policy=policy,
                noise_seed=noise_seed,
                prediction_horizon=50,
                execute_horizon=5,
                step_count=len(self.joints),
                outcome=outcome,
                termination_reason=termination_reason,
                metrics=metrics,
                provenance=provenance,
                payloads=tuple(payloads),
            )
            (partial / "manifest.json").write_text(
                json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(partial, self.output_dir)
        finally:
            if partial.exists():
                shutil.rmtree(partial)
        return self.output_dir / "manifest.json"

    def _encode_video(self, path: Path, chw: np.ndarray) -> None:
        process = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", "224x224", "-r", str(self.control_hz),
                "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        assert process.stdin is not None
        try:
            process.stdin.write(np.transpose(chw, (0, 2, 3, 1)).tobytes())
            process.stdin.close()
            return_code = process.wait()
        finally:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


__all__ = ["EvaluationRolloutRecorder"]
