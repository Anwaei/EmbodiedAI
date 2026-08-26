"""Encode contract RGB observations from an immutable NPY episode as MP4."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from embodied_ai.contracts import DataType, EpisodeMetadata, ObservationKind


@dataclass(frozen=True, slots=True)
class EncodedEpisodeVideo:
    """Description of one derived episode preview video."""

    path: Path
    camera_key: str
    frame_count: int
    width: int
    height: int
    fps: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observation_payload_path(key: str) -> str:
    filename = key.replace(".", "_").replace("-", "_")
    return f"observations/{filename}.npy"


def encode_episode_camera_video(
    episode_directory: str | Path,
    output_path: str | Path,
    *,
    camera_key: str = "camera.front.rgb",
    fps: float | None = None,
    ffmpeg_binary: str = "ffmpeg",
) -> EncodedEpisodeVideo:
    """Validate one camera payload and stream its CHW RGB frames to FFmpeg."""

    episode_root = Path(episode_directory).expanduser().resolve()
    manifest_path = episode_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"episode manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as stream:
        metadata = EpisodeMetadata.from_dict(json.load(stream))

    try:
        camera_field = metadata.observation_schema.field(camera_key)
    except KeyError as error:
        raise ValueError(f"episode has no observation field {camera_key!r}") from error
    if camera_field.kind is not ObservationKind.RGB_IMAGE:
        raise ValueError(f"observation field {camera_key!r} is not an RGB image")
    if camera_field.dtype is not DataType.UINT8:
        raise ValueError(f"observation field {camera_key!r} is not uint8")

    relative_camera_path = _observation_payload_path(camera_key)
    payload_by_path = {payload.path: payload for payload in metadata.payloads}
    payload = payload_by_path.get(relative_camera_path)
    if payload is None:
        raise ValueError(f"manifest has no camera payload {relative_camera_path!r}")
    camera_path = episode_root.joinpath(*PurePosixPath(relative_camera_path).parts)
    if not camera_path.is_file():
        raise FileNotFoundError(f"camera payload does not exist: {camera_path}")
    if camera_path.stat().st_size != payload.byte_size:
        raise ValueError("camera payload byte size does not match the manifest")
    if _sha256(camera_path) != payload.sha256:
        raise ValueError("camera payload checksum does not match the manifest")

    frames = np.load(camera_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (metadata.step_count, *camera_field.shape)
    if frames.shape != expected_shape:
        raise ValueError(f"camera payload has shape {frames.shape}, expected {expected_shape}")
    if frames.dtype != np.dtype(camera_field.dtype.value):
        raise ValueError(
            f"camera payload has dtype {frames.dtype}, expected {camera_field.dtype.value}"
        )

    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("episode preview output must use the .mp4 suffix")
    if output == episode_root or output.is_relative_to(episode_root):
        raise ValueError("derived video must be written outside the immutable episode")
    if output.exists():
        raise FileExistsError(f"video output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    video_fps = metadata.action_schema.control_hz if fps is None else fps
    if isinstance(video_fps, bool) or not isinstance(video_fps, (int, float)):
        raise ValueError("video fps must be a number")
    video_fps = float(video_fps)
    if not math.isfinite(video_fps) or video_fps <= 0.0:
        raise ValueError("video fps must be finite and positive")

    ffmpeg_path = shutil.which(ffmpeg_binary)
    if ffmpeg_path is None:
        raise FileNotFoundError(f"FFmpeg executable was not found: {ffmpeg_binary}")

    frame_count, _, height, width = frames.shape
    partial_output = output.parent / (
        f".{output.stem}.partial-{uuid.uuid4().hex}.mp4"
    )
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(video_fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial_output),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        if process.stdin is None or process.stderr is None:
            raise RuntimeError("failed to open FFmpeg pipes")
        # NPY stores contract images as CHW RGB; rawvideo expects packed HWC RGB bytes.
        try:
            for frame_chw in frames:
                frame_hwc = np.ascontiguousarray(frame_chw.transpose(1, 2, 0))
                process.stdin.write(frame_hwc.tobytes())
        finally:
            with suppress(BrokenPipeError):
                process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        process.stderr.close()
        return_code = process.wait()
        if return_code != 0:
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(f"FFmpeg exited with status {return_code}{detail}")
        if not partial_output.is_file() or partial_output.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce a non-empty video")
        if output.exists():
            raise FileExistsError(f"video output already exists: {output}")
        # Publish only a complete video; partial files are private implementation state.
        partial_output.rename(output)
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stderr is not None:
            process.stderr.close()
        with suppress(FileNotFoundError):
            partial_output.unlink()
        raise

    return EncodedEpisodeVideo(
        path=output,
        camera_key=camera_key,
        frame_count=int(frame_count),
        width=int(width),
        height=int(height),
        fps=video_fps,
    )


__all__ = ["EncodedEpisodeVideo", "encode_episode_camera_video"]
