#!/usr/bin/env python3
"""Encode one contract RGB observation stream as a derived MP4 preview."""

from __future__ import annotations

import argparse
from pathlib import Path

from embodied_ai.data.episode_video import encode_episode_camera_video


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("episode_directory", type=Path)
parser.add_argument("output_path", type=Path)
parser.add_argument("--camera_key", default="camera.front.rgb")
parser.add_argument("--fps", type=float, default=None)
parser.add_argument("--ffmpeg_binary", default="ffmpeg")
args = parser.parse_args()


def main() -> None:
    encoded = encode_episode_camera_video(
        args.episode_directory,
        args.output_path,
        camera_key=args.camera_key,
        fps=args.fps,
        ffmpeg_binary=args.ffmpeg_binary,
    )
    print(
        "EPISODE_VIDEO_OK",
        f"path={encoded.path}",
        f"camera={encoded.camera_key}",
        f"frames={encoded.frame_count}",
        f"size={encoded.width}x{encoded.height}",
        f"fps={encoded.fps:g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
