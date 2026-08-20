#!/usr/bin/env python3
"""Bounded Isaac Sim PhysX and offscreen-camera smoke test."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "renderer": "RaytracedLighting",
            "width": 64,
            "height": 64,
        }
    )

    try:
        import numpy as np

        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.sensors.camera import Camera

        world = World(stage_units_in_meters=1.0)
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Stage4Ground",
                name="stage4_ground",
                position=np.array([0.0, 0.0, -0.05]),
                scale=np.array([10.0, 10.0, 0.1]),
                color=np.array([0.25, 0.25, 0.25]),
            )
        )
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Stage4Cube",
                name="stage4_cube",
                position=np.array([0.0, 0.0, 1.0]),
                scale=np.array([0.25, 0.25, 0.25]),
                color=np.array([0.2, 0.6, 0.9]),
            )
        )
        camera = world.scene.add(
            Camera(
                prim_path="/World/Stage4Camera",
                name="stage4_camera",
                position=np.array([2.0, 2.0, 2.0]),
                frequency=20,
                resolution=(64, 64),
            )
        )

        world.reset()
        camera.initialize()
        for _ in range(args.steps):
            world.step(render=True)

        rgba = camera.get_rgba()
        if rgba is None or tuple(rgba.shape) != (64, 64, 4):
            raise RuntimeError(f"unexpected camera frame: {getattr(rgba, 'shape', None)}")

        position, _ = cube.get_world_pose()
        if not np.isfinite(position).all():
            raise RuntimeError(f"non-finite PhysX pose: {position}")

        print(
            "STAGE4_ISAAC_CORE_OK",
            f"steps={args.steps}",
            f"frame_shape={tuple(rgba.shape)}",
            f"cube_z={float(position[2]):.6f}",
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
