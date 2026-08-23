"""Small MDP terms required by the Franka pick-and-place skeleton."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg


def camera_rgb_chw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera_front"),
) -> torch.Tensor:
    """Return an unnormalized uint8 RGB image in contract-canonical CHW layout."""

    image = isaac_mdp.image(
        env,
        sensor_cfg=sensor_cfg,
        data_type="rgb",
        normalize=False,
    )
    if image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise RuntimeError(f"expected batched HWC RGB/RGBA data, got shape {tuple(image.shape)}")
    return image[..., :3].permute(0, 3, 1, 2).contiguous()


def zero_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Explicit placeholder until the evaluation interface is implemented."""

    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
