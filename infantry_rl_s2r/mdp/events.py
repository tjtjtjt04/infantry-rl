"""Reset events for the squat tasks."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg


def reset_root_state_standing_squat(
    env,
    env_ids: torch.Tensor,
    pitch_range_deg: tuple[float, float] = (-4.0, 4.0),
    roll_range_deg: tuple[float, float] = (-2.0, 2.0),
    yaw_range_deg: tuple[float, float] = (-180.0, 180.0),
    z_offset_range: tuple[float, float] = (-0.004, 0.004),
    lin_vel_range: tuple[float, float] = (-0.05, 0.05),
    ang_vel_range: tuple[float, float] = (-0.10, 0.10),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Start upright at the squat height with small pose/velocity noise.

    ``z_offset_range`` is relative to the asset's default root height, so the caller
    passes ``SQUAT_Z_OFFSET`` (squat height minus nominal height).  Orientation noise
    is applied on top of the default orientation; the joints are written separately by
    ``reset_squat_pose``.
    """
    asset = env.scene[asset_cfg.name]
    ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    if ids.numel() == 0:
        return
    n = ids.numel()
    device = asset.device
    root_states = asset.data.default_root_state[ids].clone()

    def uni(lo_hi: tuple[float, float]) -> torch.Tensor:
        return torch.empty(n, device=device).uniform_(float(lo_hi[0]), float(lo_hi[1]))

    xyz = root_states[:, 0:3] + env.scene.env_origins[ids]
    xyz[:, 2] += uni(z_offset_range)
    roll = torch.deg2rad(uni(roll_range_deg))
    pitch = torch.deg2rad(uni(pitch_range_deg))
    yaw = torch.deg2rad(uni(yaw_range_deg))
    orientation = math_utils.quat_mul(
        root_states[:, 3:7], math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    )
    velocity = root_states[:, 7:13].clone()
    velocity[:, 0:3] += torch.empty(n, 3, device=device).uniform_(
        float(lin_vel_range[0]), float(lin_vel_range[1])
    )
    velocity[:, 3:6] += torch.empty(n, 3, device=device).uniform_(
        float(ang_vel_range[0]), float(ang_vel_range[1])
    )

    asset.write_root_pose_to_sim(torch.cat([xyz, orientation], dim=-1), env_ids=ids)
    asset.write_root_velocity_to_sim(velocity, env_ids=ids)


__all__ = ["reset_root_state_standing_squat"]
