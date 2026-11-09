"""Reset events for the squat / fall-recovery tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from robot_model import fallen_geometry


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


def reset_root_state_fallen(
    env,
    env_ids: Sequence[int],
    fall_pitch_range_deg: tuple[float, float] = (30.0, 40.0),
    rest_pitch_range_deg: tuple[float, float] = (15.0, 19.0),
    rest_probability: float = 0.4,
    roll_range_deg: tuple[float, float] = (-1.5, 1.5),
    yaw_range_deg: tuple[float, float] = (-180.0, 180.0),
    contact_margin: float = 0.0015,
    lin_vel_range: tuple[float, float] = (-0.03, 0.03),
    ang_vel_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Start the robot on the ground at a fore/aft pitch, wheels included.

    The root height is **not** an offset from the standing height: it comes from the
    measured contact table, ``robot_model.fallen_geometry()``, so the robot is placed
    exactly on the ground for the sampled pitch and never starts inside it.

    Two populations are mixed (``rest_probability``):

    * ``rest_pitch_range_deg`` -- the **resting tripod** at |pitch| ~ 15-19 deg, where
      the two wheels and a chassis corner are all on the ground.  This is the state the
      robot actually ends up in (measured: every pitch >= 20 deg falls back to it), and
      it is the stable local optimum the policy has to escape;
    * ``fall_pitch_range_deg`` -- the requested 30-40 deg fallen *transient*, where only
      the chassis corner touches and the wheels are 2-5 cm in the air.  Gravity rotates
      the robot back down from there, so the policy has to catch it as the wheels land.

    The sampled pitch is stored on the environment as ``env.reset_pitch_deg`` so the
    task can log the actual distribution it trains on.  The joints are written by the
    task's separate pose reset term, exactly as for the standing task.
    """
    asset = env.scene[asset_cfg.name]
    ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    if ids.numel() == 0:
        return
    n = ids.numel()
    device = asset.device

    def uni(lo_hi: tuple[float, float]) -> torch.Tensor:
        return torch.empty(n, device=device).uniform_(float(lo_hi[0]), float(lo_hi[1]))

    # -- pitch: a mixture of the resting tripod and the 30-40 deg transient --
    rest = torch.rand(n, device=device) < float(rest_probability)
    magnitude = torch.where(rest, uni(rest_pitch_range_deg), uni(fall_pitch_range_deg))
    sign = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    pitch = torch.deg2rad(magnitude) * sign
    roll = torch.deg2rad(uni(roll_range_deg))
    yaw = torch.deg2rad(uni(yaw_range_deg))

    root_states = asset.data.default_root_state[ids].clone()
    xyz = root_states[:, 0:3] + env.scene.env_origins[ids]
    ground = fallen_geometry()["contact_height"](pitch)
    jitter = torch.empty(n, device=device).uniform_(-0.001, 0.001)
    xyz[:, 2] = ground + float(contact_margin) + jitter
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

    pitch_deg = getattr(env, "reset_pitch_deg", None)
    if pitch_deg is not None:
        pitch_deg[ids] = torch.rad2deg(pitch)


__all__ = ["reset_root_state_fallen", "reset_root_state_standing_squat"]
