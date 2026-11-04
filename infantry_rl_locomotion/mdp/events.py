"""Reset events for the fall-recovery task.

The recovery task starts the robot in a FALLEN pose instead of the nominal stance.
Only forward/backward falls are trained: the pitch is sampled in a configurable
magnitude range with a random sign, while roll is kept small.  The root is lifted a
little so the initial pose does not start deeply interpenetrating the ground; physics
settles it onto the ground during the first few steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def reset_root_state_front_back_fallen(
    env,
    env_ids: Sequence[int],
    pitch_range_deg: tuple[float, float] = (45.0, 120.0),
    roll_range_deg: tuple[float, float] = (-8.0, 8.0),
    yaw_range_deg: tuple[float, float] = (-180.0, 180.0),
    z_offset_range: tuple[float, float] = (0.05, 0.15),
    lin_vel_range: tuple[float, float] = (-0.10, 0.10),
    ang_vel_range: tuple[float, float] = (-0.30, 0.30),
    pitch_sign: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Place the robot in a random front/back fallen orientation at reset.

    The pitch magnitude is sampled from ``pitch_range_deg`` and its sign is random,
    so both nose-down and nose-up falls occur.  Set ``pitch_sign=-1`` (or +1) to
    force a backward (or forward) fall for evaluation.  Roll is deliberately small
    (front/back falls only, no side falls).  The root is raised by ``z_offset_range``
    above its default height so the rotated body/legs do not spawn inside the ground.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    if ids.numel() == 0:
        return
    n = ids.numel()
    device = asset.device
    root_states = asset.data.default_root_state[ids].clone()

    pitch_mag = torch.empty(n, device=device).uniform_(*pitch_range_deg) * (math.pi / 180.0)
    if pitch_sign > 0.0:
        pitch = pitch_mag
    elif pitch_sign < 0.0:
        pitch = -pitch_mag
    else:
        sign = torch.where(
            torch.rand(n, device=device) < 0.5,
            -torch.ones(n, device=device),
            torch.ones(n, device=device),
        )
        pitch = pitch_mag * sign
    roll = torch.empty(n, device=device).uniform_(*roll_range_deg) * (math.pi / 180.0)
    yaw = torch.empty(n, device=device).uniform_(*yaw_range_deg) * (math.pi / 180.0)
    z_offset = torch.empty(n, device=device).uniform_(*z_offset_range)

    positions = (
        root_states[:, 0:3]
        + env.scene.env_origins[ids]
        + torch.stack(
            (torch.zeros(n, device=device), torch.zeros(n, device=device), z_offset),
            dim=-1,
        )
    )
    orientation_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    lin_delta = torch.empty(n, 3, device=device).uniform_(*lin_vel_range)
    ang_delta = torch.empty(n, 3, device=device).uniform_(*ang_vel_range)
    # recovery is sagittal: no initial roll rate, which would break the symmetry
    ang_delta[:, 0] = 0.0
    velocities = torch.cat(
        (root_states[:, 7:10] + lin_delta, root_states[:, 10:13] + ang_delta), dim=-1
    )

    asset.write_root_pose_to_sim(
        torch.cat((positions, orientations), dim=-1), env_ids=ids
    )
    asset.write_root_velocity_to_sim(velocities, env_ids=ids)


def reset_hips_tucked(
    env,
    env_ids: Sequence[int],
    tuck_prob: float = 0.5,
    tuck_front_offset: float = -0.45,
    tuck_back_offset: float = 0.45,
    tuck_jitter: float = 0.03,
    random_offset: float = 0.15,
    velocity_range: tuple[float, float] = (-1.0, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset hips to the deepest symmetric squat with probability ``tuck_prob``.

    The measured five-bar tuck pose is ``front_hip = nominal - 0.45`` and
    ``back_hip = nominal + 0.45``; its leg length is 0.165 m instead of the nominal
    0.250 m, putting the wheel axle under the hip midpoint.  This is the starting
    point for the "tuck, drive the wheels, extend" recovery strategy.  The remaining
    environments get a small random offset so the policy eventually generalizes
    from arbitrary hip configurations.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    if ids.numel() == 0:
        return
    n = ids.numel()
    device = asset.device

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(asset.num_joints))
    joint_ids = list(list(joint_ids))
    names = [asset.joint_names[i] for i in joint_ids]

    def _sagittal(front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
        """Build per-joint offsets with the SAME value on the left and right sides."""
        out = torch.zeros(front.numel(), len(joint_ids), device=device)
        for j, name in enumerate(names):
            if "front_hip" in name:
                out[:, j] = front
            elif "back_hip" in name:
                out[:, j] = back
        return out

    default_pos = asset.data.default_joint_pos[ids][:, joint_ids].clone()
    default_vel = asset.data.default_joint_vel[ids][:, joint_ids].clone()

    # recovery is sagittal: randomize front/back offsets, then apply them to BOTH
    # left and right.  Randomizing each side independently is what produced the
    # 76% roll_limit in the first stage-1 attempt.
    front_rand = torch.empty(n, device=device).uniform_(-random_offset, random_offset)
    back_rand = torch.empty(n, device=device).uniform_(-random_offset, random_offset)
    offsets = _sagittal(front_rand, back_rand)
    if tuck_prob > 0.0:
        pick = torch.rand(n, device=device) < tuck_prob
        if pick.any():
            n_tuck = int(pick.sum().item())
            front = tuck_front_offset + torch.empty(n_tuck, device=device).uniform_(
                -tuck_jitter, tuck_jitter
            )
            back = tuck_back_offset + torch.empty(n_tuck, device=device).uniform_(
                -tuck_jitter, tuck_jitter
            )
            offsets[pick] = _sagittal(front, back)

    joint_pos = default_pos + offsets
    front_vel = torch.empty(n, device=device).uniform_(*velocity_range)
    back_vel = torch.empty(n, device=device).uniform_(*velocity_range)
    joint_vel = default_vel + _sagittal(front_vel, back_vel)
    pos_limits = asset.data.soft_joint_pos_limits[ids][:, joint_ids]
    joint_pos = joint_pos.clamp_(pos_limits[..., 0], pos_limits[..., 1])
    vel_limits = asset.data.soft_joint_vel_limits[ids][:, joint_ids]
    joint_vel = joint_vel.clamp_(-vel_limits, vel_limits)

    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=joint_ids, env_ids=ids
    )


def reset_root_state_standing_squat(
    env,
    env_ids: Sequence[int],
    pitch_range_deg: tuple[float, float] = (-5.0, 5.0),
    roll_range_deg: tuple[float, float] = (-3.0, 3.0),
    yaw_range_deg: tuple[float, float] = (-180.0, 180.0),
    z_offset_range: tuple[float, float] = (-0.085, -0.065),
    lin_vel_range: tuple[float, float] = (-0.05, 0.05),
    ang_vel_range: tuple[float, float] = (-0.10, 0.10),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Start UPRIGHT in the deepest squat, for the stand-from-squat skill.

    The tuck leg length is 0.165 m vs the nominal 0.250 m, so the supporting base
    height is ~0.105 m.  The default root height is ~0.190 m; the z offset lowers it
    to just above the ground.  The hips are set to the tuck by the sibling
    ``reset_hips_tucked`` event; the action term's zero target is nominal, so the
    position controller itself performs the leg extension while the policy balances.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    if ids.numel() == 0:
        return
    n = ids.numel()
    device = asset.device
    root_states = asset.data.default_root_state[ids].clone()

    pitch = torch.empty(n, device=device).uniform_(*pitch_range_deg) * (math.pi / 180.0)
    roll = torch.empty(n, device=device).uniform_(*roll_range_deg) * (math.pi / 180.0)
    yaw = torch.empty(n, device=device).uniform_(*yaw_range_deg) * (math.pi / 180.0)
    z_offset = torch.empty(n, device=device).uniform_(*z_offset_range)

    positions = (
        root_states[:, 0:3]
        + env.scene.env_origins[ids]
        + torch.stack(
            (torch.zeros(n, device=device), torch.zeros(n, device=device), z_offset),
            dim=-1,
        )
    )
    orientation_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)
    lin_delta = torch.empty(n, 3, device=device).uniform_(*lin_vel_range)
    ang_delta = torch.empty(n, 3, device=device).uniform_(*ang_vel_range)
    velocities = torch.cat(
        (root_states[:, 7:10] + lin_delta, root_states[:, 10:13] + ang_delta), dim=-1
    )

    asset.write_root_pose_to_sim(
        torch.cat((positions, orientations), dim=-1), env_ids=ids
    )
    asset.write_root_velocity_to_sim(velocities, env_ids=ids)


__all__ = [
    "reset_hips_tucked",
    "reset_root_state_front_back_fallen",
    "reset_root_state_standing_squat",
]
