"""Termination terms for the squat / fall-recovery tasks."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg


def pitch_above(
    env,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where ``|pitch| > limit_angle`` (rad)."""
    asset = env.scene[asset_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return pitch.abs() > float(limit_angle)


def roll_above(
    env,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where ``|roll| > limit_angle`` (rad)."""
    asset = env.scene[asset_cfg.name]
    roll, _, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return roll.abs() > float(limit_angle)


def upright_stable(
    env,
    pitch_limit: float = 0.12,
    roll_limit: float = 0.15,
    height_min: float = 0.155,
    height_max: float = 0.20,
    rate_limit: float = 0.7,
    hold_steps: int = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True once the base has been upright and slow for ``hold_steps`` control steps.

    "Upright" is the state the *balance* policy is happy with: deep inside its 0.24 rad
    pitch envelope, at the *squat* height, and almost stationary.  Requiring it for
    ``hold_steps`` consecutive steps (0.1 s at 100 Hz) rejects the moment the robot
    merely passes through upright while rocking.

    ``height_max`` matters for the hand-off: the balance policy holds the retracted
    (squat) stance, and extending the legs raises the base to ~0.287 m.  A recovery that
    "stands up tall" is not finished -- the successor policy would have to retract the
    legs while balancing.  The band keeps success at the squat height.

    The counter lives on the environment (``env._getup_hold``) and the per-episode
    success flag is ``env.getup_success``; the task's ``_reset_idx`` clears both.  The
    same flag feeds :func:`mdp.rewards.recovery_success`, so the terminal bonus is
    paid exactly on the step this terminates.
    """
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    rate = asset.data.root_ang_vel_b.norm(dim=-1)
    ok = (
        (pitch.abs() < float(pitch_limit))
        & (roll.abs() < float(roll_limit))
        & (height > float(height_min))
        & (height < float(height_max))
        & (rate < float(rate_limit))
    )

    hold = getattr(env, "_getup_hold", None)
    if hold is None or hold.shape[0] != env.num_envs:
        hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._getup_hold = hold
    hold[:] = torch.where(ok, hold + 1, torch.zeros_like(hold))
    done = hold >= int(hold_steps)

    success = getattr(env, "getup_success", None)
    if success is not None and bool(done.any()):
        success[done] = True
    return done


__all__ = ["pitch_above", "roll_above", "upright_stable"]

