"""Termination terms for the squat tasks."""

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


__all__ = ["pitch_above", "roll_above"]
