"""Reward terms for the pure-torque squat task.

The standard terms delegate to :mod:`isaaclab.envs.mdp`; the task-specific ones --
action jerk, wheel spin, axle skew, loop-closure error, hip-lock deviation and the
fall penalty -- are implemented here.
"""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def flat_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Keep the base level: ``sum(projected_gravity_b[:, :2] ** 2)``."""
    return mdp.flat_orientation_l2(env, asset_cfg)


def base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Quadratic penalty on the root height error, ``(N,)``."""
    return mdp.base_height_l2(env, target_height=target_height, asset_cfg=asset_cfg)


def action_rate_l2(env) -> torch.Tensor:
    """Quadratic penalty on the change of the action between two control steps."""
    return mdp.action_rate_l2(env)


def action_jerk_l2(
    env, action_name: str = "joints", exclude_wheels: bool = False
) -> torch.Tensor:
    """``sum((a[t] - 2 a[t-1] + a[t-2])^2)`` over the action dims, ``(N,)``.

    Targets the high-frequency leg tremor specifically.  ``action_rate`` (the first
    difference) charges every change, so making it strong enough to kill a tremor also
    makes the legs too stiff to catch a pitch disturbance -- the two requirements fight
    each other.  The second difference breaks the conflict: it is ~0 for any smooth
    trajectory (constant, ramp, even accelerating) and only large when the action
    reverses direction between consecutive steps, which is exactly what a dithering leg
    looks like.
    """
    term = env.action_manager.get_term(action_name)
    jerk = term.action_jerk()
    if exclude_wheels:
        jerk = jerk[:, : len(term.hip_ids)]
    return torch.sum(jerk**2, dim=-1)


def joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Quadratic penalty on the applied joint torques."""
    return mdp.joint_torques_l2(env, asset_cfg)


def joint_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Quadratic penalty on the joint accelerations."""
    return mdp.joint_acc_l2(env, asset_cfg)


def joint_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Quadratic penalty on the joint velocities."""
    return mdp.joint_vel_l2(env, asset_cfg)


def wheel_spin_l2(env, action_name: str = "joints") -> torch.Tensor:
    """Small penalty on the wheel spin rate, ``sum(v_wheel ** 2)`` over both wheels."""
    term = env.action_manager.get_term(action_name)
    return torch.sum(term._robot.data.joint_vel[:, term.wheel_ids] ** 2, dim=-1)


def leg_axle_skew_abs(env, action_name: str = "joints") -> torch.Tensor:
    """Fore-aft offset between the two wheel contact points, ``|dx_left - dx_right|`` in m.

    ``phi`` (the leg-axis angle) is measured per leg, and nothing else ties the
    left and right values together.  Without this term the policy can lean one leg
    forward and the other back -- an extra yaw actuator, but physically the two
    wheels no longer sit on a common axle, so the robot scrubs sideways.

    With ``phi`` positive toward body +X, the wheel of one leg sits at
    ``x = L0 * sin(phi)`` in the body frame, so the axle skew is

        skew = |L0_l * sin(phi_l) - L0_r * sin(phi_r)|      [m]

    This is an L1 term in metres on purpose: it reads directly as "the wheels are N
    mm out of line".  The *common* part of ``phi`` is deliberately not penalised --
    both legs leaning together is the inverted-pendulum CoG shift that produces
    forward acceleration.
    """
    term = env.action_manager.get_term(action_name)
    state = torch.nan_to_num(
        term.measured_leg_state(), nan=0.0, posinf=0.0, neginf=0.0
    )  # (L0_l, phi_l, L0_r, phi_r)
    x_left = state[:, 0] * torch.sin(state[:, 1])
    x_right = state[:, 2] * torch.sin(state[:, 3])
    return (x_left - x_right).abs()


def closure_residual_l2(env, action_name: str = "joints") -> torch.Tensor:
    """Quadratic penalty on the *measured* five-bar loop error ``|W - E|``.

    ``W`` is the wheel axle and ``E`` the loop anchor on the back calf, both taken
    from the actual simulated joint angles through the verified planar model.  This
    is the quantity the passive knee targets must keep at zero; PhysX does not
    enforce the loop itself.
    """
    term = env.action_manager.get_term(action_name)
    residual = term.measured_closure_norm()
    return torch.sum(residual**2, dim=-1)


def fall_penalty(env) -> torch.Tensor:
    """1.0 for environments that ended the episode by falling (not by timeout)."""
    terminated = getattr(env, "reset_terminated", None)
    if terminated is None:
        return torch.zeros(env.num_envs, device=env.device)
    return terminated.float()


def heading_l2(
    env,
    target_heading: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared wrapped heading error, ``(N,)`` in rad^2.

    ``flat_orientation_l2`` only penalises roll and pitch; a pure yaw rotation leaves
    it unchanged.  Without this term a robot that spins on the spot scores exactly as
    well as one that stands still.
    """
    from .observations import heading_error

    err = heading_error(env, target_heading, asset_cfg)
    return err**2


def yaw_rate_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared world-frame yaw rate, ``(N,)`` in (rad/s)^2."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2] ** 2


def heading_alive(
    env,
    target_heading: float = 0.0,
    tolerance: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """1.0 while the robot is within ``tolerance`` rad of the target heading."""
    from .observations import heading_error

    err = heading_error(env, target_heading, asset_cfg)
    return (err.abs() < tolerance).float()


def position_xy_l2(
    env,
    target_x: float = 0.0,
    target_y: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared planar distance from the target, ``(N,)`` in m^2."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    dx = pos[:, 0] - target_x
    dy = pos[:, 1] - target_y
    return dx**2 + dy**2


def lin_vel_xy_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared planar base velocity in the BODY frame, ``(N,)`` in (m/s)^2."""
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b
    return v[:, 0] ** 2 + v[:, 1] ** 2


def position_xy_alive(
    env,
    target_x: float = 0.0,
    target_y: float = 0.0,
    tolerance: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """1.0 while the robot is within ``tolerance`` metres of the target, ``(N,)``."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    dx = pos[:, 0] - target_x
    dy = pos[:, 1] - target_y
    return ((dx**2 + dy**2) < tolerance**2).float()


def lin_vel_z_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared vertical velocity, ``(N,)``.

    Stops the robot from hopping/bouncing its way along.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 2] ** 2


def _euler(env, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll, pitch, yaw (rad) of the base, each ``(N,)``."""
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.euler_xyz_from_quat(asset.data.root_quat_w)


def pitch_abs(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|pitch|`` in radians, ``(N,)``.

    L1 rather than L2 on purpose: the gradient of ``x^2`` vanishes near zero, so a
    squared penalty either barely holds the angle or explodes when the robot trips.
    """
    _, pitch, _ = _euler(env, asset_cfg)
    return pitch.abs()


def roll_abs(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|roll|`` in radians, ``(N,)``.

    A two-wheel robot banks into a turn, so this is a light baseline penalty only.
    """
    roll, _, _ = _euler(env, asset_cfg)
    return roll.abs()


def ang_vel_x_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared body roll rate, ``(N,)``; damps side-to-side wobble."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 0] ** 2


def ang_vel_y_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared body pitch rate, ``(N,)``; damps pitching without penalising a lean."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 1] ** 2


def hip_lock_deviation_abs(
    env,
    action_name: str = "joints",
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Dead-band penalty on the measured hips drifting away from the reference pose.

    The hips are torque-driven, so the reference pose is not enforced by a servo;
    any measured deviation is the policy's responsibility.  A small dead band keeps
    solver noise free while still charging a real posture collapse.
    """
    term = env.action_manager.get_term(action_name)
    measured = term._robot.data.joint_pos[:, term.hip_ids] - term.nominal_hips
    deviation = (measured - term._hip_bias).abs()
    return (deviation - tolerance).clamp(min=0.0).mean(dim=-1)


__all__ = [
    "action_jerk_l2",
    "action_rate_l2",
    "ang_vel_x_l2",
    "ang_vel_y_l2",
    "base_height_l2",
    "closure_residual_l2",
    "fall_penalty",
    "flat_orientation_l2",
    "heading_alive",
    "heading_l2",
    "hip_lock_deviation_abs",
    "joint_acc_l2",
    "joint_torques_l2",
    "joint_vel_l2",
    "leg_axle_skew_abs",
    "lin_vel_xy_l2",
    "lin_vel_z_l2",
    "pitch_abs",
    "position_xy_alive",
    "position_xy_l2",
    "roll_abs",
    "wheel_spin_l2",
    "yaw_rate_l2",
]
