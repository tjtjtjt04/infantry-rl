"""Observation terms for the infantry-rl balance task.

Everything is expressed in the action-vector order defined in
:mod:`isaac_ws_infantry.model`, so the policy sees the same layout it commands.
"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ..model import HIP_ORDER, KNEE_ORDER, WHEEL_ORDER


def _action_term(env, action_name: str):
    return env.action_manager.get_term(action_name)


def yaw_angle(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Absolute heading (yaw) of the base in the world frame, ``(N,)`` radians.

    ``+Z`` is up, so yaw is the rotation the robot would use to turn on the spot.
    At reset all environments spawn with an identity orientation, so the *target*
    heading is ``target_heading`` (0 by default).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return yaw


def heading_error(
    env,
    target_heading: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Wrapped heading error ``yaw - target`` in ``(-pi, pi]``, ``(N,)``.

    Wrapping matters: the raw quaternion-derived yaw jumps across the +/-pi seam, and
    a term that is smooth in the error lets the policy turn the short way round.
    """
    err = yaw_angle(env, asset_cfg) - target_heading
    return (err + math.pi) % (2.0 * math.pi) - math.pi


def heading_sin_cos(
    env,
    target_heading: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Heading as ``(sin(err), cos(err))``, ``(N, 2)``.

    Returned as sin/cos rather than a raw angle so the observation stays continuous
    everywhere (a raw wrapped angle has a discontinuity at +/-pi exactly when the
    robot is facing backwards, which is also the worst place to confuse a policy).
    The target heading corresponds to ``(0, 1)``.
    """
    err = heading_error(env, target_heading, asset_cfg)
    return torch.stack((torch.sin(err), torch.cos(err)), dim=-1)


def position_error(
    env,
    target_x: float = 0.0,
    target_y: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Planar position error relative to ``(target_x, target_y)``, ``(N, 2)``.

    The target is expressed in the ENVIRONMENT frame (``env_origins`` is the spawn
    point), so ``(0, 0)`` means "stay where you spawned".  The error is rotated into
    the base yaw frame, so the two components mean *forward* and *left* -- which is
    what the policy needs in order to cancel a drift with the wheels.

    WHY THIS EXISTS
    ---------------
    Same story as the heading term: nothing in the original reward mentioned the
    planar position or the forward/backward velocity, so the robot was free to run
    away in a straight line while balancing perfectly.  The policy indeed learned to
    drift backwards.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    ex = target_x - pos[:, 0]
    ey = target_y - pos[:, 1]
    yaw = yaw_angle(env, asset_cfg)
    c, s = torch.cos(yaw), torch.sin(yaw)
    # rotate the world-frame error into the yaw frame: forward / left components
    fwd = c * ex + s * ey
    left = -s * ex + c * ey
    return torch.stack((fwd, left), dim=-1)


def hip_pos_rel_nominal(env, action_name: str = "joints") -> torch.Tensor:
    """Measured hip angles minus the nominal stance, ``(N, 4)`` in :data:`HIP_ORDER`."""
    term = _action_term(env, action_name)
    return term._robot.data.joint_pos[:, term.hip_ids] - term.nominal_hips


def knee_pos_rel_solved(env, action_name: str = "joints") -> torch.Tensor:
    """Measured knee angles minus the *solved* knee targets, ``(N, 4)``.

    This is the closure tracking error: it is small exactly when the passive knees
    have reached the configuration that closes the five-bar loop.
    """
    term = _action_term(env, action_name)
    return term._robot.data.joint_pos[:, term.knee_ids] - term.knee_targets


def wheel_vel(env, asset_cfg: SceneEntityCfg | None = None, action_name: str = "joints") -> torch.Tensor:
    """Wheel joint velocities, ``(N, 2)`` in :data:`WHEEL_ORDER`."""
    term = _action_term(env, action_name)
    return term._robot.data.joint_vel[:, term.wheel_ids]


def last_action_raw(env, action_name: str = "joints") -> torch.Tensor:
    """The raw action of the given term (``(N, 6)``: 4 hips + 2 wheels)."""
    return _action_term(env, action_name).raw_actions


def solved_leg_state(env, action_name: str = "joints") -> torch.Tensor:
    """Solved ``(L0, phi)`` of both legs, ``(N, 4)``: ``(L0_l, phi_l, L0_r, phi_r)``."""
    return _action_term(env, action_name).solved_leg_state()


def measured_leg_state(env, action_name: str = "joints") -> torch.Tensor:
    """Measured ``(L0, phi)`` of both legs, ``(N, 4)``."""
    return _action_term(env, action_name).measured_leg_state()


def leg_rates(env, action_name: str = "joints") -> torch.Tensor:
    """Leg-length and leg-angle rates, ``(N, 4)``: ``(dL0_l, dphi_l, dL0_r, dphi_r)``.

    Units are m/s and rad/s.  These are the ``dL0`` / ``dphi`` states of the firmware
    LQR (``x = [theta, dtheta, xb, dxb, phi, dphi]``): the pose alone only tells the
    policy *where* the leg is, not how fast it is moving, so without these dims the
    leg attitude loop can only use proportional feedback and lags one derivative
    behind.  Estimated by one-step finite difference inside the action term, and
    reported as zero for the step right after a reset.
    """
    return _action_term(env, action_name).update_leg_rates()


def closure_residual(env, action_name: str = "joints") -> torch.Tensor:
    """Measured ``|W - E|`` per side, ``(N, 2)`` in metres."""
    return _action_term(env, action_name).measured_closure_norm()


def base_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height above the environment origin, ``(N, 1)``."""
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def velocity_commands(env, command_name: str = "base_velocity") -> torch.Tensor:
    """The active velocity command as ``(vx, wz)``, ``(N, 2)``.

    The command tensor itself is ``(N, 3)`` = ``(vx, vy, wz)``, but this robot is a
    two-wheel differential drive: it cannot translate sideways, so ``vy`` is always
    forced to 0 and is not fed to the policy (a constant dim is wasted input).

    ``wz`` here is *already* the output of the heading controller: the command manager
    runs ``wz = stiffness * wrap(heading_target - heading)`` every control step, so
    "hold the current heading" and "turn to a new heading" are the same mechanism.
    """
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, [0, 2]]


__all__ = [
    "HIP_ORDER",
    "KNEE_ORDER",
    "WHEEL_ORDER",
    "base_height",
    "closure_residual",
    "heading_error",
    "heading_sin_cos",
    "hip_pos_rel_nominal",
    "knee_pos_rel_solved",
    "last_action_raw",
    "leg_rates",
    "measured_leg_state",
    "position_error",
    "solved_leg_state",
    "velocity_commands",
    "wheel_vel",
    "yaw_angle",
]
