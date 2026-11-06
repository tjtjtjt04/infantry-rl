"""Observation terms for the chassis-rl torque squat.

The policy observation is deliberately small (25 dims) and contains only signals
that are directly measurable on the real robot:

    [base_ang_vel(3), projected_gravity(3), command(3),
     hip_pos(4), joint_vel(6), last_action(6)]

The hips are the four actuated leg joints; ``joint_vel`` covers the four hips and
both wheels; the knees are passive closed-chain followers and are not observed.
"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from robot_model import HIP_ORDER, KNEE_ORDER, WHEEL_ORDER


def _action_term(env, action_name: str):
    return env.action_manager.get_term(action_name)


def yaw_angle(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Absolute heading (yaw) of the base in the world frame, ``(N,)`` radians."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return yaw


def heading_error(
    env,
    target_heading: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Wrapped heading error ``yaw - target`` in ``(-pi, pi]``, ``(N,)``."""
    err = yaw_angle(env, asset_cfg) - target_heading
    return (err + math.pi) % (2.0 * math.pi) - math.pi


def hip_pos_rel_locked(env, action_name: str = "joints") -> torch.Tensor:
    """Measured hip angles minus the reference (locked) pose, ``(N, 4)``.

    In torque control the hips are not position-controlled, so this is the actual
    posture error the policy must null out.
    """
    term = _action_term(env, action_name)
    return term._robot.data.joint_pos[:, term.hip_ids] - term.locked_hips


def joint_vel(env, action_name: str = "joints") -> torch.Tensor:
    """Observed joint velocities, ``(N, 6)``: four hips then two wheels (rad/s).

    The action term decides whether this is PhysX's joint velocity or an
    encoder-style position difference with an optional low-pass
    (``vel_source`` / ``vel_filter_alpha``).  The term is scaled by ``0.1`` in the
    observation config, matching the deployment firmware.
    """
    return _action_term(env, action_name).measured_joint_vel()


def last_action_raw(env, action_name: str = "joints") -> torch.Tensor:
    """The raw action of the given term (``(N, 6)``: 4 hips + 2 wheels)."""
    return _action_term(env, action_name).raw_actions


def proprio_obs(env, action_name: str = "joints") -> torch.Tensor:
    """The 25-D per-frame policy observation, already scaled.

    ``[base_ang_vel(3)*0.25, projected_gravity(3), command(3),
       hip_pos_rel_locked(4), joint_vel(6)*0.05, last_action(6)]``

    This is the unit that is recorded into the 5-frame history.  Scaling happens
    here (not via ``ObsTerm.scale``) because the history term is a single tensor.
    """
    term = _action_term(env, action_name)
    robot = term._robot
    command = env.command_manager.get_command("base_velocity")
    return torch.cat(
        (
            robot.data.root_ang_vel_b * 0.25,
            robot.data.projected_gravity_b,
            command,
            robot.data.joint_pos[:, term.hip_ids] - term.locked_hips,
            term.measured_joint_vel() * 0.05,
            term.raw_actions,
        ),
        dim=-1,
    )


def history_obs(
    env,
    action_name: str = "joints",
    length: int = 5,
    noise_vec: tuple[float, ...] | None = None,
    group_name: str = "policy",
) -> torch.Tensor:
    """Flattened ``length``-frame history of :func:`proprio_obs`, ``(N, 125)``.

    The buffer lives in the action term so it is reset together with the
    environment.  Noise is applied to each *new* frame (never re-applied to stored
    frames), which is how the reference implementation records history.  It is
    enabled only when the observation group's corruption flag is set.
    """
    term = _action_term(env, action_name)
    frame = proprio_obs(env, action_name)
    enabled = bool(getattr(getattr(env.cfg.observations, group_name), "enable_corruption", True))
    return term.push_history(frame, length=length, noise_vec=noise_vec if enabled else None)


__all__ = [
    "HIP_ORDER",
    "KNEE_ORDER",
    "WHEEL_ORDER",
    "heading_error",
    "hip_pos_rel_locked",
    "history_obs",
    "joint_vel",
    "last_action_raw",
    "proprio_obs",
    "yaw_angle",
]
