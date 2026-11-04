"""Task-specific termination terms.

The shared terminations live in :mod:`isaaclab.envs.mdp`; this module only adds the
stricter posture guards used by the braking task.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def upright_and_still(
    env,
    lin_threshold: float = 0.2,
    ang_threshold: float = 0.5,
    min_upright: float = 0.9,
    hold_time: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate after being upright AND still for ``hold_time`` seconds.

    This is the recovery SUCCESS signal.  The counter lives on the recovery
    environment (``env.recovery_upright_steps``) and is cleared on reset; keeping it
    here avoids adding a mode bit to the 35-dim observation.  The task does not use
    the ordinary fall terminations, because the robot starts fallen.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright = ((1.0 - asset.data.projected_gravity_b[:, 2]) * 0.5) > min_upright
    lin_speed = torch.norm(asset.data.root_lin_vel_b, dim=-1)
    ang_speed = torch.norm(asset.data.root_ang_vel_b, dim=-1)
    ok = upright & (lin_speed < lin_threshold) & (ang_speed < ang_threshold)

    counter = getattr(env, "recovery_upright_steps", None)
    if counter is None:
        # no recovery counter (e.g. a bare test env): fall back to the instantaneous test
        return ok
    env.recovery_upright_steps = torch.where(
        ok, counter + 1, torch.zeros_like(counter)
    )
    hold_steps = max(1, int(round(hold_time / float(env.step_dt))))
    return env.recovery_upright_steps >= hold_steps


def roll_above(
    env,
    limit_angle: float = 1.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when ``|roll|`` exceeds ``limit_angle`` (radians), ``(N,)`` bool.

    The recovery task trains front/back falls only.  A large roll is a side fall,
    which is explicitly out of scope: letting those episodes continue lets PPO
    discover the stable side-lying local optimum seen in the first recovery run.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return roll.abs() > limit_angle


def unrecoverable_fall(
    env,
    side_limit: float = 1.05,
    upside_down_gz: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on a side or over-rotated fall that recovery was not trained for.

    ``roll_above`` uses Euler angles and false-triggers whenever a front/back fall
    passes pitch +/-90 deg (Euler roll flips to ~180 deg).  This guard uses projected
    gravity instead, so it has no gimbal-lock degeneracy:

    * side fall: lateral gravity ``atan2(|g_y|, -g_z) > side_limit`` (60 deg default);
    * over-rotated: ``g_z > upside_down_gz`` (body z pointing up, >120 deg tilt).

    A pure front/back fall (``g_y ~ 0``) is left to the recovery policy until it
    exceeds the 120 deg training envelope.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    lateral = torch.atan2(g[:, 1].abs(), -g[:, 2])
    return (lateral > side_limit) | (g[:, 2] > upside_down_gz)


def pitch_above(
    env,
    limit_angle: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when ``|pitch|`` exceeds ``limit_angle`` (radians), ``(N,)`` bool.

    WHY A SEPARATE LIMIT FOR BRAKING
    --------------------------------
    ``bad_orientation`` terminates on the angle between the body z-axis and vertical,
    whose limit is 1.2 rad.  A braking robot can pitch 40-60 deg while still looking
    "upright enough" for that test, drag its body on the ground, and continue collecting
    reward; the same episode then ends late or not at all.  This term ends the episode
    once the pitch alone is beyond the limit, so the fall penalty applies and PPO cannot
    keep a near-flip as a valid braking strategy.  The default 0.8 rad (~46 deg) leaves
    the useful braking lean free but rejects the 60 deg nose-up poses measured in play.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return pitch.abs() > limit_angle


__all__ = ["pitch_above", "roll_above", "unrecoverable_fall", "upright_and_still"]
