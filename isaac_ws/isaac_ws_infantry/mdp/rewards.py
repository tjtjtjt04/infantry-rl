"""Reward terms for the infantry-rl five-bar balance task.

The standard locomotion-style terms delegate to :mod:`isaaclab.envs.mdp` (they are
verified to exist at Isaac Lab 2.3.2, see ``isaac_ws/docs/API_NOTES.md`` section
6.3); the three task-specific ones -- wheel spin, fall penalty and loop-closure
error -- are implemented here.
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


def wheel_action_l2(env, action_name: str = "joints") -> torch.Tensor:
    """Small penalty on the commanded wheel velocity targets."""
    term = env.action_manager.get_term(action_name)
    return torch.sum(term.processed_actions[:, 4:] ** 2, dim=-1)


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

    WHY THIS TERM EXISTS
    --------------------
    ``flat_orientation_l2`` only penalises the x/y components of the projected
    gravity, i.e. roll and pitch.  A pure yaw rotation leaves those components
    completely unchanged, so without an explicit heading term a robot that spins on
    the spot scores exactly as well as one that stands still -- and PPO finds the
    spinning solution.  This term closes that hole by making the actual heading
    matter.
    """
    from .observations import heading_error

    err = heading_error(env, target_heading, asset_cfg)
    return err**2


def yaw_rate_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared world-frame yaw rate, ``(N,)`` in (rad/s)^2.

    Complementary to :func:`heading_l2`: the heading term removes the slow drift, the
    rate term damps the fast spin so the policy does not oscillate around the target.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2] ** 2


def position_xy_l2(
    env,
    target_x: float = 0.0,
    target_y: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared planar distance from the target, ``(N,)`` in m^2.

    This is the term that removes the steady-state drift.  On its own it would make
    the robot oscillate around the target (a pure proportional pull with no damping),
    which is why :func:`lin_vel_xy_l2` is used together with it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    dx = pos[:, 0] - target_x
    dy = pos[:, 1] - target_y
    return dx**2 + dy**2


def lin_vel_xy_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared planar base velocity in the BODY frame, ``(N,)`` in (m/s)^2.

    Body frame (not world) so it penalises actual translation regardless of which way
    the robot faces, and so it works together with the heading controller.
    """
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
    """1.0 while the robot is within ``tolerance`` metres of the target, ``(N,)``.

    A bounded bonus so staying put is actively rewarded near the target, rather than
    only being "less punished" as the quadratic term fades to zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    dx = pos[:, 0] - target_x
    dy = pos[:, 1] - target_y
    return ((dx**2 + dy**2) < tolerance**2).float()


# ---------------------------------------------------------------------- #
# velocity-command tracking (locomotion task)
# ---------------------------------------------------------------------- #
def track_lin_vel_x_exp(
    env,
    command_name: str = "base_velocity",
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-(vx - vx_cmd)^2 / std^2)``, ``(N,)`` in [0, 1].

    The exponential kernel is the standard Isaac Lab tracking reward: it is bounded,
    gives a strong gradient near the command and saturates instead of exploding when
    the robot is far off, which keeps it from dominating the balance terms.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 0]
    err = cmd - asset.data.root_lin_vel_b[:, 0]
    return torch.exp(-err**2 / std**2)


def track_ang_vel_z_exp(
    env,
    command_name: str = "base_velocity",
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-(wz - wz_cmd)^2 / std^2)``, ``(N,)`` in [0, 1].

    ``wz_cmd`` already includes the heading controller, so this single term teaches
    both "turn at the commanded rate" and "stop turning when the heading is reached".
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    err = cmd - asset.data.root_ang_vel_b[:, 2]
    return torch.exp(-err**2 / std**2)


def lin_vel_y_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared sideways velocity, ``(N,)``.

    A two-wheel differential drive has no lateral actuator, so sideways motion is
    always slip/skidding.  Penalising it keeps the robot from dragging itself
    sideways instead of turning properly.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 1] ** 2


def lin_vel_z_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared vertical velocity, ``(N,)``.

    Stops the robot from hopping/bouncing its way along: with only a height *position*
    penalty the cheapest way to move is often to bounce between two heights, which
    looks and feels terrible even if the average height is right.
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

    WHY AN L1 TERM AND NOT ``flat_orientation_l2``
    ----------------------------------------------
    ``flat_orientation_l2`` penalises the squared x/y components of the projected
    gravity, i.e. **roll and pitch together**.  For a two-wheel robot that is wrong:
    leaning sideways (roll) is exactly how it should corner, while nose-up/nose-down
    pitch is always wasted energy.  Splitting the two lets roll stay free while pitch
    is driven to zero.

    It is L1 rather than L2 on purpose: the gradient of ``x^2`` vanishes near zero and
    grows only quadratically far away, so a squared penalty either barely holds the
    angle near 0 or explodes when the robot trips.  ``|x|`` pulls with a constant
    force from any distance, which is what "keep pitch at 0" wants.
    """
    _, pitch, _ = _euler(env, asset_cfg)
    return pitch.abs()


def roll_abs(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|roll|`` in radians, ``(N,)``.

    A *light* baseline penalty; :func:`roll_when_straight_abs` adds the extra pressure
    that actually removes a permanent sideways lean.

    Why a lean appears at all: a two-wheel robot has no lateral force source, so when
    it corners gravity has to supply the centripetal component, which means banking
    into the turn.  That bank is genuinely wanted -- but it is also *small*: at
    ``v = 0.3 m/s`` and ``w = 1.5 rad/s`` the required bank is only
    ``atan(v*w/g) = 2.6 deg``.  A sustained 10-20 deg lean (what the policy actually
    learned) buys nothing, so a light global term plus a strong straight-line term is
    the right split.
    """
    roll, _, _ = _euler(env, asset_cfg)
    return roll.abs()


def roll_when_straight_abs(
    env,
    command_name: str = "base_velocity",
    wz_scale: float = 2.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|roll|`` scaled DOWN as the commanded yaw rate grows, ``(N,)``.

    This term encodes "walk upright, but you may bank while cornering".  Its value is
    multiplied by ``1 - min(|wz_cmd| / wz_scale, 1)``:

    * driving straight (``wz_cmd = 0``)   -> full penalty, the base must be level;
    * turning at the maximum trained rate  -> zero penalty, roll is free to bank;
    * rotating in place (``wz_cmd`` large, ``vx = 0``) -> also free, and no bank is
      needed anyway because there is no centripetal acceleration when ``v = 0``.

    Gating on the *commanded* yaw rate rather than on the measured one is deliberate:
    gating on the measurement would let the robot escape the penalty by leaning, which
    is exactly the behaviour being removed.
    """
    roll, _, _ = _euler(env, asset_cfg)
    term = env.command_manager.get_term(command_name)
    turn = (term.command[:, 2].abs() / max(wz_scale, 1e-6)).clamp(0.0, 1.0)
    return roll.abs() * (1.0 - turn)


def ang_vel_x_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared body roll rate, ``(N,)``.

    Damps side-to-side wobble without penalising a *static* bank (a steady bank has
    zero rate), so it does not fight cornering the way a plain roll-angle term would.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 0] ** 2


def base_height_abs(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|root_z - target_height|`` in metres, ``(N,)``.

    L1 again, for the same reason as :func:`pitch_abs`: measured behaviour with the
    squared version (weight -10) was a robot that walked fine but **crouched about
    5 cm too low**, because at that offset the squared penalty contributes ~80x less
    per step than the velocity-tracking reward and never wins the trade.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height - target_height).abs()


def heading_error_l2(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared heading error relative to the *command manager's* heading target.

    This is the inner-loop error of the cascaded controller: ``wz_cmd`` is derived
    from it, so penalising it as well tells the policy "do not merely match the yaw
    rate, actually arrive at the target heading".
    """
    term = env.command_manager.get_term(command_name)
    if not getattr(term.cfg, "heading_command", False):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    err = math_utils.wrap_to_pi(term.heading_target - asset.data.heading_w)
    return err**2


def heading_alive(
    env,
    target_heading: float = 0.0,
    tolerance: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """1.0 while the robot is within ``tolerance`` rad of the target heading.

    A bounded bonus (in contrast to the quadratic penalty) gives a clear "keep doing
    this" signal near the target instead of a vanishingly small gradient.
    """
    from .observations import heading_error

    err = heading_error(env, target_heading, asset_cfg)
    return (err.abs() < tolerance).float()


def base_height_below(env, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty (as a reward term) for being below a minimum root height."""
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height < minimum_height).float()


__all__ = [
    "action_rate_l2",
    "ang_vel_x_l2",
    "base_height_abs",
    "base_height_below",
    "base_height_l2",
    "closure_residual_l2",
    "fall_penalty",
    "flat_orientation_l2",
    "heading_alive",
    "heading_error_l2",
    "heading_l2",
    "joint_acc_l2",
    "joint_torques_l2",
    "joint_vel_l2",
    "lin_vel_xy_l2",
    "lin_vel_y_l2",
    "lin_vel_z_l2",
    "pitch_abs",
    "position_xy_alive",
    "position_xy_l2",
    "roll_abs",
    "roll_when_straight_abs",
    "track_ang_vel_z_exp",
    "track_lin_vel_x_exp",
    "wheel_action_l2",
    "wheel_spin_l2",
    "yaw_rate_l2",
]
