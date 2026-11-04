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

    ``exclude_wheels=True`` restricts the penalty to the four hip channels.  A braking
    task needs the wheel command to change abruptly (that IS the brake), and charging
    that change as "jerk" would directly fight the behaviour being trained; the leg
    tremor this term exists for lives in the hip channels.

    Cache: ``FiveBarJointAction.process_actions`` keeps ``a[t-1]`` and ``a[t-2]``, and
    ``FiveBarJointAction.action_jerk()`` produces the second difference.
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


def wheel_torques_excess_l2(
    env, threshold: float = 1.0, action_name: str = "joints"
) -> torch.Tensor:
    """``sum(max(|tau_wheel| - threshold, 0) ** 2)`` over the two wheel joints, ``(N,)``.

    WHY A DEADBAND AND NOT A PLAIN L2
    ---------------------------------
    This term exists to stop the policy from using the WHEEL reaction torque as its
    attitude actuator (Segway-style balancing) instead of the hips.  But the wheels are
    also the only longitudinal actuator, so a plain ``sum(tau^2)`` charges legitimate
    driving: holding 2 m/s needs almost no torque, yet accelerating 0 -> 2 m/s at
    2 m/s^2 needs about 0.9 Nm per wheel, and a command step from +2 to -2 m/s reaches
    roughly 1.7 Nm.

    The behaviour that must be removed is the *spike*: wheel-torque balancing drives the
    wheels to 2-3 Nm in fast oscillations.  With ``threshold = 1.0`` Nm the smooth
    propulsive band is free and only the spikes are charged, so the leg-dominance
    pressure survives widening the speed range.
    """
    term = env.action_manager.get_term(action_name)
    torque = term._robot.data.applied_torque[:, term.wheel_ids].abs()
    excess = (torque - threshold).clamp(min=0.0)
    return torch.sum(excess**2, dim=-1)


def wheel_action_l2(env, action_name: str = "joints") -> torch.Tensor:
    """Small penalty on the commanded wheel velocity targets."""
    term = env.action_manager.get_term(action_name)
    return torch.sum(term.processed_actions[:, 4:] ** 2, dim=-1)


def leg_axle_skew_abs(env, action_name: str = "joints") -> torch.Tensor:
    """Fore-aft offset between the two wheel contact points, ``|dx_left - dx_right|`` in m.

    WHY THIS TERM EXISTS
    --------------------
    ``phi`` (the leg-axis angle) is measured per leg, and nothing in the reward set tied
    the left and right values together.  The policy discovered that leaning one leg
    forward while leaning the other one back gives it an extra yaw actuator -- but the
    physical result is that the two wheels no longer sit on a common axle: one is ahead
    of the other, the axle is skewed, and the robot scrubs sideways instead of driving
    straight.  Visually it looks like "the robot moves forward with its wheels one in
    front of the other", which is never correct for a differential drive.

    With ``phi`` positive toward body +X, the wheel of one leg sits at
    ``x = L0 * sin(phi)`` in the body frame, so the axle skew is

        skew = |L0_l * sin(phi_l) - L0_r * sin(phi_r)|      [m]

    This is an L1 term in metres on purpose: it reads directly as "the wheels are N mm
    out of line", which is the quantity you can measure in a replay.

    Note what is deliberately NOT penalised: the *common* part of ``phi``.  Both legs
    leaning together is exactly the inverted-pendulum centre-of-gravity shift that
    produces forward acceleration, so it must stay free.  Only the left-right
    difference is constrained -- which also matches the reference firmware (``VMC`` +
    LQR), where both leg angles are driven to the same reference.

    Measured leg state is used (not the commanded one), and the result is sanitised:
    the planar solve can return NaN in unreachable poses.
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


def track_lin_vel_x_l2(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``(vx - vx_cmd)^2``, ``(N,)``.

    WHY THIS IS NEEDED ON TOP OF THE EXP KERNEL
    -------------------------------------------
    ``exp(-e^2/std^2)`` has ZERO gradient at ``e = 0`` (the derivative is
    ``-2 e / std^2 * exp(...)``, which vanishes with e).  That makes the tracking reward
    flat right at the command, so small deviations -- a few percent of the commanded
    speed, oscillating at the control rate -- cost almost nothing.  That is exactly the
    "stop-and-go" ripple that a bounded exp kernel cannot remove: the policy has no
    restoring force towards a *steady* speed, only towards "roughly the right speed".

    A quadratic term adds the missing local stiffness: its gradient is ``-2 e``, i.e.
    proportional to the error for ANY error, so it pulls hardest exactly where the exp
    kernel is blind.  The two together give "get close fast" (exp) and "stay close"
    (L2).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err = env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0]
    return err**2


def track_ang_vel_z_l2(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``(wz - wz_cmd)^2``, ``(N,)``.

    The yaw-rate counterpart of :func:`track_lin_vel_x_l2`, for the same reason: the exp
    kernel is blind at zero error, so residual yaw oscillation (a two-wheel robot is
    constantly making small steering corrections) is nearly free.  With a quadratic
    term any yaw rate that is not the commanded one costs something, which is what
    removes the visible wobble.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err = env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
    return err**2


def track_lin_vel_x_l1(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|vx - vx_cmd|``, ``(N,)``.

    WHY AN L1 TERM ON TOP OF L2 (v1.0.8)
    ------------------------------------
    L2's restoring force is ``-2*w*e``: it grows with the error, which is good for large
    errors but *weak exactly where precision is wanted*.  To hold 2 cm/s the L2 gradient
    is only ``2*6*0.02 = 0.24`` -- comparable to the noise in the other penalty terms, so
    it cannot pin the speed down.  L1 has a CONSTANT gradient ``-w`` at every error, so
    once the policy is within a few cm/s it still feels the same full pull towards the
    command.  That is what turns "roughly the commanded speed" into "the commanded speed".
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err = env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0]
    return err.abs()


def track_ang_vel_z_l1(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|wz - wz_cmd|``, ``(N,)`` -- the yaw-rate counterpart of the term above."""
    asset: Articulation = env.scene[asset_cfg.name]
    err = env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
    return err.abs()


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


def pitch_excess_abs(
    env,
    tolerance: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``max(|pitch| - tolerance, 0)`` in radians, ``(N,)``.

    A dead-band pitch penalty.  A braking robot legitimately leans a few degrees, so
    penalising every radian would fight the manoeuvre; the tolerance (default ~5.7 deg)
    leaves that free.  Beyond it, the L1 excess gives a constant pull back towards
    level, which is what stops the body from dragging on the ground during a hard stop.
    It is symmetric, so it also catches an unexpected nose-down dive.
    """
    _, pitch, _ = _euler(env, asset_cfg)
    return (pitch.abs() - tolerance).clamp(min=0.0)


def ang_vel_y_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared body pitch rate, ``(N,)``.

    Damps the pitching oscillation during braking without penalising a static lean
    (a steady pitch has zero rate).  The roll-rate counterpart is :func:`ang_vel_x_l2`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 1] ** 2


def hip_deviation_abs(
    env,
    action_name: str = "joints",
    tolerance: float = 0.1,
) -> torch.Tensor:
    """``mean(max(|hip - nominal_hip| - tolerance, 0))`` over the four hips, ``(N,)``.

    WHY THIS EXISTS (braking)
    -------------------------
    The hips are the only way to shift the body's centre of gravity fore/aft, so a
    braking policy legitimately uses them.  In practice it used almost the whole
    ``hip_range`` (mean 18 deg, max 25 deg against a 25.8 deg range) and lifted a
    wheel/leg off the ground, which is what produced the nose-up fall.  A dead band
    keeps normal attitude work free while charging only the excessive, saturating
    poses; it does not forbid hip use.
    """
    term = env.action_manager.get_term(action_name)
    deviation = (term._robot.data.joint_pos[:, term.hip_ids] - term.nominal_hips).abs()
    return (deviation - tolerance).clamp(min=0.0).mean(dim=-1)


def hip_lock_deviation_abs(
    env,
    action_name: str = "joints",
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Dead-band penalty on the measured hips drifting away from the locked pose.

    Only meaningful for an action term with ``lock_hips=True`` (the squat task): the
    four hips are held at ``nominal + hip_bias`` by the position controller, so any
    measured deviation is compliance/load, not a policy choice.  A small dead band
    keeps solver noise free while still charging a real posture collapse.
    """
    term = env.action_manager.get_term(action_name)
    measured = term._robot.data.joint_pos[:, term.hip_ids] - term.nominal_hips
    deviation = (measured - term._hip_bias).abs()
    return (deviation - tolerance).clamp(min=0.0).mean(dim=-1)


def hip_deviation_stance_abs(
    env,
    action_name: str = "joints",
    tolerance: float = 0.10,
) -> torch.Tensor:
    """Like :func:`hip_deviation_abs`, but measured against the intended stance.

    The scripted stance bias (A + RB leg extension) moves the hips by
    ``[+d, -d, +d, -d]``.  Charging that deviation as "excessive hip use" would
    fight the extension, so subtract the intended offsets first.  The term still
    charges asymmetric bracing and policy actions beyond the dead band, which is
    what it was introduced for in the braking task.
    """
    term = env.action_manager.get_term(action_name)
    measured = term._robot.data.joint_pos[:, term.hip_ids] - term.nominal_hips
    deviation = (measured - term.intended_hip_offsets).abs()
    return (deviation - tolerance).clamp(min=0.0).mean(dim=-1)


def stance_hold_abs(
    env,
    action_name: str = "joints",
    tolerance: float = 0.10,
    min_stance: float = 0.05,
) -> torch.Tensor:
    """Dead-band penalty on the policy cancelling the scripted stance.

    Active only while the stance bias is non-trivial.  The inherited
    ``hip_deviation`` dead band is wide (0.22 rad in the teleop checkpoint), so
    without this term the policy could quietly undo most of a +0.30 extension by
    pushing the hips the other way.  The dead band leaves normal balance work and
    small asymmetric corrections free.
    """
    term = env.action_manager.get_term(action_name)
    measured = term._robot.data.joint_pos[:, term.hip_ids] - term.nominal_hips
    err = (measured - term.stance_offsets).abs()
    active = term.stance.abs() > float(min_stance)
    return active.float() * (err - float(tolerance)).clamp(min=0.0).mean(dim=-1)


def base_height_stance_abs(
    env,
    target_height: float,
    stance_gain: float = 0.195,
    stance_gain2: float = 0.0,
    action_name: str = "joints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|root_z - (target_height + stance_gain * d + stance_gain2 * d^2)|``, ``(N,)``.

    The measured leg kinematics are curved at deep crouch (the five-bar folds):
    ``base_z ≈ 0.1888 + 0.1901 d + 0.0173 d^2`` fits the full [-0.6, +0.45]
    range to ~3 mm.  A pure linear gain overestimates the drop at d=-0.55/-0.6
    by ~8-11 mm, which would pull the height reward below the teleop fall
    threshold.  The quadratic term (default 0) keeps every other task unchanged.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    term = env.action_manager.get_term(action_name)
    d = term.stance
    target = float(target_height) + float(stance_gain) * d + float(stance_gain2) * d * d
    return (height - target).abs()


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


def heading_error_l1(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``|heading error|`` relative to the command manager's heading target, ``(N,)`` rad.

    WHY AN L1 TERM ON TOP OF ``heading_error_l2``
    ---------------------------------------------
    ``heading_error_l2`` is squared, so its gradient ``2 * e`` vanishes exactly where
    the requirement is strict: a 2 deg (0.035 rad) yaw error with weight -0.6 has a
    penalty of 7e-4 -- far below the noise of the velocity terms.  The policy is
    therefore free to cruise with a few degrees of yaw offset, which at 3 m/s means
    sideways travel of ~0.15 m/s.  ``|e|`` has a constant gradient, so it keeps
    pulling until the heading error is actually small, from any distance.

    Only active in the heading-command task; zero otherwise (same guard as
    :func:`heading_error_l2`).
    """
    term = env.command_manager.get_term(command_name)
    if not getattr(term.cfg, "heading_command", False):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    err = math_utils.wrap_to_pi(term.heading_target - asset.data.heading_w)
    return err.abs()


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


# ---------------------------------------------------------------------- #
# fall recovery
# ---------------------------------------------------------------------- #
def uprightness(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``(1 - projected_gravity_z) / 2``: 1 upright, 0 upside down, 0.5 at 90 deg.

    Unlike :func:`flat_orientation_l2`, this distinguishes upright from upside down:
    the projected gravity z is -1 upright and +1 on the back, while its x/y
    components are zero in both cases.  It is the main recovery reward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return (1.0 - asset.data.projected_gravity_b[:, 2]) * 0.5


def upright_still(
    env,
    lin_threshold: float = 0.2,
    ang_threshold: float = 0.5,
    min_upright: float = 0.9,
    min_height: float = 0.0,
    hold_time: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """1.0 while upright, tall enough and nearly still, else 0, ``(N,)``.

    The upright reward alone pays for passing through the upright pose; this bonus
    pays for STAYING there.  ``min_height`` is essential: without it the policy can
    count a deep squat (upright orientation, root z~0.10 m, hips still tucked) as
    success and never execute the final leg extension.  It also maintains the hold
    counter / per-episode success flag used by the logging and one-shot bonus.
    Recovery does NOT terminate early on success.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright = uprightness(env, asset_cfg) > min_upright
    if min_height > 0.0:
        upright = upright & (_root_height(env, asset_cfg) > min_height)
    lin_speed = torch.norm(asset.data.root_lin_vel_b, dim=-1)
    ang_speed = torch.norm(asset.data.root_ang_vel_b, dim=-1)
    ok = upright & (lin_speed < lin_threshold) & (ang_speed < ang_threshold)

    counter = getattr(env, "recovery_upright_steps", None)
    if counter is not None:
        env.recovery_upright_steps = torch.where(ok, counter + 1, torch.zeros_like(counter))
        hold_steps = max(1, int(round(hold_time / float(env.step_dt))))
        success = getattr(env, "recovery_success", None)
        if success is not None:
            success |= env.recovery_upright_steps >= hold_steps
    return ok.float()


def uprightness_power(
    env,
    power: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``uprightness ** power``: a much sharper pull towards fully upright.

    The linear uprightness is already 0.85 at 46 deg, so a policy can sit at a
    comfortable lean and collect most of the reward without ever standing up.  The
    cubic (or higher) version makes the last 20-30 deg worth substantially more.
    """
    return uprightness(env, asset_cfg) ** float(power)


def _root_height(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def upright_zone(
    env,
    min_upright: float = 0.97,
    min_height: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """1.0 when clearly upright (and tall enough), else 0, ``(N,)``.

    A bounded near-upright bonus that does not require standing still; the stillness
    requirement is handled by :func:`upright_still`.  ``min_height`` prevents the
    deep-squat false positive (upright orientation but still tucked at z~0.10 m).
    """
    ok = uprightness(env, asset_cfg) > min_upright
    if min_height > 0.0:
        ok = ok & (_root_height(env, asset_cfg) > min_height)
    return ok.float()


def leg_extension_abs(
    env,
    action_name: str = "joints",
    upright_min: float = 0.90,
    max_height: float = 0.16,
) -> torch.Tensor:
    """Mean ``|hip - nominal|`` while upright but still too low, ``(N,)``.

    This is the dense gradient for the last phase of the user's strategy: after the
    robot has tuck-rolled itself upright it often stops in a deep squat (root z~0.10)
    because the success/height rewards only pay at the very end.  While the base is
    upright but below ``max_height``, this term pulls all four hips from the tuck
    (-0.45/+0.45) back to the nominal stance.  It is gated off once the robot is tall
    enough, so it does not fight the initial tuck.
    """
    term = env.action_manager.get_term(action_name)
    hip = (term._robot.data.joint_pos[:, term.hip_ids] - term.nominal_hips).abs().mean(dim=-1)
    upright = uprightness(env) > upright_min
    low = _root_height(env, SceneEntityCfg("robot")) < max_height
    return (upright & low).float() * hip


def recovered_bonus(
    env,
    hold_time: float = 0.5,
) -> torch.Tensor:
    """One-step +1 when the recovery hold counter FIRST reaches the success length.

    The per-step upright/still rewards alone create a strong local optimum at
    ``uprightness ~= 0.90``: the robot can lean there, collect most of the upright
    reward and never cross the success threshold.  This one-shot bonus (weight 50)
    makes actually finishing the stand-up much more valuable than hovering near it.
    The counter is updated by :func:`mdp.terminations.upright_and_still`; the reward
    manager runs after the termination manager, so the value is visible here.
    """
    counter = getattr(env, "recovery_upright_steps", None)
    if counter is None:
        return torch.zeros(env.num_envs, device=env.device)
    hold_steps = max(1, int(round(hold_time / float(env.step_dt))))
    return (counter == hold_steps).float()


def roll_fail_penalty(
    env,
    limit_angle: float = 1.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-step penalty on episodes that END by exceeding the roll limit.

    Without it, a side-roll termination and a timeout are nearly equivalent (both
    lose the remaining upright reward), so PPO has little reason to avoid rolling.
    """
    terminated = getattr(env, "reset_terminated", None)
    if terminated is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return (terminated & (roll.abs() > limit_angle)).float()


def tuck_match(
    env,
    action_name: str = "joints",
    front_tuck: float = -0.45,
    back_tuck: float = 0.45,
    upright_max: float = 0.90,
    sigma: float = 0.15,
) -> torch.Tensor:
    """Bounded reward for the deepest symmetric squat while still fallen.

    The measured five-bar pose with hip offsets ``front=-0.45, back=+0.45`` has
    ``L0 = 0.165 m`` (nominal 0.250 m) and the wheel axle directly under the hip
    midpoint.  This is the "tuck the legs, drive the wheels, then extend" recovery
    strategy the user asked for.  The gate switches the term off once the robot is
    close to upright, so it cannot become a reward for simply staying squatting.
    """
    term = env.action_manager.get_term(action_name)
    hip_offset = term._robot.data.joint_pos[:, term.hip_ids] - term.nominal_hips
    target = hip_offset.new_tensor([front_tuck, back_tuck, front_tuck, back_tuck])
    err = ((hip_offset - target) ** 2).mean(dim=-1)
    upright = (1.0 - term._robot.data.projected_gravity_b[:, 2]) * 0.5
    gate = (upright < upright_max).float()
    return gate * torch.exp(-err / max(sigma * sigma, 1e-9))


# ---------------------------------------------------------------------- #
# braking task
# ---------------------------------------------------------------------- #
def _brake_term(env, command_name: str):
    """The :class:`~mdp.commands.BrakingVelocityCommand` term that owns the schedule."""
    return env.command_manager.get_term(command_name)


def brake_distance(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Planar distance travelled since the brake command started, ``(N,)`` in metres.

    Zero before the brake.  This is the quantity the user cares about ("how far does
    it slide before it stops"), and it deliberately does NOT distinguish rolling from
    sliding: a short skid is allowed if it stops the robot sooner.
    """
    term = _brake_term(env, command_name)
    asset: Articulation = env.scene[term.cfg.asset_name]
    delta = asset.data.root_pos_w[:, :2] - term.brake_position
    distance = torch.norm(delta, dim=-1)
    return torch.where(term.brake_started, distance, torch.zeros_like(distance))


def brake_distance_exp(
    env, command_name: str = "base_velocity", scale: float = 0.5
) -> torch.Tensor:
    """``exp(-distance / scale)`` after the brake, ``(N,)`` in [0, 1]; 0 before it.

    A bounded "shorter is better" signal.  With ``scale=0.5`` a 0.5 m stop scores
    e^-1 = 0.37 and a 1.0 m stop scores 0.14, while a 0.2 m stop scores 0.67.
    """
    d = brake_distance(env, command_name)
    term = _brake_term(env, command_name)
    return torch.where(term.brake_started, torch.exp(-d / max(scale, 1e-6)), torch.zeros_like(d))


def brake_distance_l1(env, command_name: str = "base_velocity") -> torch.Tensor:
    """``distance`` after the brake, ``(N,)`` in metres; 0 before it.

    The exp kernel is flat near zero error; this L1 term keeps a constant pull towards
    a shorter stop at every distance.
    """
    return brake_distance(env, command_name)


def brake_overshoot(
    env, command_name: str = "base_velocity", tolerance: float = 1.0
) -> torch.Tensor:
    """``max(distance - tolerance, 0)`` in metres, ``(N,)``.

    Encodes "a small slide is allowed": up to ``tolerance`` metres is free, and only
    the excess is charged.  With the measured 3.69 Nm / 12.8 kg drivetrain the
    physical minimum from 3 m/s is about 0.53 m, so a tolerance around 1.0 m is a
    realistic "short stop with some skid" target.
    """
    d = brake_distance(env, command_name)
    term = _brake_term(env, command_name)
    excess = (d - tolerance).clamp(min=0.0)
    return torch.where(term.brake_started, excess, torch.zeros_like(excess))


def brake_standstill(
    env,
    command_name: str = "base_velocity",
    speed_threshold: float = 0.1,
    yaw_threshold: float = 0.3,
) -> torch.Tensor:
    """1.0 once the robot is stopped after braking, ``(N,)``.

    Requires the brake phase to have started, a small body forward speed and a small
    yaw rate.  This gives an explicit "stay balanced at zero" bonus instead of relying
    only on the vanishing velocity-tracking error.
    """
    term = _brake_term(env, command_name)
    asset: Articulation = env.scene[term.cfg.asset_name]
    speed = asset.data.root_lin_vel_b[:, 0].abs()
    yaw_rate = asset.data.root_ang_vel_b[:, 2].abs()
    stopped = (
        term.brake_started
        & (speed < speed_threshold)
        & (yaw_rate < yaw_threshold)
    )
    return stopped.float()


__all__ = [
    "action_jerk_l2",
    "action_rate_l2",
    "ang_vel_x_l2",
    "base_height_abs",
    "base_height_below",
    "base_height_l2",
    "base_height_stance_abs",
    "brake_distance",
    "brake_distance_exp",
    "brake_distance_l1",
    "brake_overshoot",
    "brake_standstill",
    "closure_residual_l2",
    "fall_penalty",
    "flat_orientation_l2",
    "heading_alive",
    "heading_error_l1",
    "heading_error_l2",
    "heading_l2",
    "joint_acc_l2",
    "joint_torques_l2",
    "joint_vel_l2",
    "leg_axle_skew_abs",
    "lin_vel_xy_l2",
    "lin_vel_y_l2",
    "lin_vel_z_l2",
    "ang_vel_y_l2",
    "hip_deviation_abs",
    "hip_deviation_stance_abs",
    "hip_lock_deviation_abs",
    "stance_hold_abs",
    "pitch_abs",
    "pitch_excess_abs",
    "position_xy_alive",
    "position_xy_l2",
    "roll_abs",
    "roll_when_straight_abs",
    "track_ang_vel_z_exp",
    "track_ang_vel_z_l2",
    "track_lin_vel_x_exp",
    "track_lin_vel_x_l2",
    "tuck_match",
    "leg_extension_abs",
    "upright_still",
    "upright_zone",
    "uprightness",
    "uprightness_power",
    "recovered_bonus",
    "roll_fail_penalty",
    "wheel_action_l2",
    "wheel_torques_excess_l2",
    "wheel_spin_l2",
    "yaw_rate_l2",
]
