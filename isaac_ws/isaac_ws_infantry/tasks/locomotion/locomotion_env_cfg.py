"""Locomotion (velocity-command tracking) task for the infantry-rl five-bar robot.

This task turns the *balance* environment into a drivable one.  The difference is
where the target comes from:

* balance task -- a fixed target (heading 0, position 0): "stand there and do not
  move";
* this task -- a **velocity command** ``(vx, wz)`` produced by a command term: "drive
  forward at vx and turn at wz".

Everything else (robot, loop-closure action term, PhysX side) is shared with the
balance task; only the observation layout and the reward set differ.

Why ``(vx, wz)`` and not ``(vx, vy, wz)``
----------------------------------------
The robot is a two-wheel differential drive: forward/backward comes from the mean
wheel speed and turning from the difference.  There is **no lateral actuator**, so a
``vy`` command could never be tracked; it is pinned to 0 and dropped from the
observation instead of feeding the policy a constant dimension.

Why ``heading_command=True``
----------------------------
With heading commands the manager converts the heading error into the yaw-rate
command every control step::

    wz_cmd = clip(heading_control_stiffness * wrap(heading_target - heading))

So "hold the current direction" and "turn to a new direction" become the same
mechanism, and the policy only ever has to track a yaw rate.  This is what makes
in-place rotation (``vx = 0, heading_target moving``) and turning while driving
(``vx != 0``) the same skill.
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaac_ws_infantry.mdp as mdp
from isaac_ws_infantry.robots import INFANTRY_RL_CFG
from isaac_ws_infantry.tasks.common.robot_env_cfg import (
    FALL_ANGLE,
    FALL_HEIGHT,
    NOMINAL_HEIGHT,
    ActionsCfg,
    RobotSceneCfg,
    TerminationsCfg,
)

# Forward speed range the policy is trained on (m/s).  Both bounds are configurable so
# the envelope can be staged without touching code.
#
# v1.0.6: default is [0.2, 1.2] -- a FORWARD-ONLY band, as requested.  Consequences to be
# aware of:
#   * the robot never sees a reverse command, so it will not learn to back up;
#   * it never sees a near-zero *moving* command either -- stopping is still trained,
#     because `rel_standing_envs = 0.1` zeroes the whole command for 10% of the envs
#     (that is the braking sample).
# To get reverse back, set INFANTRY_RL_VX_MIN to a negative number.
#
# What the hardware can actually do:
#   wheel velocity limit 61.471 rad/s x 0.068 m      -> 4.18 m/s
#   wheel effort 2 x 3.69 Nm / 0.068 m = 108 N       -> 8.5 m/s^2 of thrust
#   total mass 12.776 kg, so ~8.5 m/s^2 available vs the ~7.8 m/s^2 a tyre can grip
# 1.2 m/s therefore sits at only ~29% of the wheel limit (17.6 rad/s), which leaves a
# large margin for slip, slope and the 3.69 Nm effort cap.
VX_MIN_MPS = float(os.environ.get("INFANTRY_RL_VX_MIN", "0.2"))
VX_MAX_MPS = float(os.environ.get("INFANTRY_RL_VX_MAX", "1.2"))

#: forward speed range the policy is trained on (m/s)
LIN_VEL_X_RANGE = (VX_MIN_MPS, VX_MAX_MPS)

#: TRAIN TURNING OR NOT.
#:
#: Off (default) is the "straight line only" task: the heading target is pinned to 0, so
#: the robot is never asked to turn to a new direction.  It still *holds* its heading,
#: because the heading controller keeps running --
#:
#:     wz* = clip(1.5 * wrap(0 - psi), +/-2.5)
#:
#: i.e. any yaw drift still produces a correcting command, which is exactly what was
#: asked for ("going forward must not accumulate a heading offset / yaw rate").
#:
#: Turning it back on restores random heading targets in [-pi, pi].  Doing so is a
#: ONE-LINE switch rather than a code change, because the difference is entirely in the
#: command sampling range -- no observation, reward or network change is involved.
_ENABLE_TURNING = os.environ.get("INFANTRY_RL_ENABLE_TURNING", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
#: heading target sampling range.  Fixed at 0 unless INFANTRY_RL_ENABLE_TURNING=1.
HEADING_RANGE = (-3.141592653589793, 3.141592653589793) if _ENABLE_TURNING else (0.0, 0.0)

#: yaw-rate range -- also the CLAMP on the heading controller output (rad/s).
#: It is NOT a sampling range when heading_command=True.  Kept at +/-2.5 even with
#: turning disabled: it sets how hard the robot may correct a heading drift.
ANG_VEL_Z_RANGE = (-2.5, 2.5)
#: how strongly a heading error is turned into a yaw-rate command (1/s). 1.5 means a
#: 1 rad heading error immediately asks for the maximum correction rate.
HEADING_STIFFNESS = 1.5
#: fraction of environments commanded to stand still (teaches braking)
REL_STANDING_ENVS = 0.1


@configclass
class LocomotionCommandsCfg:
    """Velocity commands for the locomotion task."""

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=HEADING_STIFFNESS,
        rel_standing_envs=REL_STANDING_ENVS,
        rel_heading_envs=1.0,
        # a new command every 5-10 s: long enough to actually track it, short enough
        # to see several commands per episode
        resampling_time_range=(5.0, 10.0),
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=LIN_VEL_X_RANGE,
            lin_vel_y=(0.0, 0.0),  # not actuated on a differential drive
            ang_vel_z=ANG_VEL_Z_RANGE,
            heading=HEADING_RANGE,
        ),
    )


@configclass
class LocomotionObservationsCfg:
    """Observation specifications for the MDP (35 dims)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """What the policy sees.

        Compared with the balance task, the fixed ``heading`` and ``position`` dims are
        replaced by the ``velocity_commands``: the target is no longer "stay at 0" but
        "track this (vx, wz)", so the command *is* the goal.

        ``leg_rates`` was added so the hips can be the attitude actuator.  The firmware
        controller this robot is based on (``infantry-chassis``: LQR + VMC) uses the
        state ``[theta, dtheta, xb, dxb, phi, dphi]``, i.e. it feeds back the leg-angle
        RATE explicitly (the ``Tp`` row of its gain matrix has a gain of ~5.9 on
        ``dphi``).  An MLP has no memory, so without ``dL0``/``dphi`` as inputs the legs
        can only do proportional feedback on their pose and will lag when they start
        carrying the pitch/roll loop.
        """

        # base state
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=0.25)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)

        # the goal: commanded forward speed and yaw rate
        velocity_commands = ObsTerm(
            func=mdp.velocity_commands, params={"command_name": "base_velocity"}, scale=1.0
        )

        # leg state, relative to the commanded / solved values
        hip_pos_rel_nominal = ObsTerm(func=mdp.hip_pos_rel_nominal, params={"action_name": "joints"})
        knee_pos_rel_solved = ObsTerm(func=mdp.knee_pos_rel_solved, params={"action_name": "joints"})
        wheel_vel = ObsTerm(func=mdp.wheel_vel, params={"action_name": "joints"}, scale=0.1)

        # last action and the solved leg length / angle
        last_action = ObsTerm(func=mdp.last_action_raw, params={"action_name": "joints"})
        solved_leg_state = ObsTerm(func=mdp.solved_leg_state, params={"action_name": "joints"})

        # leg-length / leg-angle rates (dL0_l, dphi_l, dL0_r, dphi_r):
        # the "dphi" of the reference LQR state, needed to damp the leg attitude loop
        leg_rates = ObsTerm(func=mdp.leg_rates, params={"action_name": "joints"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LocomotionRewardsCfg:
    """Reward terms for the locomotion task.

    The objective, in the order the terms enforce it:

    1. **stay up** -- ``alive`` + ``fall_penalty``;
    2. **hold the stance** -- ``base_height`` (L1, strong) keeps the body at the
       measured nominal height instead of crouching, ``pitch`` (L1, strong) keeps the
       body level nose-up/nose-down, ``lin_vel_z`` stops bouncing;
    3. **walk upright** -- ``roll`` plus ``roll_when_straight_abs``.  A *permanent*
       sideways lean is not acceptable; the second term is the one that removes it,
       because it charges the full roll price only while the commanded yaw rate is
       near zero and lets roll go free when actually cornering;
    4. **track the command** -- the two exp kernels;
    5. **turn fast enough** -- ``wz`` is the dominant tracking term and the command
       range is wide, because a slow heading loop is what made the previous policy feel
       sluggish.

    Why a *small* amount of roll is still fine: a two-wheel robot has no lateral force
    source, so cornering needs a bank of ``atan(v*w/g)`` -- about 2.6 deg at the trained
    speed/turn-rate limits.  That is cheap under these weights, while the 10-20 deg lean
    the previous policy settled into is not.
    """

    # -- staying up ------------------------------------------------------
    alive = RewTerm(func=base_mdp.is_alive, weight=0.5)
    fall_penalty = RewTerm(func=mdp.fall_penalty, weight=-50.0)

    # -- hold the stance -------------------------------------------------
    # L1 terms: a constant pull from any distance, see mdp.rewards.pitch_abs /
    # base_height_abs for why the squared versions were not enough.
    #
    # base_height is deliberately relaxed to -4 (was -10): a leg-based balance strategy
    # HAS to change the stance -- leaning the leg by phi drops the body by
    # L0*(1-cos(phi)) (~1.5 cm at 20 deg) and lengthening/shortening the leg changes it
    # even more.  Penalising that at -10 pushes the policy straight back to "rigid legs".
    base_height = RewTerm(
        func=mdp.base_height_abs, weight=-4.0, params={"target_height": NOMINAL_HEIGHT}
    )
    # pitch stays the strict quantity: this is what the legs are supposed to regulate.
    pitch = RewTerm(func=mdp.pitch_abs, weight=-.0)

    # -- walk upright, but bank freely while cornering --------------------
    # The previous -0.3 "token" weight was what let the policy settle into a permanent
    # sideways lean.  The pair below fixes that without blocking cornering: the global
    # term keeps a sane bound, the conditional one pays in full only when the commanded
    # yaw rate is ~0 (i.e. when driving straight).
    roll = RewTerm(func=mdp.roll_abs, weight=-1.5)
    roll_straight = RewTerm(
        func=mdp.roll_when_straight_abs,
        weight=-2.5,
        params={"command_name": "base_velocity", "wz_scale": ANG_VEL_Z_RANGE[1]},
    )
    roll_rate = RewTerm(func=mdp.ang_vel_x_l2, weight=-0.1)

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # -- tracking the command --------------------------------------------
    # WEIGHT REBALANCE (v1.0.2).  This used to be lin 2.0 / yaw 2.5, and that asymmetry
    # produced a "stand still and rotate" local optimum that 2000 iterations could not
    # escape -- measured on the 2026-10-21 run:
    #
    #     track_ang_vel_z  2.2185 / 2.5  = 89% of its ceiling
    #     track_lin_vel_x  0.3959 / 2.0  = 20%
    #     Policy/mean_std  0.497 -> 0.067  (exploration collapsed)
    #
    # Why a stationary robot scores so well on the YAW term: `wz*` is produced by the
    # heading controller and decays to 0 once the target heading is reached; a robot that
    # does not move has `wz = 0`, so `exp(0) = 1` -- it collects the full 2.5 for free,
    # without ever translating.  Standing still therefore paid ~0.5 + 2.5 + 0.47 = 2.7
    # against ~4.5 for actually walking, i.e. 60% of the score with none of the risk.
    #
    # After the rebalance the gap is ~3.06 per step instead of ~1.8, and translation --
    # the actual goal -- is the dominant term.
    # v1.0.7 REBALANCE (measured on the 2026-10-22 00:00 run, 650 iterations):
    #   error_vel_xy      = 1.204 m/s   (the command mean is only 0.7 m/s!)
    #   track_lin_vel_x   = 0.620 / 4.0 = 15.5%
    #   track_ang_vel_z   = 1.359 / 1.5 = 91%   <- free while standing still
    #   mean_episode_length = 1000 (never falls), mean_std = 0.031 (exploration dead)
    # i.e. the policy settled on "stand still and be safe".  Two causes:
    #
    #  (a) standing still still PAID: 0.5 alive + 1.5 yaw + 0.23 exp - 0.57 L2 = +1.66/step;
    #  (b) the exp kernel is dead at large errors: with cmd 1.2 and v 0,
    #      exp(-(1.2)^2/0.25^2) = exp(-23) ~ 0, so it supplies NO gradient to start moving.
    #
    # The fix raises the cost of being at the wrong speed (exp +6.0, L2 -3.0 -- the L2
    # gradient -2*3*e grows with the error, so it still pulls when the exp is saturated)
    # and lowers the free yaw reward (1.5 -> 1.0).  Standing still now pays ~+0.12/step
    # against ~7.5 for walking, instead of +1.66 vs 6.0.
    # v1.0.8 PRECISION: target is `Metrics/base_velocity/error_vel_xy` and `..._yaw`
    # below 0.05.  NOTE the scale of that metric: the command term accumulates
    # `error / (resampling_time_max/step_dt) = error/500` every step and only zeroes it at
    # the EPISODE reset, so over a 1000-step episode the logged number is
    # `sum(error)/500 = 2 * mean_error`.  A logged 0.05 therefore means a true mean error
    # of 0.025 m/s.  Getting there needs three things, hence the changes below:
    #   (a) a reward that still discriminates at a few cm/s -> narrower exp kernel;
    #   (b) a term that keeps pulling at cm/s errors -> the L1 terms (constant gradient);
    #   (c) less friction on the fast corrections -> relaxed action_jerk / wheel_torques.
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_exp, weight=8.0, params={"command_name": "base_velocity", "std": 0.15}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.15}
    )
    # reaching the commanded heading, not merely matching the yaw rate
    heading_error = RewTerm(
        func=mdp.heading_error_l2, weight=-0.6, params={"command_name": "base_velocity"}
    )

    # -- steady-state stiffness the exp kernels cannot provide ------------
    # `exp(-e^2/std^2)` has ZERO gradient at e = 0, so small deviations oscillating at
    # the control rate are almost free: the policy is pushed to "roughly the commanded
    # speed" but not to a *steady* one.  Measured symptom: the robot advances in bursts
    # ("move, stop, move") and keeps a visible yaw wobble, because a two-wheel robot is
    # continuously making small steering corrections and nothing charged for them.
    #
    #   * L2 (gradient -2*w*e) grows with the error -> it is what breaks the "stand still"
    #     optimum and cleans up large errors (v1.0.7: -3.0);
    #   * L1 (gradient -w, CONSTANT) is the one that fixes the last few cm/s: at e = 0.02
    #     the L2 gradient is only 2*6*0.02 = 0.24, while L1 still pulls with its full 2.0.
    #     Without L1 the policy has no reason to go from 10 cm/s to 2 cm/s (v1.0.8).
    track_lin_vel_x_l1 = RewTerm(
        func=mdp.track_lin_vel_x_l1, weight=-2.0, params={"command_name": "base_velocity"}
    )
    track_ang_vel_z_l1 = RewTerm(
        func=mdp.track_ang_vel_z_l1, weight=-1.0, params={"command_name": "base_velocity"}
    )
    track_lin_vel_x_l2 = RewTerm(
        func=mdp.track_lin_vel_x_l2, weight=-6.0, params={"command_name": "base_velocity"}
    )
    track_ang_vel_z_l2 = RewTerm(
        func=mdp.track_ang_vel_z_l2, weight=-2.0, params={"command_name": "base_velocity"}
    )

    # no lateral actuator -> skidding sideways is never the right answer
    lin_vel_y = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.5)

    # -- keep the two wheels on a common axle ----------------------------
    # `phi` is measured per leg and NOTHING tied the left and right values together, so
    # the policy learned to lean one leg forward and the other one back and use that as
    # an extra yaw actuator.  The physical result is a SKEWED axle: the two wheels no
    # longer sit at the same fore-aft position, so the robot drives with one wheel ahead
    # of the other and scrubs sideways instead of going straight.
    #
    # The term is |L0_l*sin(phi_l) - L0_r*sin(phi_r)| in metres -- literally "how many
    # millimetres are the wheels out of line" -- and it is L1 so it pulls with a constant
    # force at any offset.
    #
    # Deliberately NOT penalised: the COMMON part of phi.  Both legs leaning together is
    # the inverted-pendulum CG shift that produces forward acceleration, so it must stay
    # free.  Only the left-right difference is constrained, which is also what the
    # reference firmware does (its VMC + LQR drives both leg angles to the same
    # reference).
    #
    # Scale check: 3 deg of asymmetry per leg at L0 = 0.25 m gives 26 mm of skew, and
    # 26 mm * 8 = 0.21 per step against an alive bonus of 0.5 -- a clear signal without
    # dominating the tracking terms.
    wheel_axle = RewTerm(
        func=mdp.leg_axle_skew_abs, weight=-8.0, params={"action_name": "joints"}
    )

    # -- make the LEGS the balancing actuator ----------------------------
    # Pitch is controllable by two redundant actuators on this robot:
    #   (a) wheel torque -- the reaction torque twists the body (Segway style);
    #   (b) the hips    -- changing the leg angle phi slides the body's centre of
    #                     gravity fore/aft over the wheel contact (inverted-pendulum
    #                     style), which needs no wheel torque at all.
    # Nothing in the old reward set charged anything for (a): with joint_torques at
    # -1e-4 a 2 Nm wheel torque costs 8e-6 per step against an alive bonus of 1e-2,
    # i.e. 1250x cheaper than staying alive.  PPO therefore kept the legs rigid and
    # balanced entirely with the wheels.
    #
    # `wheel_torques` charges only the WHEEL joints, so it does not discourage the
    # hips.  Inverted-pendulum locomotion by leaning needs almost no wheel torque
    # (the wheels only have to roll under the body), while wheel-torque balancing
    # spikes into the 1-3 Nm range.  This term is what flips the trade-off.
    #
    # It uses a DEADBAND at 1.0 Nm rather than a plain L2: the wheels are also the only
    # longitudinal actuator, and with the speed range now up to 2 m/s a plain sum(tau^2)
    # would charge normal acceleration (~0.9 Nm per wheel) and command steps (~1.7 Nm).
    # Only the attitude-correction spikes are penalised -- see mdp.rewards.
    wheel_torques = RewTerm(
        func=mdp.wheel_torques_excess_l2,
        weight=-5.0e-2,
        # v1.0.8: dead band 1.0 -> 1.5 N*m.  The maximum wheel torque is 3.69 N*m, so a
        # 1.0 dead band left only 2.7 N*m of free authority and charged for every
        # acceleration burst -- exactly what tight velocity tracking needs.  The term
        # still keeps the wheels from becoming the balance actuator (the reason it exists).
        params={"threshold": 1.5, "action_name": "joints"},
    )

    # -- smooth actuation ------------------------------------------------
    # action_rate is relaxed (was -1e-2): the legs must be allowed to move FAST to
    # track pitch/roll, otherwise the penalty alone keeps them stiff.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-4.0e-3)
    # NEW in v1.0.5: penalise the SECOND difference of the action, not the first.
    #
    # The legs still tremble a little after the reward/dynamics fixes.  Raising
    # `action_rate` cannot fix that -- it charges every change, including the fast leg
    # motion that keeps the body upright -- so the tremor needs a term that is blind to
    # smooth motion.  For a constant / ramp / accelerating trajectory
    #     a[t] - 2a[t-1] + a[t-2]
    # is ~0, while for a step-alternating (dithering) action it is 4x the amplitude.
    #
    # Scale: a tremor of +/-0.05 in every action dim gives sum(jerk^2) ~ 0.24, so -0.5
    # charges 0.12 per step against an alive bonus of 0.5 -- clearly worth removing,
    # while a smooth leg swing is essentially free.
    action_jerk = RewTerm(func=mdp.action_jerk_l2, weight=-0.3, params={"action_name": "joints"})
    # joint_torques covers the hips too, so keep it light (was -1e-4) -- penalising
    # hip torque directly fights the behaviour we want.
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    # was -1e-7 (effectively zero).  Joint acceleration is the most direct measure of the
    # tremor, so give it a real weight; 1e-6 still stays well below the tracking terms.
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    # weight lowered from -1e-4: at 2 m/s the wheels legitimately spin at 29 rad/s, so a
    # plain sum(w^2) costs 0.17 per step -- a 3.4% drag that biases the policy against
    # actually using the top of the speed range.
    wheel_spin = RewTerm(func=mdp.wheel_spin_l2, weight=-3.0e-5, params={"action_name": "joints"})

    # -- the closed loop must stay closed --------------------------------
    closure_residual = RewTerm(
        func=mdp.closure_residual_l2, weight=-1.0e3, params={"action_name": "joints"}
    )


@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity-command tracking environment."""

    #: 1024 is the largest environment count this machine can run without running the
    #: 8 GB GPU (and PhysX' fabric path) out of memory; ``--num_envs`` overrides it.
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: LocomotionObservationsCfg = LocomotionObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: LocomotionRewardsCfg = LocomotionRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: LocomotionCommandsCfg = LocomotionCommandsCfg()

    #: name of the action term that owns the loop-closing knee targets
    joint_action_name: str = "joints"

    def __post_init__(self):
        # general settings
        self.decimation = 4  # 50 Hz control on a 200 Hz physics step
        self.episode_length_s = 20.0  # long enough for 2-4 commands per episode
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Isaac Lab's default is 5 * 2**15; this scene only has a flat ground plane and
        # 1024 small robots (~11 links each), so the doubled buffer bought nothing and
        # only added GPU memory pressure -- the exact resource whose exhaustion shows up
        # as "CUDA error: an illegal memory access" inside the PhysX GPU allocator.
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15

        import os as _os

        # NOTE: these are per-process environment variables, NOT persistent settings.
        #   * they only live in the terminal window that exported them;
        #   * `set INFANTRY_RL_USE_FABRIC=` (empty) or closing the window reverts them;
        #   * nothing is written to the system/user environment.
        # The line below always prints the resolved state so a run can be checked.
        _fabric_off = _os.environ.get("INFANTRY_RL_USE_FABRIC", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        )
        if _fabric_off:
            self.sim.use_fabric = False
        print(f"[cfg] PhysX fabric        : {'DISABLED (INFANTRY_RL_USE_FABRIC)' if _fabric_off else 'enabled (default)'}")

        _iters = _os.environ.get("INFANTRY_RL_PHYSX_POS_ITERS")
        if _iters:
            try:
                self.sim.physx.max_position_iteration_count = int(_iters)
                print(f"[cfg] PhysX position iters: {int(_iters)} (INFANTRY_RL_PHYSX_POS_ITERS)")
            except ValueError:
                pass

        # command envelope: printed every run so it is obvious what the policy was
        # actually trained (or evaluated) on -- a play command outside this range is
        # out of distribution, not a policy failure
        print(
            f"[cfg] command ranges      : vx [{VX_MIN_MPS:+.2f}, {VX_MAX_MPS:+.2f}] m/s "
            f"(INFANTRY_RL_VX_MIN / INFANTRY_RL_VX_MAX), "
            f"standing envs {REL_STANDING_ENVS:.0%}, "
            f"wz clamp +/-{ANG_VEL_Z_RANGE[1]:.2f} rad/s, "
            f"heading stiffness {HEADING_STIFFNESS:.2f}"
        )
        if _ENABLE_TURNING:
            print(
                "[cfg] turning             : ENABLED (INFANTRY_RL_ENABLE_TURNING=1) -- "
                "heading targets sampled in [-pi, pi]"
            )
        else:
            print(
                "[cfg] turning             : DISABLED -- heading TARGET pinned to 0, the "
                "heading loop still cancels yaw drift (straight-line task)"
            )

        # viewer (only used in non-headless play)
        self.viewer.eye = (2.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.2)


@configclass
class LocomotionEnvCfg_PLAY(LocomotionEnvCfg):
    """Variant used for replay / scripted evaluation: no random command resampling."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6  # never time out during a replay
        # no observation corruption at play time
        self.observations.policy.enable_corruption = False
        # a fixed command owns the task: never resample it, never force "stand still"
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.base_velocity.rel_standing_envs = 0.0


__all__ = [
    "ANG_VEL_Z_RANGE",
    "HEADING_STIFFNESS",
    "LIN_VEL_X_RANGE",
    "LocomotionCommandsCfg",
    "LocomotionEnvCfg",
    "LocomotionEnvCfg_PLAY",
    "LocomotionObservationsCfg",
    "LocomotionRewardsCfg",
    "ActionsCfg",
    "RobotSceneCfg",
    "TerminationsCfg",
    "FALL_ANGLE",
    "FALL_HEIGHT",
    "NOMINAL_HEIGHT",
    "CommandTermCfg",
    "DoneTerm",
    "INFANTRY_RL_CFG",
]
