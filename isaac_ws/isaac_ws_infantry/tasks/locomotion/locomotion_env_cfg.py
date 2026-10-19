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

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaac_ws_infantry.mdp as mdp
from isaac_ws_infantry.robots import INFANTRY_RL_CFG
from isaac_ws_infantry.tasks.balance.balance_env_cfg import (
    FALL_ANGLE,
    FALL_HEIGHT,
    NOMINAL_HEIGHT,
    ActionsCfg,
    BalanceSceneCfg,
    TerminationsCfg,
)

#: forward/backward speed range the policy is trained on (m/s)
LIN_VEL_X_RANGE = (-0.5, 0.5)
#: yaw-rate range -- also the clamp on the heading controller output (rad/s).
#: 2.5 rad/s is ~143 deg/s, fast enough for a crisp in-place turn; the previous 1.5
#: made the robot feel sluggish.
ANG_VEL_Z_RANGE = (-2.5, 2.5)
#: how strongly a heading error is turned into a yaw-rate command (1/s). 1.5 means a
#: 1 rad heading error immediately asks for the maximum trained yaw rate.
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
            heading=(-3.14, 3.14),
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
    pitch = RewTerm(func=mdp.pitch_abs, weight=-4.0)

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
    # Rotation is weighted slightly above translation so in-place turns are crisp.
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.25}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.5, params={"command_name": "base_velocity", "std": 0.35}
    )
    # reaching the commanded heading, not merely matching the yaw rate
    heading_error = RewTerm(
        func=mdp.heading_error_l2, weight=-0.4, params={"command_name": "base_velocity"}
    )
    # no lateral actuator -> skidding sideways is never the right answer
    lin_vel_y = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.5)

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
    wheel_torques = RewTerm(
        func=base_mdp.joint_torques_l2,
        weight=-5.0e-2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_wheel", "right_wheel"], preserve_order=True
            )
        },
    )

    # -- smooth actuation ------------------------------------------------
    # action_rate is relaxed (was -1e-2): the legs must be allowed to move FAST to
    # track pitch/roll, otherwise the penalty alone keeps them stiff.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-4.0e-3)
    # joint_torques covers the hips too, so keep it light (was -1e-4) -- penalising
    # hip torque directly fights the behaviour we want.
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    wheel_spin = RewTerm(func=mdp.wheel_spin_l2, weight=-1.0e-4, params={"action_name": "joints"})

    # -- the closed loop must stay closed --------------------------------
    closure_residual = RewTerm(
        func=mdp.closure_residual_l2, weight=-1.0e3, params={"action_name": "joints"}
    )


@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity-command tracking environment."""

    #: 1024 is the largest environment count this machine can run without running the
    #: 8 GB GPU (and PhysX' fabric path) out of memory; ``--num_envs`` overrides it.
    scene: BalanceSceneCfg = BalanceSceneCfg(num_envs=1024, env_spacing=2.5)
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
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

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
    "BalanceSceneCfg",
    "TerminationsCfg",
    "FALL_ANGLE",
    "FALL_HEIGHT",
    "NOMINAL_HEIGHT",
    "CommandTermCfg",
    "DoneTerm",
    "INFANTRY_RL_CFG",
]
