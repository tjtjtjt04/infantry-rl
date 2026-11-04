"""Locomotion (velocity-command tracking) environment for the five-bar wheel-leg.

This file is the flat-layout equivalent of four original modules:

* ``tasks/common/robot_env_cfg.py`` -- the shared scene / action / termination config;
* ``tasks/common/five_bar_env.py``  -- :class:`FiveBarEnv`, which re-imposes the
  five-bar loop every control step;
* ``tasks/locomotion/locomotion_env_cfg.py`` -- commands, observations, rewards;
* ``tasks/locomotion/locomotion_env.py``     -- the environment class.

The robot is a two-wheel differential drive with a five-bar leg on each side.
PhysX sees a tree (10 DOFs): the two loop joints are not articulation DOFs, so the
passive knees are driven to the positions that close the loop, solved from the
measured hip angles every control step.

Why ``(vx, wz)`` and not ``(vx, vy, wz)``: there is no lateral actuator, so a
``vy`` command could never be tracked.  ``heading_command=True`` makes the command
manager derive ``wz = clip(stiffness * wrap(heading_target - heading))`` every
step, so "hold heading" and "turn to a heading" are the same mechanism.

The task values here are copied verbatim from the original workspace (including
``pitch`` at ``-.0`` and ``action_jerk`` at ``-0.3``).  Do not "correct" them
against the old README tables -- the code is the source of truth.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import mdp
from config import INFANTRY_RL_CFG, ensure_ground_usd
from robot_model import nominal_base_height

# --------------------------------------------------------------------------- #
# Shared stance / fall constants
# --------------------------------------------------------------------------- #
#: root height (m) of the measured nominal stance; the base-height reward holds this
NOMINAL_HEIGHT = nominal_base_height()
#: root height below which the robot counts as fallen (m)
FALL_HEIGHT = 0.11
#: tilt at which the robot counts as fallen (rad)
FALL_ANGLE = 1.2

#: heading (rad) the robot is supposed to hold when the action term's outer heading loop
#: is enabled.  All envs spawn with an identity base orientation, so 0 = "as at reset".
TARGET_HEADING = 0.0
#: planar position (m) the robot is supposed to hold, in the ENVIRONMENT frame
#: (``env_origins`` is the spawn point, so (0, 0) = "do not translate").
TARGET_X = 0.0
TARGET_Y = 0.0

#: hip action authority and hard travel.  Overridable through the environment so a
#: wider-hip teleop policy can be trained/deployed without touching every task cfg.
#: Defaults preserve the original checkpoint semantics.
HIP_SCALE = float(os.environ.get("INFANTRY_RL_HIP_SCALE", "0.25"))
HIP_RANGE = float(os.environ.get("INFANTRY_RL_HIP_RANGE", "0.45"))


# --------------------------------------------------------------------------- #
# Scene / actions / terminations (shared by every task)
# --------------------------------------------------------------------------- #
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat ground + the robot, shared by every task. Everything is local, nothing is fetched."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=ensure_ground_usd(),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot = INFANTRY_RL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=500.0, color=(0.6, 0.6, 0.6)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joints = mdp.FiveBarJointActionCfg(
        asset_name="robot",
        hip_scale=HIP_SCALE,
        wheel_scale=5.0,
        hip_range=HIP_RANGE,
        limit_margin=0.01,
        # 2 is provably enough (worst-case residual 3.6 um vs 750 um achievable in sim);
        # each extra iteration costs ~150 CUDA kernel launches per control step.
        solver_iters=2,
        # outer heading loop; enabled via balance task env vars
        heading_control=False,
        heading_mode="override",
        target_heading=TARGET_HEADING,
        heading_kp=3.0,
        heading_kd=0.35,
        heading_max_yaw_rate=3.0,
        wheel_radius=0.068,
        track_width=0.4379,
        # outer position loop; enabled via balance task env vars
        position_control=False,
        target_position_x=TARGET_X,
        target_position_y=TARGET_Y,
        position_kp=1.5,
        position_kd=0.6,
        position_max_speed=0.6,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the tasks."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": FALL_ANGLE})
    base_too_low = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": FALL_HEIGHT}
    )


# --------------------------------------------------------------------------- #
# Locomotion constants
# --------------------------------------------------------------------------- #
# Forward speed range the policy is trained on (m/s).  Both bounds are configurable
# through the environment so the envelope can be staged without touching code.
#   [0.2, 1.2] is a FORWARD-ONLY band:
#     * the robot never sees a reverse command, so it will not learn to back up;
#     * it never sees a near-zero *moving* command either -- stopping is still trained,
#       because `rel_standing_envs = 0.1` zeroes the whole command for 10% of the envs.
#   To get reverse back, set INFANTRY_RL_VX_MIN to a negative number.
# Hardware check: wheel limit 61.471 rad/s * 0.068 m = 4.18 m/s; 1.2 m/s sits at
# ~29% of the wheel limit (17.6 rad/s).
VX_MIN_MPS = float(os.environ.get("INFANTRY_RL_VX_MIN", "0.2"))
VX_MAX_MPS = float(os.environ.get("INFANTRY_RL_VX_MAX", "1.2"))

#: forward speed range the policy is trained on (m/s)
LIN_VEL_X_RANGE = (VX_MIN_MPS, VX_MAX_MPS)

#: TRAIN TURNING OR NOT.
#: Off (default) is the "straight line only" task: the heading target is pinned to 0,
#: so the robot is never asked to turn to a new direction.  It still *holds* its
#: heading, because the heading controller keeps running:
#:     wz* = clip(1.5 * wrap(0 - psi), +/-2.5)
#: i.e. any yaw drift still produces a correcting command.
#: Turning it back on restores random heading targets in [-pi, pi].
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

#: OPTIONAL extra penalty on ``|heading error|`` (rad), on top of the frozen 24-term set.
#: 0 (default) keeps the original reward set byte-identical.  Set a negative weight
#: (e.g. -1.5) for the wide-speed curriculum: the built-in heading term is quadratic,
#: so it has almost no gradient for the few-degree drift that matters at 3 m/s, while
#: an L1 term keeps pulling with a constant gradient all the way to zero.
HEADING_L1_WEIGHT = float(os.environ.get("INFANTRY_RL_HEADING_L1", "0"))

#: OPTIONAL multiplier applied to every yaw/heading reward term:
#: ``heading_error`` (L2), ``heading_error_l1`` (if enabled), ``track_ang_vel_z`` (exp),
#: ``track_ang_vel_z_l1`` and ``track_ang_vel_z_l2``.  1.0 (default) keeps the frozen
#: weights.  1.5-2.0 is a useful range when the measured heading deviation is too large:
#: it makes both "the heading is off" and "the yaw rate is off" cost more, while leaving
#: the forward-speed terms untouched.  Watch for a slower robot / more falls -- a yaw
#: correction needs banking, and that can fight the roll/height terms.
YAW_SCALE = float(os.environ.get("INFANTRY_RL_YAW_SCALE", "1"))


# --------------------------------------------------------------------------- #
# Locomotion task config
# --------------------------------------------------------------------------- #
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

        The target is not "stay at 0" but "track this (vx, wz)", so the command is
        the goal.  ``leg_rates`` was added so the hips can be the attitude actuator:
        the firmware LQR feeds back ``dphi`` explicitly, and an MLP has no memory.
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
    """Reward terms for the locomotion task (24 active terms, +1 opt-in heading term)."""

    # -- staying up ------------------------------------------------------
    alive = RewTerm(func=base_mdp.is_alive, weight=0.5)
    fall_penalty = RewTerm(func=mdp.fall_penalty, weight=-50.0)

    # -- hold the stance -------------------------------------------------
    # L1 terms pull with a constant force from any distance; the squared versions
    # either barely hold a small offset or explode when the robot trips.
    # base_height is deliberately relaxed to -4 (was -10): a leg-based balance
    # strategy has to change the stance.
    base_height = RewTerm(
        func=mdp.base_height_abs, weight=-4.0, params={"target_height": NOMINAL_HEIGHT}
    )
    # NOTE: the original on-disk workspace ships this at -0.0 (disabled).  Kept as-is.
    pitch = RewTerm(func=mdp.pitch_abs, weight=-.0)

    # -- walk upright, but bank freely while cornering --------------------
    roll = RewTerm(func=mdp.roll_abs, weight=-1.5)
    roll_straight = RewTerm(
        func=mdp.roll_when_straight_abs,
        weight=-2.5,
        params={"command_name": "base_velocity", "wz_scale": ANG_VEL_Z_RANGE[1]},
    )
    roll_rate = RewTerm(func=mdp.ang_vel_x_l2, weight=-0.1)

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # -- tracking the command --------------------------------------------
    # Translation is the dominant term: standing still must not be able to score
    # well on the yaw kernel (which decays to zero once the heading is reached).
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
    # exp(-e^2/std^2) has ZERO gradient at e = 0, so small deviations oscillating at
    # the control rate are almost free.  L2 breaks the large-error optimum; L1 has a
    # constant gradient and fixes the last few cm/s.
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
    # |L0_l*sin(phi_l) - L0_r*sin(phi_r)| in metres: literally "how many
    # millimetres are the wheels out of line".  Only the left-right difference is
    # constrained; the common part of phi is the inverted-pendulum CG shift.
    wheel_axle = RewTerm(
        func=mdp.leg_axle_skew_abs, weight=-8.0, params={"action_name": "joints"}
    )

    # -- make the LEGS the balancing actuator ----------------------------
    # Pitch has two redundant actuators: wheel reaction torque (Segway style) and
    # the hips (inverted-pendulum CG shift).  This term charges only the WHEEL
    # joints, with a dead band, so normal driving is free but attitude-correction
    # torque spikes are not.  It uses a DEADBAND at 1.5 Nm rather than a plain L2
    # because the wheels are also the only longitudinal actuator.
    wheel_torques = RewTerm(
        func=mdp.wheel_torques_excess_l2,
        weight=-5.0e-2,
        params={"threshold": 1.5, "action_name": "joints"},
    )

    # -- smooth actuation ------------------------------------------------
    # action_rate is relaxed so the legs can move fast enough to catch pitch.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-4.0e-3)
    # Second difference: ~0 for smooth motion, large only for a dithering action.
    action_jerk = RewTerm(func=mdp.action_jerk_l2, weight=-0.3, params={"action_name": "joints"})
    # joint_torques covers the hips too, so keep it light -- penalising hip torque
    # directly fights the behaviour we want.
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    # at 1.2 m/s the wheels legitimately spin at 17.6 rad/s, so the weight is small.
    wheel_spin = RewTerm(func=mdp.wheel_spin_l2, weight=-3.0e-5, params={"action_name": "joints"})

    # -- the closed loop must stay closed --------------------------------
    closure_residual = RewTerm(
        func=mdp.closure_residual_l2, weight=-1.0e3, params={"action_name": "joints"}
    )

    def __post_init__(self):
        # Opt-in extra heading term.  Added dynamically so the DEFAULT reward dict stays
        # exactly the 24-term set that tools/compare_cfg.py freezes (and that the
        # 1 m/s checkpoint was trained with).  Enable with INFANTRY_RL_HEADING_L1 or
        # train.py --heading_l1.
        if HEADING_L1_WEIGHT != 0.0:
            self.heading_error_l1 = RewTerm(
                func=mdp.heading_error_l1,
                weight=HEADING_L1_WEIGHT,
                params={"command_name": "base_velocity"},
            )
        # Opt-in yaw emphasis: scale the heading error and yaw-rate terms together.  The
        # speed-tracking and balance terms are untouched, so the trade-off is explicit
        # and the same in every curriculum stage.
        if YAW_SCALE != 1.0:
            for name in (
                "track_ang_vel_z",
                "track_ang_vel_z_l1",
                "track_ang_vel_z_l2",
                "heading_error",
                "heading_error_l1",
            ):
                term = getattr(self, name, None)
                if term is not None:
                    term.weight *= YAW_SCALE
            extra = (
                f", heading_l1 {self.heading_error_l1.weight:+.2f}"
                if hasattr(self, "heading_error_l1")
                else ""
            )
            print(
                f"[cfg] yaw weights        : x{YAW_SCALE:g}  (track_ang_vel_z "
                f"{self.track_ang_vel_z.weight:+.2f}, wz_l1 {self.track_ang_vel_z_l1.weight:+.2f}, "
                f"wz_l2 {self.track_ang_vel_z_l2.weight:+.2f}, "
                f"heading_l2 {self.heading_error.weight:+.2f}{extra})"
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
        # only added GPU memory pressure.
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15

        # -------------------------------------------------------------- #
        # GPU stability knobs (environment variables, so no code edit is needed)
        #
        #   INFANTRY_RL_USE_FABRIC=0        disable the PhysX fabric path
        #   INFANTRY_RL_PHYSX_POS_ITERS=4   lower the PhysX position iterations
        #
        # NOTE: these are per-process environment variables, NOT persistent settings.
        # -------------------------------------------------------------- #
        _fabric_off = os.environ.get("INFANTRY_RL_USE_FABRIC", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        )
        if _fabric_off:
            self.sim.use_fabric = False
        print(f"[cfg] PhysX fabric        : {'DISABLED (INFANTRY_RL_USE_FABRIC)' if _fabric_off else 'enabled (default)'}")

        # Optional ACTION-LEVEL heading loop (PD on yaw rate, applied on top of the
        # policy's wheel commands).  Off by default.  This is the "firmware" path from
        # mdp.actions: in "override" mode the policy's yaw-producing differential is
        # replaced by the PD, while its common-mode (forward-speed) command is kept.
        # Useful as a deployment guarantee for straight-line driving when the learned
        # yaw tracking is not tight enough -- no retraining required.
        if os.environ.get("INFANTRY_RL_HEADING_CONTROL", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            self.actions.joints.heading_control = True
            # Align the command manager's heading loop with the action-level PD, so the
            # wz the policy tracks (and is rewarded for) is the wz the PD actually
            # produces: wz_cmd = k_desired * error.  Without this the command would use
            # stiffness 1.5 while the PD uses kp 3.0, and the yaw-tracking reward would
            # punish the correction it is supposed to want.
            self.commands.base_velocity.heading_control_stiffness = self.actions.joints.heading_kp
            self.commands.base_velocity.ranges.ang_vel_z = (
                -self.actions.joints.heading_max_yaw_rate,
                self.actions.joints.heading_max_yaw_rate,
            )
            print(
                "[cfg] action-level heading loop: ENABLED (INFANTRY_RL_HEADING_CONTROL) -- "
                f"kp={self.actions.joints.heading_kp:.2f}, kd={self.actions.joints.heading_kd:.2f}, "
                f"clamp +/-{self.actions.joints.heading_max_yaw_rate:.2f} rad/s "
                f"(command stiffness matched to kp)"
            )

        _iters = os.environ.get("INFANTRY_RL_PHYSX_POS_ITERS")
        if _iters:
            try:
                self.sim.physx.max_position_iteration_count = int(_iters)
                print(f"[cfg] PhysX position iters: {int(_iters)} (INFANTRY_RL_PHYSX_POS_ITERS)")
            except ValueError:
                pass

        # command envelope: printed every run so it is obvious what the policy was
        # actually trained (or evaluated) on.
        vx_min, vx_max = self.commands.base_velocity.ranges.lin_vel_x
        print(
            f"[cfg] command ranges      : vx [{vx_min:+.2f}, {vx_max:+.2f}] m/s "
            f"(INFANTRY_RL_VX_MIN / INFANTRY_RL_VX_MAX), "
            f"standing envs {self.commands.base_velocity.rel_standing_envs:.0%}, "
            f"wz clamp +/-{self.commands.base_velocity.ranges.ang_vel_z[1]:.2f} rad/s, "
            f"heading stiffness {self.commands.base_velocity.heading_control_stiffness:.2f}"
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

        # Viewer (only used in non-headless play).  Keep it in "world" mode: it is only
        # set once at startup, after that the viewport camera is under the user's
        # control (orbit / WASD / select the robot and press F), which is what is
        # needed to inspect the wheel-leg posture up close.  A tracking camera is
        # available opt-in via play.py --follow.
        self.viewer.origin_type = "world"
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


# --------------------------------------------------------------------------- #
# Environment classes
# --------------------------------------------------------------------------- #
class FiveBarEnv(ManagerBasedRLEnv):
    """Manager-based RL environment with control-level five-bar loop closure.

    PhysX drops the closed chain (the articulation has 10 joints), so the passive
    knee targets are re-solved from the measured hip angles on every control step.
    """

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._joint_action_name: str = getattr(cfg, "joint_action_name", "joints")
        self._joint_action = self.action_manager.get_term(self._joint_action_name)
        self._robot = self.scene["robot"]

        #: measured loop error ``|W - E|`` per side (metres), ``(N, 2)`` = (left, right)
        self.closure_residual = torch.zeros(self.num_envs, 2, device=self.device)
        #: solver residual of the same quantity (metres), ``(N, 2)``
        self.solver_residual = torch.zeros(self.num_envs, 2, device=self.device)
        #: number of knee targets that had to be clipped to the URDF limits
        self.knee_target_clamped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        #: Per-env running sums for the console diagnostics.  Logged at every reset as
        #: ``Metrics/<name>_mean`` (plus the heading one in degrees), so the values are
        #: available regardless of the reward weights.  These are the quantities that
        #: actually tell you whether the run is healthy:
        #:   heading error, vx command/actual, yaw-rate command/actual,
        #:   wheel velocity target/actual and their gap (slip / tracking).
        self._diag_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "heading_error_abs_rad",
                "vx_cmd_abs",
                "vx_actual_abs",
                "wz_cmd_abs",
                "wz_actual_abs",
                # in-place subset: the A-spin / high-yaw diagnostic (vx_cmd = 0)
                "wz_inplace_cmd_abs",
                "wz_inplace_actual_abs",
                "inplace_count",
                "wheel_target_abs",
                "wheel_actual_abs",
                "wheel_error_abs",
                "roll_abs_rad",
                "pitch_abs_rad",
            )
        }
        self._diag_count = torch.zeros(self.num_envs, device=self.device)
        #: command mode of the LAST control step, used to split termination rates
        #: into spin (in-place yaw) and driving (everything else)
        self._last_inplace = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #: every active command term that carries a ``(vx, vy, wz)`` velocity command.
        #: The locomotion task uses a heading-derived ``wz``; the turning task samples
        #: ``wz`` directly.  Diagnostics (vx/wz tracking, wheel targets, command
        #: histogram) apply to both; the *heading error* metric only makes sense for
        #: the heading-command flavour, which is why the two lists are kept apart.
        self._velocity_command_names = [
            name
            for name in self.command_manager.active_terms
            if getattr(getattr(self.command_manager.get_term(name).cfg, "ranges", None), "lin_vel_x", None)
            is not None
        ]
        self._heading_command_names = [
            name
            for name in self._velocity_command_names
            if getattr(self.command_manager.get_term(name).cfg, "heading_command", False)
        ]
        #: Command-distribution histogram (5 equal bins over the sampled vx range +
        #: "standing").  The panel's "vx cmd" is the MEAN command, which for a uniform
        #: range sits in the middle; this histogram shows the actual coverage.
        self._vx_hist_bins = 5
        self._vx_hist_counts = torch.zeros(self.num_envs, self._vx_hist_bins, device=self.device)
        self._vx_stand_count = torch.zeros(self.num_envs, device=self.device)
        self._cmd_low, self._cmd_high = 0.0, 0.0
        if self._velocity_command_names:
            rng = getattr(self.command_manager.get_term(self._velocity_command_names[0]).cfg, "ranges", None)
            lin = getattr(rng, "lin_vel_x", None)
            if lin is not None:
                self._cmd_low, self._cmd_high = float(lin[0]), float(lin[1])

        # make sure the very first physics step already sees closed knees
        self._write_knee_targets()

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #
    @property
    def joint_action(self):
        """The :class:`~mdp.actions.FiveBarJointAction` term."""
        return self._joint_action

    @property
    def max_closure_residual(self) -> float:
        """Largest measured loop error over all environments and both legs (m)."""
        return float(self.closure_residual.max().item())

    # ------------------------------------------------------------------ #
    # loop closure
    # ------------------------------------------------------------------ #
    def _write_knee_targets(self) -> None:
        self._robot.set_joint_position_target(self._joint_action.knee_targets, joint_ids=self._joint_action.knee_ids)

    def step(self, action: torch.Tensor):
        """Recompute the knee targets, then run the standard environment step."""
        # 1) close the loop for the *measured* hip configuration of this control step
        self._joint_action.update_knee_targets()
        # 2) push the solved knee targets into the articulation buffers (they are
        #    re-written to PhysX on every physics sub-step inside super().step())
        self._write_knee_targets()

        obs, rewards, terminated, truncated, extras = super().step(action)

        # 3) record what the simulation actually did, for logging and verification.
        # PERFORMANCE: everything here stays on the GPU.  ``measured_closure_norm``
        # reuses the tensor the reward already computed during ``super().step()``.
        self.closure_residual = self._joint_action.measured_closure_norm(reuse_cache=True)
        self.solver_residual = self._joint_action.solver_residual
        self.knee_target_clamped = self._joint_action.knee_target_clamped

        # 4) weight-independent diagnostics, accumulated every control step and logged
        #    as episode means at reset.  All values stay on the GPU.
        if self._velocity_command_names:
            name = self._velocity_command_names[0]
            term = self.command_manager.get_term(name)
            cmd = self.command_manager.get_command(name)
            wheel_target = self._joint_action.applied_wheel_targets
            wheel_actual = self._robot.data.joint_vel[:, self._joint_action.wheel_ids]
            s = self._diag_sums
            if name in self._heading_command_names:
                s["heading_error_abs_rad"] += math_utils.wrap_to_pi(
                    term.heading_target - self._robot.data.heading_w
                ).abs()
            s["vx_cmd_abs"] += cmd[:, 0].abs()
            s["vx_actual_abs"] += self._robot.data.root_lin_vel_b[:, 0].abs()
            s["wz_cmd_abs"] += cmd[:, 2].abs()
            s["wz_actual_abs"] += self._robot.data.root_ang_vel_b[:, 2].abs()
            # in-place rotation: vx = 0 and a non-zero yaw command.  Kept separate
            # from the aggregate because a 50/50 mix of driving and spinning hides
            # whether the HIGH-spin band is actually being tracked.
            inplace = (cmd[:, 0].abs() < 1e-6) & (cmd[:, 2].abs() > 1e-6)
            s["wz_inplace_cmd_abs"] += cmd[:, 2].abs() * inplace
            s["wz_inplace_actual_abs"] += (
                self._robot.data.root_ang_vel_b[:, 2].abs() * inplace
            )
            s["inplace_count"] += inplace.float()
            self._last_inplace[:] = inplace
            s["wheel_target_abs"] += wheel_target.abs().mean(dim=1)
            s["wheel_actual_abs"] += wheel_actual.abs().mean(dim=1)
            s["wheel_error_abs"] += (wheel_actual - wheel_target).abs().mean(dim=1)
            roll, pitch, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
            s["roll_abs_rad"] += roll.abs()
            s["pitch_abs_rad"] += pitch.abs()
            # command-distribution histogram (standing commands counted separately).
            # In the turning task a pure rotation has vx = 0 but is obviously moving,
            # so the yaw-rate command counts as motion there; the heading task keeps
            # the original vx-only definition.
            moving = cmd[:, 0].abs() > 1e-6
            if not self._heading_command_names:
                moving = moving | (cmd[:, 2].abs() > 1e-6)
            if self._cmd_high - self._cmd_low > 1e-9:
                frac = ((cmd[:, 0] - self._cmd_low) / (self._cmd_high - self._cmd_low)).clamp(0.0, 0.999999)
                idx = (frac * self._vx_hist_bins).long()
                for b in range(self._vx_hist_bins):
                    self._vx_hist_counts[:, b] += (moving & (idx == b)).float()
            self._vx_stand_count += (~moving).float()
            self._diag_count += 1.0

        if isinstance(extras, dict):
            extras["closure_residual"] = self.closure_residual
            extras["solver_residual"] = self.solver_residual
        return obs, rewards, terminated, truncated, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the loop solver warm start together with the environment."""
        super()._reset_idx(env_ids)
        # the reset event has just written the hip pose; restart the solver at the
        # knee configuration that closes *that* pose.  Nominal for every task except
        # squat (where the action term holds the deepest feasible closed-chain pose).
        reset_knees = getattr(self._joint_action, "reset_knees", None)
        if reset_knees is None:
            reset_knees = self._joint_action.nominal_knees[0]
        self._joint_action._knee_seed[env_ids] = reset_knees.view(1, 2, 2)
        self._joint_action._knee_targets[env_ids] = reset_knees
        self._robot.set_joint_position_target(
            self._joint_action.knee_targets[env_ids],
            joint_ids=self._joint_action.knee_ids,
            env_ids=env_ids,
        )
        self.closure_residual[env_ids] = 0.0
        self.solver_residual[env_ids] = 0.0

        # Log the episode means for the envs that just finished, then clear their
        # accumulators.  These are independent of the reward weights (and exist even for
        # the PD runs that do not enable --heading_l1).
        if self._velocity_command_names:
            count = self._diag_count[env_ids].clamp(min=1.0)
            log = self.extras.setdefault("log", {})
            # Heading error is relative to the command manager's heading target and is
            # meaningless when `wz` is commanded directly (turning task), so it is only
            # logged for heading-command tasks.
            if self._heading_command_names:
                log["Metrics/heading_error_mean_deg"] = float(
                    (
                        (self._diag_sums["heading_error_abs_rad"][env_ids] / count)
                        * 57.29577951308232
                    ).mean().item()
                )
            # In-place spin: aggregate ratio over the reset envs, NOT a per-env
            # mean divided by a clamped count.  With few envs, an env that saw no
            # in-place command would otherwise contribute a fake 0 and drag the
            # reported spin speed down.
            inplace_steps = self._diag_sums["inplace_count"][env_ids].sum()
            inplace_n = inplace_steps.clamp(min=1.0)
            log["Metrics/wz_inplace_cmd_abs_mean"] = float(
                (self._diag_sums["wz_inplace_cmd_abs"][env_ids].sum() / inplace_n).item()
            )
            log["Metrics/wz_inplace_actual_abs_mean"] = float(
                (self._diag_sums["wz_inplace_actual_abs"][env_ids].sum() / inplace_n).item()
            )
            log["Metrics/inplace_pct"] = float(
                (inplace_steps / self._diag_count[env_ids].sum().clamp(min=1.0) * 100.0).item()
            )
            # Split the termination reasons by command mode.  The global
            # Episode_Termination/* rates mix high-speed driving with spin, so a
            # healthy spin can look bad (or vice versa) in the training panel.
            tm = getattr(self, "termination_manager", None)
            flags_all = getattr(tm, "_last_episode_dones", None) if tm is not None else None
            if flags_all is not None:
                names = list(getattr(tm, "active_terms", []))
                flags = flags_all[env_ids]
                spin = self._last_inplace[env_ids]
                n_total = int(flags.shape[0])
                n_spin = int(spin.sum().item())
                n_drive = n_total - n_spin
                log["Metrics/spin_reset_pct"] = 100.0 * n_spin / max(n_total, 1)
                for term in ("pitch_limit", "base_too_low", "bad_orientation"):
                    if term not in names:
                        continue
                    idx = names.index(term)
                    log[f"Metrics/spin_term_{term}"] = (
                        float(flags[spin, idx].float().mean().item()) if n_spin else 0.0
                    )
                    log[f"Metrics/drive_term_{term}"] = (
                        float(flags[~spin, idx].float().mean().item()) if n_drive else 0.0
                    )
            for name, buf in self._diag_sums.items():
                if name in (
                    "heading_error_abs_rad",
                    "roll_abs_rad",
                    "pitch_abs_rad",
                    "wz_inplace_cmd_abs",
                    "wz_inplace_actual_abs",
                    "inplace_count",
                ):
                    continue
                log[f"Metrics/{name}_mean"] = float((buf[env_ids] / count).mean().item())
            rad2deg = 57.29577951308232
            for name, key in (
                ("roll_abs_rad", "roll_abs_mean_deg"),
                ("pitch_abs_rad", "pitch_abs_mean_deg"),
            ):
                log[f"Metrics/{key}"] = float(
                    ((self._diag_sums[name][env_ids] / count) * rad2deg).mean().item()
                )
            hist_pct = (self._vx_hist_counts[env_ids] / count.unsqueeze(1)) * 100.0
            for b in range(self._vx_hist_bins):
                log[f"Metrics/vx_cmd_hist{b}_pct"] = float(hist_pct[:, b].mean().item())
            log["Metrics/vx_cmd_standing_pct"] = float(
                ((self._vx_stand_count[env_ids] / count) * 100.0).mean().item()
            )
            for buf in self._diag_sums.values():
                buf[env_ids] = 0.0
            self._vx_hist_counts[env_ids] = 0.0
            self._vx_stand_count[env_ids] = 0.0
            self._diag_count[env_ids] = 0.0


class FiveBarLocomotionEnv(FiveBarEnv):
    """Five-bar wheel-leg robot with velocity commands and loop-closure control."""

    cfg: LocomotionEnvCfg


__all__ = [
    "ANG_VEL_Z_RANGE",
    "ActionsCfg",
    "CommandTermCfg",
    "DoneTerm",
    "FALL_ANGLE",
    "HIP_RANGE",
    "HIP_SCALE",
    "FALL_HEIGHT",
    "FiveBarEnv",
    "FiveBarLocomotionEnv",
    "HEADING_L1_WEIGHT",
    "HEADING_RANGE",
    "HEADING_STIFFNESS",
    "INFANTRY_RL_CFG",
    "LIN_VEL_X_RANGE",
    "LocomotionCommandsCfg",
    "LocomotionEnvCfg",
    "LocomotionEnvCfg_PLAY",
    "LocomotionObservationsCfg",
    "LocomotionRewardsCfg",
    "NOMINAL_HEIGHT",
    "REL_STANDING_ENVS",
    "RobotSceneCfg",
    "TARGET_HEADING",
    "TARGET_X",
    "TARGET_Y",
    "TerminationsCfg",
    "VX_MAX_MPS",
    "VX_MIN_MPS",
    "YAW_SCALE",
]
