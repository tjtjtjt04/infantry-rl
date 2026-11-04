"""Emergency-braking environment for the five-bar wheel-leg.

Goal: drive up to the target speed, then stop in a short distance and balance, with a
small amount of wheel slip allowed.  The task is built around an explicit two-phase
command schedule (see :class:`mdp.commands.BrakingVelocityCommand`):

1. **drive** -- ``vx = target_speed`` for a randomly sampled time (3-5 s).  The robot
   starts from rest, so this phase also trains the acceleration needed to be at speed
   when the brake arrives.
2. **brake** -- ``vx = 0`` for the rest of the episode.  The command step is the
   emergency-brake signal.

Unlike the turning task, the geometry is straight-line: ``wz = 0`` throughout.  The
observation and action dimensions are unchanged (35 / 6), so the policy can be
warm-started from the locomotion or turning checkpoints.

Physical feasibility (measured from the URDF and actuator config):

* total mass ~12.8 kg, wheel radius 0.068 m, track 0.4379 m;
* wheel effort limit 3.69 Nm per wheel -> max braking force ~108 N -> ~8.5 m/s^2;
* friction (mu ~0.9) allows ~8.8 m/s^2, so torque is the binding limit;
* from 3 m/s the ideal minimum is ``v^2 / (2a) ~ 0.53 m`` and ~0.35 s.  With the
  balance/leg constraints, 0.6-1.0 m is a realistic target; the default tolerance is
  1.0 m.  Do not set it to 0.2-0.3 m -- that is outside the actuator envelope.

Reward design (see :class:`BrakingRewardsCfg`):

* inherited velocity tracking already rewards accelerating to the target and then
  matching zero during the brake;
* ``brake_distance_exp`` / ``brake_distance_l1`` / ``brake_overshoot`` reward a SHORT
  stopping distance and charge only the part beyond ``stop_dist_tolerance``;
* ``brake_standstill`` gives an explicit bonus for staying stopped after the brake;
* the excess wheel-torque penalty is disabled and ``action_jerk`` is restricted to the
  hips, because a hard brake NEEDS a rapid wheel-torque change.  Wheel slip itself is
  not penalised -- it is allowed as long as the stopping distance stays short.
"""

from __future__ import annotations

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import mdp
from env_locomotion import (
    ActionsCfg,
    FiveBarEnv,
    LocomotionObservationsCfg,
    LocomotionRewardsCfg,
    RobotSceneCfg,
    TerminationsCfg,
)

# --------------------------------------------------------------------------- #
# Task envelope
# --------------------------------------------------------------------------- #
#: target speed range of the drive phase (m/s).  The user's maximum is 3.0 m/s;
#: starting the range at 2.0 keeps the first stage from being dominated by falls.
#: Set both to 3.0 for a max-speed-only final stage.
BRAKE_SPEED_MIN = float(os.environ.get("INFANTRY_RL_BRAKE_SPEED_MIN", "2.0"))
BRAKE_SPEED_MAX = float(os.environ.get("INFANTRY_RL_BRAKE_SPEED_MAX", "3.0"))

#: time spent at the target speed before the brake command (s)
DRIVE_TIME_MIN = float(os.environ.get("INFANTRY_RL_DRIVE_TIME_MIN", "3.0"))
DRIVE_TIME_MAX = float(os.environ.get("INFANTRY_RL_DRIVE_TIME_MAX", "5.0"))

#: one 9 s episode = drive (3-5 s) + brake (~1 s) + balance (~3-5 s)
EPISODE_LENGTH_S = float(os.environ.get("INFANTRY_RL_BRAKE_EPISODE", "9.0"))

#: thresholds that count as "stopped" for the standstill bonus / stop time
STOP_SPEED_THRESHOLD = float(os.environ.get("INFANTRY_RL_STOP_SPEED", "0.1"))
STOP_YAW_THRESHOLD = float(os.environ.get("INFANTRY_RL_STOP_YAW", "0.3"))

#: allowed stopping distance (m) before ``brake_overshoot`` starts charging.  This is
#: the "small slide is allowed" budget.  Physical minimum from 3 m/s is ~0.53 m.
STOP_DIST_TOLERANCE = float(os.environ.get("INFANTRY_RL_STOP_DIST_TOL", "1.0"))
#: distance scale (m) of the exp braking reward
BRAKE_DIST_SCALE = float(os.environ.get("INFANTRY_RL_BRAKE_DIST_SCALE", "0.5"))

#: Pitch control.  A hard stop naturally pitches the body a few degrees; the dead band
#: leaves that free, and beyond it a strong L1 penalty keeps the body from dragging on
#: the ground.  Measured play showed 21-62 deg transients with the old defaults, so these
#: are deliberately tighter now.
BRAKE_PITCH_TOLERANCE = float(os.environ.get("INFANTRY_RL_BRAKE_PITCH_TOL", "0.05"))  # rad ~2.9 deg
BRAKE_PITCH_WEIGHT = float(os.environ.get("INFANTRY_RL_BRAKE_PITCH_WEIGHT", "-10.0"))
#: damping on the pitch RATE (body y angular velocity)
BRAKE_PITCH_RATE_WEIGHT = float(os.environ.get("INFANTRY_RL_BRAKE_PITCH_RATE_WEIGHT", "-0.5"))
#: small symmetric |pitch| term (the inherited `pitch` term is disabled at 0 in locomotion)
BRAKE_PITCH_SYM_WEIGHT = float(os.environ.get("INFANTRY_RL_BRAKE_PITCH_SYM_WEIGHT", "-1.2"))

#: Dead-band penalty on excessive HIP deviation from the nominal stance.  Play showed a
#: mean 18 deg / max 25 deg (the range is 25.8 deg): the policy was saturating the hips,
#: lifting a wheel and falling.  Normal attitude work stays free inside the dead band.
BRAKE_HIP_DEV_TOL = float(os.environ.get("INFANTRY_RL_BRAKE_HIP_DEV_TOL", "0.10"))  # rad ~5.7 deg
BRAKE_HIP_DEV_WEIGHT = float(os.environ.get("INFANTRY_RL_BRAKE_HIP_DEV_WEIGHT", "-2.0"))

#: Terminate an episode when |pitch| alone exceeds this (rad).  The inherited
#: bad_orientation limit (1.2 rad on the gravity angle) is too lenient for a body that
#: can drag on the ground at 40-60 deg of pitch.
BRAKE_PITCH_LIMIT = float(os.environ.get("INFANTRY_RL_BRAKE_PITCH_LIMIT", "0.8"))  # rad ~45.8 deg

#: Straight-line braking uses the action term's heading-hold PD (target heading 0).
#: The policy then only owns the COMMON-mode wheel speed (drive/brake) and the hips,
#: while the PD produces the differential that keeps the robot straight.  This is the
#: same mechanism the original locomotion policy was trained with, so warm-start from
#: a locomotion checkpoint whose params/env.yaml has ``heading_control: true``.
#: Set INFANTRY_RL_BRAKE_HEADING_CONTROL=0 to let the policy own yaw instead.
BRAKE_HEADING_CONTROL = os.environ.get("INFANTRY_RL_BRAKE_HEADING_CONTROL", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# --------------------------------------------------------------------------- #
# MDP config
# --------------------------------------------------------------------------- #
@configclass
class BrakingCommandsCfg:
    """Drive at a target speed, then command zero and stay there."""

    base_velocity = mdp.BrakingVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        rel_standing_envs=0.0,
        # the schedule is the whole episode; never resample in the middle
        resampling_time_range=(1.0e6, 1.0e6),
        drive_time_range=(DRIVE_TIME_MIN, DRIVE_TIME_MAX),
        stop_speed_threshold=STOP_SPEED_THRESHOLD,
        stop_yaw_threshold=STOP_YAW_THRESHOLD,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(BRAKE_SPEED_MIN, BRAKE_SPEED_MAX),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),  # straight-line braking
            heading=None,
        ),
    )


@configclass
class BrakingObservationsCfg(LocomotionObservationsCfg):
    """Same 35-dim observation as locomotion.

    ``velocity_commands`` carries the drive target during the first phase and zero
    during the brake, which is exactly the signal the policy needs.  No extra
    observation is required, so existing checkpoints remain loadable.
    """


@configclass
class BrakingRewardsCfg(LocomotionRewardsCfg):
    """Locomotion balance terms + explicit stopping-distance rewards."""

    #: bounded "shorter stop is better" bonus, active only after the brake
    brake_distance_exp = RewTerm(
        func=mdp.brake_distance_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "scale": BRAKE_DIST_SCALE},
    )
    #: constant pull towards a shorter stop at every distance
    brake_distance_l1 = RewTerm(
        func=mdp.brake_distance_l1,
        weight=-2.0,
        params={"command_name": "base_velocity"},
    )
    #: allow a small slide; charge only the distance beyond the tolerance
    brake_overshoot = RewTerm(
        func=mdp.brake_overshoot,
        weight=-5.0,
        params={"command_name": "base_velocity", "tolerance": STOP_DIST_TOLERANCE},
    )
    #: explicit bonus for staying stopped after the brake
    brake_standstill = RewTerm(
        func=mdp.brake_standstill,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "speed_threshold": STOP_SPEED_THRESHOLD,
            "yaw_threshold": STOP_YAW_THRESHOLD,
        },
    )

    #: dead-band pitch penalty: no cost within +/-4.6 deg, then an L1 pull back to level
    pitch_excess = RewTerm(
        func=mdp.pitch_excess_abs,
        weight=BRAKE_PITCH_WEIGHT,
        params={"tolerance": BRAKE_PITCH_TOLERANCE},
    )
    #: damping on the pitch rate (body y angular velocity)
    pitch_rate = RewTerm(func=mdp.ang_vel_y_l2, weight=BRAKE_PITCH_RATE_WEIGHT)
    #: dead-band penalty on hips frozen near their range limits (wheel-lifting poses)
    hip_deviation = RewTerm(
        func=mdp.hip_deviation_abs,
        weight=BRAKE_HIP_DEV_WEIGHT,
        params={"action_name": "joints", "tolerance": BRAKE_HIP_DEV_TOL},
    )

    def __post_init__(self):
        super().__post_init__()
        # A hard brake NEEDS wheel torque.  The inherited term charges wheel torque
        # above 1.5 Nm, which would directly tax the behaviour being trained.
        self.wheel_torques.weight = 0.0
        # Rapid wheel-action changes are the brake actuator, not a leg tremor; keep
        # the jerk penalty on the four hip channels only.
        self.action_jerk.params["exclude_wheels"] = True
        # The inherited pitch term ships at 0 (disabled) in locomotion; the braking
        # task needs a base symmetric pull in addition to the dead-band term.
        self.pitch.weight = BRAKE_PITCH_SYM_WEIGHT


@configclass
class BrakingTerminationsCfg(TerminationsCfg):
    """Shared terminations + a braking-specific strict pitch limit."""

    pitch_limit = DoneTerm(
        func=mdp.pitch_above,
        params={"limit_angle": BRAKE_PITCH_LIMIT},
    )


@configclass
class BrakingEnvCfg(ManagerBasedRLEnvCfg):
    """Emergency braking: reach the target speed, stop in a short distance, balance."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: BrakingObservationsCfg = BrakingObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: BrakingRewardsCfg = BrakingRewardsCfg()
    terminations: BrakingTerminationsCfg = BrakingTerminationsCfg()
    commands: BrakingCommandsCfg = BrakingCommandsCfg()

    joint_action_name: str = "joints"

    def __post_init__(self):
        # simulation settings -- identical to locomotion / turning
        self.decimation = 4  # 50 Hz control
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15

        _fabric_off = os.environ.get("INFANTRY_RL_USE_FABRIC", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        )
        if _fabric_off:
            self.sim.use_fabric = False
        print(f"[cfg] PhysX fabric        : {'DISABLED (INFANTRY_RL_USE_FABRIC)' if _fabric_off else 'enabled (default)'}")

        _iters = os.environ.get("INFANTRY_RL_PHYSX_POS_ITERS")
        if _iters:
            try:
                self.sim.physx.max_position_iteration_count = int(_iters)
                print(f"[cfg] PhysX position iters: {int(_iters)} (INFANTRY_RL_PHYSX_POS_ITERS)")
            except ValueError:
                pass

        # Straight-line braking: the PD holds heading 0 and produces the differential
        # wheel command; the policy produces the common-mode (drive/brake) and the hips.
        # This matches the action semantics of a locomotion checkpoint trained with
        # heading_control=True -- use model_5448.pt as the warm start.
        self.actions.joints.heading_control = BRAKE_HEADING_CONTROL
        self.actions.joints.target_heading = 0.0
        self.actions.joints.position_control = False
        if BRAKE_HEADING_CONTROL:
            print(
                "[cfg] heading hold         : ENABLED (target 0 rad; "
                f"kp={self.actions.joints.heading_kp:.2f}, "
                f"kd={self.actions.joints.heading_kd:.2f}, "
                f"clamp +/-{self.actions.joints.heading_max_yaw_rate:.2f} rad/s). "
                "Warm-start from a checkpoint trained with heading_control=true "
                "(e.g. locomotion model_5448.pt)."
            )
        else:
            print(
                "[cfg] heading hold         : DISABLED (INFANTRY_RL_BRAKE_HEADING_CONTROL=0) "
                "-- the policy owns the wheel differential too."
            )

        vx_min, vx_max = self.commands.base_velocity.ranges.lin_vel_x
        print(
            f"[cfg] braking envelope    : drive vx [{vx_min:+.2f}, {vx_max:+.2f}] m/s, "
            f"then vx=0; drive time {self.commands.base_velocity.drive_time_range[0]:.1f}-"
            f"{self.commands.base_velocity.drive_time_range[1]:.1f} s, "
            f"episode {self.episode_length_s:.1f} s"
        )
        print(
            f"[cfg] stop criterion      : |vx| < {self.commands.base_velocity.stop_speed_threshold:.2f} m/s "
            f"and |wz| < {self.commands.base_velocity.stop_yaw_threshold:.2f} rad/s; "
            f"distance tolerance {STOP_DIST_TOLERANCE:.2f} m"
        )
        print(
            f"[cfg] pitch penalty       : |pitch| > {BRAKE_PITCH_TOLERANCE:.2f} rad "
            f"({BRAKE_PITCH_TOLERANCE * 57.2958:.1f} deg) weight {BRAKE_PITCH_WEIGHT:+.1f}, "
            f"rate {BRAKE_PITCH_RATE_WEIGHT:+.2f}, symmetric {BRAKE_PITCH_SYM_WEIGHT:+.2f}"
        )
        print(
            f"[cfg] hip penalty         : |hip dev| > {BRAKE_HIP_DEV_TOL:.2f} rad "
            f"({BRAKE_HIP_DEV_TOL * 57.2958:.1f} deg) weight {BRAKE_HIP_DEV_WEIGHT:+.1f}; "
            f"pitch termination at {BRAKE_PITCH_LIMIT:.2f} rad "
            f"({BRAKE_PITCH_LIMIT * 57.2958:.1f} deg)"
        )

        self.viewer.origin_type = "world"
        self.viewer.eye = (2.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.2)


@configclass
class BrakingEnvCfg_PLAY(BrakingEnvCfg):
    """Playback variant: never times out, so the post-stop balance can be observed."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)


# --------------------------------------------------------------------------- #
# Environment class
# --------------------------------------------------------------------------- #
class FiveBarBrakingEnv(FiveBarEnv):
    """Five-bar wheel-leg with a drive-then-emergency-stop command schedule."""

    cfg: BrakingEnvCfg


__all__ = [
    "BRAKE_DIST_SCALE",
    "BRAKE_HEADING_CONTROL",
    "BRAKE_HIP_DEV_TOL",
    "BRAKE_HIP_DEV_WEIGHT",
    "BRAKE_PITCH_LIMIT",
    "BRAKE_PITCH_RATE_WEIGHT",
    "BRAKE_PITCH_SYM_WEIGHT",
    "BRAKE_PITCH_TOLERANCE",
    "BRAKE_PITCH_WEIGHT",
    "BRAKE_SPEED_MAX",
    "BRAKE_SPEED_MIN",
    "BrakingCommandsCfg",
    "BrakingEnvCfg",
    "BrakingEnvCfg_PLAY",
    "BrakingObservationsCfg",
    "BrakingRewardsCfg",
    "DRIVE_TIME_MAX",
    "DRIVE_TIME_MIN",
    "EPISODE_LENGTH_S",
    "FiveBarBrakingEnv",
    "STOP_DIST_TOLERANCE",
    "STOP_SPEED_THRESHOLD",
    "STOP_YAW_THRESHOLD",
]
