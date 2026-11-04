"""Unified teleop environment for the five-bar wheel-leg.

This is the single policy that a hand controller drives.  It answers the same
``(vx, wz)`` command language as the turning task, but its command distribution also
contains the two things a joystick does that turning never trained:

* **stops**: ``(vx, wz)`` are set to zero every so often, from whatever speed and
  yaw rate the robot currently has.  A moving -> zero transition starts a brake
  event, so the braking rewards (short stop distance, no overshoot, stand still)
  are applied to "stop from any state" instead of the fixed drive-then-brake
  schedule of :mod:`env_braking`;
* **reverse**: ``vx`` goes down to ``-0.8`` m/s.  The first training stage can set
  ``--teleop_vx_min 0.0`` and add reverse in a short second stage.

Heading hold
------------
The 35-dim observation is unchanged (no heading-error input), so the straight-line
drift is removed at the action level: while ``|wz_cmd|`` is below a threshold the
action term's PD owns the wheel differential and locks the heading the robot had
when the straight command arrived.  While ``|wz_cmd|`` is larger the policy owns the
differential and can actually turn.  See
:meth:`mdp.actions.FiveBarJointAction._conditional_heading_active`.

Command envelope (defaults, all configurable):
    vx in [-0.8, 3.0] m/s; wz in [-1.5, 1.5] rad/s while driving
    in-place rotation: vx = 0, |wz| in [0.5, 2.0] rad/s
    low-speed creep: 25% of the remaining moving envs in vx 0.05-0.20 m/s, full yaw
    |vx*wz| <= 4.0 m/s^2 (the same feasibility cap as turning)
    resample every 2-4 s; 10% standing; 20% forced stop of the non-standing envs;
    20% in-place rotation of the moving envs.

Rewards are the braking task's balance + posture + brake set, with the turning
task's ``roll_straight`` scale.  Observations (35) and actions (6) are identical to
turning/locomotion, so ``model_6447.pt`` can be loaded with ``--resume``.
"""

from __future__ import annotations

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import mdp
from env_braking import (
    BRAKE_HIP_DEV_TOL,
    BRAKE_HIP_DEV_WEIGHT,
    BRAKE_PITCH_LIMIT,
    BRAKE_PITCH_RATE_WEIGHT,
    BRAKE_PITCH_SYM_WEIGHT,
    BRAKE_PITCH_TOLERANCE,
    BRAKE_PITCH_WEIGHT,
    STOP_DIST_TOLERANCE,
    STOP_SPEED_THRESHOLD,
    STOP_YAW_THRESHOLD,
    BrakingRewardsCfg,
    BrakingTerminationsCfg,
)
from env_locomotion import (
    ActionsCfg,
    FiveBarEnv,
    LocomotionObservationsCfg,
    RobotSceneCfg,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        print(f"[cfg] WARNING: {name}={os.environ.get(name)!r} is not a number; using {default}")
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Command envelope
# --------------------------------------------------------------------------- #
#: forward / reverse speed range (m/s).  The full teleop envelope includes reverse;
#: the first recommended training stage overrides the lower bound to 0.0.
VX_MIN_MPS = _env_float("INFANTRY_RL_TELEOP_VX_MIN", -0.8)
VX_MAX_MPS = _env_float("INFANTRY_RL_TELEOP_VX_MAX", 3.0)

#: normal driving yaw-rate limit (rad/s).  `--wz_max` / INFANTRY_RL_WZ_MAX also
#: controls this, so the same CLI flag works for turning and teleop.
WZ_MAX_RPS = _env_float("INFANTRY_RL_WZ_MAX", 1.5)

#: in-place rotation limit (rad/s).  It is separate on purpose: a two-wheel robot can
#: spin faster standing still than it can while also tracking a forward speed.
WZ_INPLACE_MAX_RPS = _env_float("INFANTRY_RL_TELEOP_WZ_INPLACE_MAX", 2.0)

#: fraction of each resample that gets a pure in-place rotation
TURN_IN_PLACE_PROB = _env_float("INFANTRY_RL_TURN_IN_PLACE_PROB", 0.2)
#: smallest |wz| used by the in-place mode (rad/s)
TURN_IN_PLACE_MIN_WZ = _env_float("INFANTRY_RL_TELEOP_TURN_MIN_WZ", 0.5)

#: fraction of the NON-standing samples forced to (0, 0).  With 10% standing this
#: makes 28% of all resamples a stop, and every moving -> zero one is a brake event.
BRAKE_PROB = _env_float("INFANTRY_RL_TELEOP_BRAKE_PROB", 0.2)

#: feasibility cap on |vx*wz| for driving commands (m/s^2); 0 disables
MAX_LATERAL_ACCEL = _env_float("INFANTRY_RL_MAX_LAT_ACCEL", 4.0)

#: weight on the always-on roll penalty.  Banking into a turn needs the body to
#: roll toward the centre, so high-yaw training sets this to 0 and relies on
#: ``roll_straight`` (which fades out with commanded yaw).  Default preserves the
#: original checkpoint's reward.
TELEOP_ROLL_WEIGHT = _env_float("INFANTRY_RL_TELEOP_ROLL_WEIGHT", -1.5)

#: P2 low-speed creep: fraction of the remaining moving samples overdrawn in the
#: small-stick band, and the band itself (m/s).  The env clamps the band to the
#: active vx range and disables the mode when they do not overlap.
LOW_SPEED_PROB = _env_float("INFANTRY_RL_TELEOP_LOW_SPEED_PROB", 0.25)
LOW_SPEED_MIN = _env_float("INFANTRY_RL_TELEOP_LOW_SPEED_MIN", 0.05)
LOW_SPEED_MAX = _env_float("INFANTRY_RL_TELEOP_LOW_SPEED_MAX", 0.2)

#: command resampling interval (s): 2-4 s gives an episode 6-10 different commands,
#: covering start, continuous change, turn and stop.
RESAMPLING_TIME_RANGE = (
    _env_float("INFANTRY_RL_TELEOP_RESAMPLE_MIN", 2.0),
    _env_float("INFANTRY_RL_TELEOP_RESAMPLE_MAX", 4.0),
)

#: fraction of resamples commanded to stand still (same as locomotion/turning)
REL_STANDING_ENVS = 0.1

#: conditional heading hold (see the module and mdp.actions docstrings)
HEADING_HOLD_ENABLED = _env_bool("INFANTRY_RL_TELEOP_HEADING_HOLD", True)
HEADING_HOLD_WZ_THRESHOLD = _env_float("INFANTRY_RL_TELEOP_HEADING_HOLD_WZ", 0.15)
#: clamp on the PD's yaw-rate demand (rad/s).  This is a correction loop, not a
#: turning command, so it does not need to exceed the in-place range.
HEADING_HOLD_MAX_YAW_RATE = 2.0
#: cap on the PD correction's lateral acceleration |vx_actual * wz_pd| (m/s^2).
#: The command sampler obeys the same cap; the PD must too, otherwise a heading
#: correction at high speed can exceed the robot's bank capability and tip it over.
HEADING_HOLD_MAX_LATERAL_ACCEL = _env_float("INFANTRY_RL_TELEOP_HEADING_LAT_ACCEL", 4.0)

#: scripted symmetric stance bias (A + RB leg extension).  ``STANCE_MIN`` is the
#: crouch used while spinning, ``STANCE_MAX`` the extension held with RB.  The
#: amounts are hip offsets in rad: measured kinematics give
#: ``base_height ~= NOMINAL + 0.195 * d``, so -0.35 lowers the body to ~0.122 m
#: (the mechanical deep crouch in the hardware photos) and +0.30 raises it ~6 cm.
#: Randomization is training-only; deployment drives the same target from the
#: gamepad (A = crouch, A+RB = extension).
STANCE_RANDOMIZE = _env_bool("INFANTRY_RL_STANCE_RANDOMIZE", False)
STANCE_MIN = _env_float("INFANTRY_RL_STANCE_MIN", -0.60)
STANCE_MAX = _env_float("INFANTRY_RL_STANCE_MAX", 0.30)
STANCE_SLEW = _env_float("INFANTRY_RL_STANCE_SLEW", 0.5)
STANCE_GATE_VX = _env_float("INFANTRY_RL_STANCE_GATE_VX", 0.10)
STANCE_GATE_WZ = _env_float("INFANTRY_RL_STANCE_GATE_WZ", 1.0)
#: measured base_z(d) = 0.1888 + 0.1901 d + 0.0173 d^2 (fit to the five-bar
#: kinematics over [-0.6, +0.45], max error ~3 mm)
STANCE_HEIGHT_GAIN = _env_float("INFANTRY_RL_STANCE_HEIGHT_GAIN", 0.1901)
STANCE_HEIGHT_GAIN2 = _env_float("INFANTRY_RL_STANCE_HEIGHT_GAIN2", 0.0173)
#: the deep-crouch pose sits at base_z≈0.084 at d=-0.60 and ~0.078 at d=-0.65,
#: so the default 0.11 m fall threshold would terminate every episode.  0.065
#: leaves ~13-19 mm margin; orientation/pitch guards still catch real falls.
TELEOP_FALL_HEIGHT = _env_float("INFANTRY_RL_TELEOP_FALL_HEIGHT", 0.065)
#: discrete sampling probabilities during training.  The remainder is extension.
#: The deployment default spin is the crouch, so training runs for it use a
#: crouch-heavy mix (e.g. 0.5 crouch / 0.1 nominal / 0.4 extend).
STANCE_PROB_CROUCH = _env_float("INFANTRY_RL_STANCE_PROB_CROUCH", 0.30)
STANCE_PROB_NOMINAL = _env_float("INFANTRY_RL_STANCE_PROB_NOMINAL", 0.20)
#: uniform jitter added to every stance sample (rad); set 0 for a perfectly
#: fixed crouch/extension during a focused balance fine-tune.
STANCE_JITTER = _env_float("INFANTRY_RL_STANCE_JITTER", 0.02)

#: "off" or "master_rf".  master_rf slaves all four hip targets to the right-front
#: leg: [+t, -t, +t, -t] about nominal.  That is the "all legs like the right-front
#: leg" fix: no front/back angle difference (nose stays level) and left/right equal.
#: Opt-in because it changes the action semantics and must be trained.
HIP_SYMMETRY = os.environ.get("INFANTRY_RL_TELEOP_HIP_SYMMETRY", "off").strip().lower()
#: only apply master_rf for in-place commands; normal driving/turning keeps the
#: original independent-hip gait.  Must be trained with the same gate.
HIP_SYMMETRY_GATE = _env_bool("INFANTRY_RL_TELEOP_HIP_SYMMETRY_GATE", False)
#: weight / dead band of the gated "do not cancel the scripted stance" term.
#: Raise the weight and/or lower the tolerance if the policy keeps undoing the
#: extension (measured: the deep60 model only reached base_z 0.175 at d=+0.30,
#: vs the 0.247 geometric target).
STANCE_HOLD_WEIGHT = _env_float("INFANTRY_RL_STANCE_HOLD_WEIGHT", -2.0)
STANCE_HOLD_TOL = _env_float("INFANTRY_RL_STANCE_HOLD_TOL", 0.10)


def _validate_envelope() -> None:
    if VX_MIN_MPS > VX_MAX_MPS:
        raise ValueError(
            f"teleop vx range is empty: [{VX_MIN_MPS}, {VX_MAX_MPS}] "
            "(check --teleop_vx_min / --teleop_vx_max)"
        )
    if WZ_MAX_RPS <= 0.0:
        raise ValueError("teleop wz_max must be positive")
    if WZ_INPLACE_MAX_RPS < TURN_IN_PLACE_MIN_WZ:
        raise ValueError(
            f"in-place wz max ({WZ_INPLACE_MAX_RPS}) is below the in-place minimum "
            f"({TURN_IN_PLACE_MIN_WZ})"
        )
    if RESAMPLING_TIME_RANGE[0] > RESAMPLING_TIME_RANGE[1]:
        raise ValueError(f"bad teleop resampling range {RESAMPLING_TIME_RANGE}")
    if LOW_SPEED_MIN > LOW_SPEED_MAX:
        raise ValueError(
            f"teleop low-speed band is empty: [{LOW_SPEED_MIN}, {LOW_SPEED_MAX}]"
        )
    for name, value in (
        ("turn_in_place_prob", TURN_IN_PLACE_PROB),
        ("brake_prob", BRAKE_PROB),
        ("rel_standing_envs", REL_STANDING_ENVS),
        ("low_speed_prob", LOW_SPEED_PROB),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"teleop {name}={value} must be in [0, 1]")
    if STANCE_MIN > 0.0 or STANCE_MAX < 0.0:
        raise ValueError(
            f"stance range must contain 0: [{STANCE_MIN}, {STANCE_MAX}]"
        )
    if STANCE_SLEW <= 0.0:
        raise ValueError(f"stance slew must be positive, got {STANCE_SLEW}")
    if HIP_SYMMETRY not in ("off", "master_rf"):
        raise ValueError(
            f"teleop hip symmetry must be 'off' or 'master_rf', got {HIP_SYMMETRY!r}"
        )
    for name, value in (
        ("stance_prob_crouch", STANCE_PROB_CROUCH),
        ("stance_prob_nominal", STANCE_PROB_NOMINAL),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"teleop {name}={value} must be in [0, 1]")
    if STANCE_PROB_CROUCH + STANCE_PROB_NOMINAL > 1.0:
        raise ValueError(
            "stance_prob_crouch + stance_prob_nominal must be <= 1, got "
            f"{STANCE_PROB_CROUCH} + {STANCE_PROB_NOMINAL}"
        )


_validate_envelope()


# --------------------------------------------------------------------------- #
# Task config
# --------------------------------------------------------------------------- #
@configclass
class TeleopCommandsCfg:
    """Direct ``(vx, wz)`` commands with straight, turn, in-place and brake samples."""

    base_velocity = mdp.TeleopVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        rel_standing_envs=REL_STANDING_ENVS,
        resampling_time_range=RESAMPLING_TIME_RANGE,
        turn_in_place_prob=TURN_IN_PLACE_PROB,
        turn_in_place_min_abs=TURN_IN_PLACE_MIN_WZ,
        turn_in_place_max_abs=WZ_INPLACE_MAX_RPS,
        max_lateral_accel=MAX_LATERAL_ACCEL,
        brake_prob=BRAKE_PROB,
        brake_speed_threshold=STOP_SPEED_THRESHOLD,
        brake_yaw_threshold=STOP_YAW_THRESHOLD,
        low_speed_prob=LOW_SPEED_PROB,
        low_speed_min=LOW_SPEED_MIN,
        low_speed_max=LOW_SPEED_MAX,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(VX_MIN_MPS, VX_MAX_MPS),
            lin_vel_y=(0.0, 0.0),  # not actuated on a differential drive
            ang_vel_z=(-WZ_MAX_RPS, WZ_MAX_RPS),
            heading=None,
        ),
    )


@configclass
class TeleopObservationsCfg(LocomotionObservationsCfg):
    """The unchanged 35-dim observation.

    ``velocity_commands`` already carries ``(vx, wz)``.  The conditional heading
    hold is an action-level guarantee, so no heading observation (and no change to
    the 35-dim interface) is needed.
    """


@configclass
class TeleopRewardsCfg(BrakingRewardsCfg):
    """Braking task rewards (balance + stop distance + posture) for direct turning.

    Inheriting :class:`~env_braking.BrakingRewardsCfg` gives, verbatim:

    * the four brake rewards (``brake_distance_exp/l1``, ``brake_overshoot``,
      ``brake_standstill``), now driven by any moving -> zero command transition;
    * ``pitch_excess`` / ``pitch_rate`` / symmetric ``pitch`` and
      ``hip_deviation``, which stop the s2a-style "brace on the hips" solution;
    * wheel-torque penalty disabled and action jerk restricted to the hips, because
      a hard stop NEEDS a rapid wheel-torque change.

    The roll terms are adjusted for banking: the always-on ``roll`` penalty is
    configurable (default unchanged) and ``roll_straight`` relaxes over the
    DRIVING yaw range, not the (possibly much larger) in-place range.  Using the
    in-place cap here would remove the straight-line roll penalty over most of the
    driving envelope.
    """

    def __post_init__(self):
        super().__post_init__()
        self.roll.weight = TELEOP_ROLL_WEIGHT
        self.roll_straight.params["wz_scale"] = WZ_MAX_RPS

        # -- scripted stance (leg extension) ---------------------------------
        # The inherited constant-height and nominal-hip terms would fight the
        # extension; rebind only the TELEOP instances, leaving recovery/stand
        # (which use the same underlying functions) untouched.
        self.base_height.func = mdp.base_height_stance_abs
        self.base_height.params.update(
            {
                "action_name": "joints",
                "stance_gain": STANCE_HEIGHT_GAIN,
                "stance_gain2": STANCE_HEIGHT_GAIN2,
            }
        )
        self.hip_deviation.func = mdp.hip_deviation_stance_abs
        self.hip_deviation.params.setdefault("action_name", "joints")
        # Keep the extension visible when the stance is active.  The inherited
        # hip_deviation dead band is wide (0.22 rad in the current checkpoint),
        # inside which the policy could otherwise cancel most of a +0.30 offset.
        self.stance_hold = RewTerm(
            func=mdp.stance_hold_abs,
            weight=STANCE_HOLD_WEIGHT,
            params={
                "action_name": "joints",
                "tolerance": STANCE_HOLD_TOL,
                "min_stance": 0.05,
            },
        )


@configclass
class TeleopEnvCfg(ManagerBasedRLEnvCfg):
    """One policy for forward/reverse driving, turning, in-place rotation and stops."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: TeleopObservationsCfg = TeleopObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: TeleopRewardsCfg = TeleopRewardsCfg()
    #: shared terminations + the strict pitch limit from braking
    terminations: BrakingTerminationsCfg = BrakingTerminationsCfg()
    commands: TeleopCommandsCfg = TeleopCommandsCfg()

    joint_action_name: str = "joints"

    def __post_init__(self):
        # simulation settings -- identical to locomotion / turning / braking
        self.decimation = 4  # 50 Hz control on a 200 Hz physics step
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15

        # -------------------------------------------------------------- #
        # GPU stability knobs (same environment variables as locomotion)
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

        _iters = os.environ.get("INFANTRY_RL_PHYSX_POS_ITERS")
        if _iters:
            try:
                self.sim.physx.max_position_iteration_count = int(_iters)
                print(f"[cfg] PhysX position iters: {int(_iters)} (INFANTRY_RL_PHYSX_POS_ITERS)")
            except ValueError:
                pass

        # -------------------------------------------------------------- #
        # The policy must own the differential WHILE TURNING, otherwise a
        # commanded yaw rate would be cancelled.  The conditional PD below takes
        # the differential back only while the command is (nearly) straight.
        # -------------------------------------------------------------- #
        self.actions.joints.heading_control = HEADING_HOLD_ENABLED
        self.actions.joints.conditional_heading_control = HEADING_HOLD_ENABLED
        self.actions.joints.heading_mode = "override"
        self.actions.joints.target_heading = 0.0
        self.actions.joints.heading_hold_command_name = "base_velocity"
        self.actions.joints.heading_hold_wz_threshold = HEADING_HOLD_WZ_THRESHOLD
        self.actions.joints.heading_max_yaw_rate = HEADING_HOLD_MAX_YAW_RATE
        self.actions.joints.heading_hold_max_lateral_accel = HEADING_HOLD_MAX_LATERAL_ACCEL
        self.actions.joints.position_control = False
        # scripted stance bias (A + RB leg extension)
        self.actions.joints.stance_randomize = STANCE_RANDOMIZE
        self.actions.joints.stance_min = STANCE_MIN
        self.actions.joints.stance_max = STANCE_MAX
        self.actions.joints.stance_slew_rate = STANCE_SLEW
        self.actions.joints.stance_gate_vx = STANCE_GATE_VX
        self.actions.joints.stance_gate_wz = STANCE_GATE_WZ
        self.actions.joints.stance_prob_crouch = STANCE_PROB_CROUCH
        self.actions.joints.stance_prob_nominal = STANCE_PROB_NOMINAL
        self.actions.joints.stance_jitter = STANCE_JITTER
        self.actions.joints.hip_symmetry = HIP_SYMMETRY
        self.actions.joints.hip_symmetry_gate = HIP_SYMMETRY_GATE
        # Deep crouches sit below the locomotion fall height; keep the orientation
        #/pitch guards, but let the teleop base go down to 0.075 m.
        self.terminations.base_too_low.params["minimum_height"] = TELEOP_FALL_HEIGHT

        if os.environ.get("INFANTRY_RL_HEADING_CONTROL", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            print(
                "[cfg] WARNING: INFANTRY_RL_HEADING_CONTROL=1 is IGNORED by the teleop "
                "task -- it uses its own CONDITIONAL heading hold, otherwise turning "
                "commands would be cancelled."
            )

        vx_min, vx_max = self.commands.base_velocity.ranges.lin_vel_x
        cmd = self.commands.base_velocity
        # Low-speed mode must respect the active vx range.  A focus stage such as
        # [2.0, 2.5] must NOT silently receive 0.05-0.2 m/s samples.
        low_lo = max(LOW_SPEED_MIN, vx_min)
        low_hi = min(LOW_SPEED_MAX, vx_max)
        if LOW_SPEED_PROB > 0.0 and low_lo < low_hi:
            cmd.low_speed_prob = LOW_SPEED_PROB
            cmd.low_speed_min = low_lo
            cmd.low_speed_max = low_hi
        else:
            cmd.low_speed_prob = 0.0
        print(
            f"[cfg] teleop envelope     : vx [{vx_min:+.2f}, {vx_max:+.2f}] m/s, "
            f"wz +/-{self.commands.base_velocity.ranges.ang_vel_z[1]:.2f} rad/s "
            f"(in-place up to +/-{cmd.turn_in_place_max_abs:.2f})"
        )
        print(
            f"[cfg] teleop schedule     : resample {cmd.resampling_time_range[0]:.1f}-"
            f"{cmd.resampling_time_range[1]:.1f} s, standing "
            f"{cmd.rel_standing_envs:.0%}, forced stop {cmd.brake_prob:.0%} of "
            f"non-standing, in-place {cmd.turn_in_place_prob:.0%} of moving"
        )
        if cmd.low_speed_prob > 0.0:
            print(
                f"[cfg] low-speed creep     : {cmd.low_speed_prob:.0%} of remaining moving "
                f"samples in vx [{cmd.low_speed_min:.2f}, {cmd.low_speed_max:.2f}] m/s "
                "(full yaw range, P2)"
            )
        else:
            print(
                "[cfg] low-speed creep     : DISABLED (low-speed band is outside the "
                f"active vx range [{vx_min:+.2f}, {vx_max:+.2f}] or prob=0)"
            )
        if cmd.max_lateral_accel > 0.0:
            print(
                f"[cfg] lateral accel cap   : |vx*wz| <= {cmd.max_lateral_accel:.2f} m/s^2 "
                "(INFANTRY_RL_MAX_LAT_ACCEL; 0 disables)"
            )
        else:
            print("[cfg] lateral accel cap   : DISABLED (INFANTRY_RL_MAX_LAT_ACCEL=0)")
        if HEADING_HOLD_ENABLED:
            print(
                "[cfg] heading hold         : CONDITIONAL PD, active while "
                f"|wz_cmd| < {HEADING_HOLD_WZ_THRESHOLD:.2f} rad/s "
                f"(kp={self.actions.joints.heading_kp:.2f}, "
                f"kd={self.actions.joints.heading_kd:.2f}, "
                f"clamp +/-{HEADING_HOLD_MAX_YAW_RATE:.2f} rad/s, "
                f"|vx*wz_pd| <= {HEADING_HOLD_MAX_LATERAL_ACCEL:.2f} m/s^2); "
                "the policy owns yaw above the threshold"
            )
        else:
            print(
                "[cfg] heading hold         : DISABLED (INFANTRY_RL_TELEOP_HEADING_HOLD=0) "
                "-- the policy owns the wheel differential in every mode"
            )
        print(
            f"[cfg] brake criterion     : |vx| < {STOP_SPEED_THRESHOLD:.2f} m/s and "
            f"|wz| < {STOP_YAW_THRESHOLD:.2f} rad/s; distance tolerance "
            f"{STOP_DIST_TOLERANCE:.2f} m"
        )
        print(
            f"[cfg] posture             : pitch > {BRAKE_PITCH_TOLERANCE:.2f} rad "
            f"w {BRAKE_PITCH_WEIGHT:+.1f}, rate {BRAKE_PITCH_RATE_WEIGHT:+.2f}, "
            f"symmetric {BRAKE_PITCH_SYM_WEIGHT:+.2f}, hip dev > "
            f"{BRAKE_HIP_DEV_TOL:.2f} rad w {BRAKE_HIP_DEV_WEIGHT:+.1f}; "
            f"pitch termination {BRAKE_PITCH_LIMIT:.2f} rad"
        )
        print(
            f"[cfg] hip action          : scale {self.actions.joints.hip_scale:.2f} rad/unit, "
            f"range +/-{self.actions.joints.hip_range:.2f} rad "
            f"({self.actions.joints.hip_range * 57.2958:.1f} deg)"
        )
        print(
            f"[cfg] hip symmetry        : {HIP_SYMMETRY}"
            + (
                " (right-front leg is the master: targets [+t,-t,+t,-t]; "
                "wheels own pitch balance"
                + (
                    "; GATED to in-place commands, independent hips while driving/turning)"
                    if HIP_SYMMETRY_GATE
                    else "; active for ALL commands)"
                )
                if HIP_SYMMETRY == "master_rf"
                else " (front/back and left/right hips are independent)"
            )
        )
        stance_mode = "RANDOMIZED" if STANCE_RANDOMIZE else "external/gamepad only"
        print(
            f"[cfg] stance bias         : {stance_mode} [{STANCE_MIN:+.2f}, "
            f"{STANCE_MAX:+.2f}] rad, slew {STANCE_SLEW:.2f} rad/s, gate "
            f"|vx| <= {STANCE_GATE_VX:.2f} m/s and |wz| >= {STANCE_GATE_WZ:.2f} rad/s; "
            f"height {STANCE_HEIGHT_GAIN:.3f} d + {STANCE_HEIGHT_GAIN2:.3f} d^2; sample "
            f"{STANCE_PROB_CROUCH:.0%} crouch / {STANCE_PROB_NOMINAL:.0%} nominal / "
            f"{1.0 - STANCE_PROB_CROUCH - STANCE_PROB_NOMINAL:.0%} extend; "
            f"hold {STANCE_HOLD_WEIGHT:+.1f} @ tol {STANCE_HOLD_TOL:.2f}"
        )
        print(
            f"[cfg] teleop fall height  : base_too_low below {TELEOP_FALL_HEIGHT:.3f} m "
            "(deep crouch at d=-0.65 is ~0.078-0.084 m)"
        )
        if STANCE_MAX > 0.45 or STANCE_MIN < -0.60:
            print(
                "[cfg] WARNING: stance range goes beyond the safe envelope "
                "[-0.60, +0.45] rad (crouch approaches the fall height / hip clamp)"
            )
        print(
            f"[cfg] roll penalty        : constant {TELEOP_ROLL_WEIGHT:+.2f}, "
            f"straight-only fades out at |wz| >= {WZ_MAX_RPS:.2f} rad/s"
        )

        self.viewer.origin_type = "world"
        self.viewer.eye = (2.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.2)


@configclass
class TeleopEnvCfg_PLAY(TeleopEnvCfg):
    """Playback variant: a fixed command (and optional stop) owns the task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.turn_in_place_prob = 0.0
        self.commands.base_velocity.brake_prob = 0.0
        # replay / deployment: the stance is driven from outside (gamepad or
        # play.py --stance_bias), never resampled
        self.actions.joints.stance_randomize = False


# --------------------------------------------------------------------------- #
# Environment class
# --------------------------------------------------------------------------- #
class FiveBarTeleopEnv(FiveBarEnv):
    """Five-bar wheel-leg driven by joystick-style ``(vx, wz)`` commands."""

    cfg: TeleopEnvCfg


__all__ = [
    "BRAKE_PROB",
    "HEADING_HOLD_ENABLED",
    "HEADING_HOLD_MAX_LATERAL_ACCEL",
    "HEADING_HOLD_MAX_YAW_RATE",
    "HEADING_HOLD_WZ_THRESHOLD",
    "LOW_SPEED_MAX",
    "LOW_SPEED_MIN",
    "LOW_SPEED_PROB",
    "MAX_LATERAL_ACCEL",
    "REL_STANDING_ENVS",
    "RESAMPLING_TIME_RANGE",
    "TURN_IN_PLACE_MIN_WZ",
    "TURN_IN_PLACE_PROB",
    "TeleopCommandsCfg",
    "TeleopEnvCfg",
    "TeleopEnvCfg_PLAY",
    "TeleopObservationsCfg",
    "TeleopRewardsCfg",
    "FiveBarTeleopEnv",
    "VX_MAX_MPS",
    "VX_MIN_MPS",
    "WZ_INPLACE_MAX_RPS",
    "WZ_MAX_RPS",
]
