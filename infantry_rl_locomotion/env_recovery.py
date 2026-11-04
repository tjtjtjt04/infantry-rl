"""Fall-recovery environment: right itself after a forward/backward fall.

Scope decided with the user: **only front/back falls**, no side falls.  The robot is
reset in a random pitched pose (pitch magnitude 45-120 deg, random sign, roll within
+/-8 deg) and must get back upright and still.  It uses the same 35-dim observation
and 6-dim action as the teleop task, so a teleop checkpoint can warm-start it and a
deployment can switch between the two policies without changing the interface.

What is different from driving:

* the command is a constant zero (the 2 command dims are still present);
* the ordinary ``bad_orientation`` / ``base_too_low`` terminations are NOT used --
  the robot starts fallen and is allowed to lie on the ground;
* success is ``upright and still for hold_time``; that term resets the episode;
* the reward uses an explicit uprightness term (upright = 1, upside down = 0), not
  the driving velocity-tracking terms.  A zero-command tracking reward would pay a
  fallen robot for being still, which is the last thing recovery should learn.

Only forward/backward falls are sampled.  The roll range is small by design.
"""

from __future__ import annotations

import os

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import mdp
from config import HIP_JOINT_REGEX
from env_locomotion import (
    ActionsCfg,
    FiveBarEnv,
    LocomotionObservationsCfg,
    NOMINAL_HEIGHT,
    RobotSceneCfg,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        print(f"[cfg] WARNING: {name}={os.environ.get(name)!r} is not a number; using {default}")
        return float(default)


# --------------------------------------------------------------------------- #
# Fall envelope
# --------------------------------------------------------------------------- #
#: pitch magnitude range (degrees); the sign is sampled per environment, so both
#: forward and backward falls occur.  Start easier and widen in a second stage.
PITCH_MIN_DEG = _env_float("INFANTRY_RL_RECOVERY_PITCH_MIN", 45.0)
PITCH_MAX_DEG = _env_float("INFANTRY_RL_RECOVERY_PITCH_MAX", 120.0)
#: roll is deliberately small: this task does not train side falls
ROLL_MAX_DEG = _env_float("INFANTRY_RL_RECOVERY_ROLL_MAX", 8.0)
#: lift above the default root height at reset (m); the fallen body is dropped
#: onto the ground and settles during the first control steps
Z_OFFSET_MIN = _env_float("INFANTRY_RL_RECOVERY_Z_MIN", 0.05)
Z_OFFSET_MAX = _env_float("INFANTRY_RL_RECOVERY_Z_MAX", 0.15)
#: 0 = random sign, +1 = force forward, -1 = force backward (evaluation)
PITCH_SIGN = _env_float("INFANTRY_RL_RECOVERY_PITCH_SIGN", 0.0)

#: fraction of resets that start in the deepest symmetric squat.  The measured
#: five-bar pose front=-0.45 / back=+0.45 has L0=0.165 m (nominal 0.250 m) and the
#: wheel axle directly under the hip midpoint: the "tuck, drive the wheels, extend"
#: strategy.  Curriculum: start at 1.0, then lower it so arbitrary hip poses appear.
TUCK_PROB = _env_float("INFANTRY_RL_RECOVERY_TUCK_PROB", 0.50)
TUCK_FRONT_OFFSET = _env_float("INFANTRY_RL_RECOVERY_TUCK_FRONT", -0.45)
TUCK_BACK_OFFSET = _env_float("INFANTRY_RL_RECOVERY_TUCK_BACK", 0.45)
TUCK_JITTER = _env_float("INFANTRY_RL_RECOVERY_TUCK_JITTER", 0.03)

#: episode length and success hold time (s)
EPISODE_LENGTH_S = _env_float("INFANTRY_RL_RECOVERY_EPISODE", 12.0)
HOLD_TIME_S = _env_float("INFANTRY_RL_RECOVERY_HOLD", 0.5)
#: strong side-roll guard: side falls are out of scope, and without this the
#: policy converges to a stable side-lying pose instead of getting up
ROLL_LIMIT_RAD = _env_float("INFANTRY_RL_RECOVERY_ROLL_LIMIT", 1.05)
ROLL_WEIGHT = _env_float("INFANTRY_RL_RECOVERY_ROLL_WEIGHT", -3.5)
ROLL_FAIL_WEIGHT = _env_float("INFANTRY_RL_RECOVERY_ROLL_FAIL_WEIGHT", -20.0)
TUCK_WEIGHT = _env_float("INFANTRY_RL_RECOVERY_TUCK_WEIGHT", 2.0)
UPRIGHT_POWER = _env_float("INFANTRY_RL_RECOVERY_UPRIGHT_POWER", 3.0)
#: one-shot payoff for actually completing the upright-and-still hold.  Without it
#: PPO learned to hover at uprightness ~0.90, collecting the per-step reward
#: forever (timeout) instead of crossing the success threshold.
RECOVERED_BONUS_WEIGHT = _env_float("INFANTRY_RL_RECOVERY_BONUS_WEIGHT", 100.0)
UPRIGHT_ZONE_MIN = _env_float("INFANTRY_RL_RECOVERY_ZONE_MIN", 0.95)

#: success thresholds
STOP_LIN_VEL = _env_float("INFANTRY_RL_RECOVERY_STOP_LIN", 0.20)
STOP_ANG_VEL = _env_float("INFANTRY_RL_RECOVERY_STOP_ANG", 0.50)
#: uprightness threshold for the still/success check.  0.95 means tilt < ~25.8 deg;
#: 0.90 (tilt < 36.9 deg) was lenient enough that the policy could "succeed" while
#: leaning visibly.
UPRIGHT_MIN = _env_float("INFANTRY_RL_RECOVERY_UPRIGHT_MIN", 0.95)
#: minimum root height for the still/success check (m).  The nominal standing height
#: is ~0.190 m; the deepest squat is ~0.10 m.  Without this gate the policy counted
#: a deep squat as "recovered" and never extended its legs.
MIN_HEIGHT = _env_float("INFANTRY_RL_RECOVERY_MIN_HEIGHT", 0.16)

if PITCH_MIN_DEG >= PITCH_MAX_DEG:
    raise ValueError(
        f"recovery pitch range is empty: [{PITCH_MIN_DEG}, {PITCH_MAX_DEG}] degrees"
    )
if Z_OFFSET_MIN >= Z_OFFSET_MAX:
    raise ValueError(f"recovery z offset range is empty: [{Z_OFFSET_MIN}, {Z_OFFSET_MAX}]")


# --------------------------------------------------------------------------- #
# Reset / commands / rewards / terminations
# --------------------------------------------------------------------------- #
@configclass
class RecoveryEventCfg:
    """Drop the robot into a random front/back fallen pose."""

    reset_root = EventTerm(
        func=mdp.reset_root_state_front_back_fallen,
        mode="reset",
        params={
            "pitch_range_deg": (PITCH_MIN_DEG, PITCH_MAX_DEG),
            "roll_range_deg": (-ROLL_MAX_DEG, ROLL_MAX_DEG),
            "yaw_range_deg": (-180.0, 180.0),
            "z_offset_range": (Z_OFFSET_MIN, Z_OFFSET_MAX),
            "lin_vel_range": (-0.10, 0.10),
            "ang_vel_range": (-0.30, 0.30),
            "pitch_sign": PITCH_SIGN,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    #: the deepest symmetric squat for a fraction of resets; otherwise randomized.
    #: The passive knees are re-solved by the action term after the reset.
    reset_hips = EventTerm(
        func=mdp.reset_hips_tucked,
        mode="reset",
        params={
            "tuck_prob": TUCK_PROB,
            "tuck_front_offset": TUCK_FRONT_OFFSET,
            "tuck_back_offset": TUCK_BACK_OFFSET,
            "tuck_jitter": TUCK_JITTER,
            "random_offset": 0.15,
            "velocity_range": (-1.0, 1.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(HIP_JOINT_REGEX)),
        },
    )


@configclass
class RecoveryCommandsCfg:
    """A constant zero command keeps the 35-dim observation interface unchanged."""

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        rel_standing_envs=1.0,
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class RecoveryRewardsCfg:
    """Uprightness + staying upright; no velocity-tracking rewards.

    The uprightness term is the main signal: it is 1 upright, 0 upside down and
    0.5 at 90 deg, so it has a gradient through the whole fall range.  The still
    bonus pays for holding the recovered pose.  Leg/loop/action terms keep the
    five-bar closed and the motion smooth, but wheel torque is not penalised (the
    wheels are a legitimate recovery actuator).
    """

    alive = RewTerm(func=base_mdp.is_alive, weight=0.5)
    #: cubic uprightness: the linear version is already 0.85 at 46 deg, which let
    #: the first recovery run settle at a comfortable lean instead of standing up
    upright = RewTerm(
        func=mdp.uprightness_power,
        weight=10.0,
        params={"power": UPRIGHT_POWER},
    )
    #: bounded near-upright bonus, paid even while the robot is still moving
    upright_zone = RewTerm(
        func=mdp.upright_zone,
        weight=3.0,
        params={"min_upright": UPRIGHT_ZONE_MIN, "min_height": MIN_HEIGHT},
    )
    #: IMPORTANT ORDER: upright_still updates the hold counter / success flag that
    #: recovered_bonus reads in the SAME step, so it must be defined first.
    upright_still = RewTerm(
        func=mdp.upright_still,
        weight=2.0,
        params={
            "lin_threshold": STOP_LIN_VEL,
            "ang_threshold": STOP_ANG_VEL,
            "min_upright": UPRIGHT_MIN,
            "min_height": MIN_HEIGHT,
            "hold_time": HOLD_TIME_S,
        },
    )
    #: one-shot payoff the first time the hold counter reaches the success length;
    #: the episode does NOT terminate, so the policy can keep standing afterwards
    recovered_bonus = RewTerm(
        func=mdp.recovered_bonus,
        weight=RECOVERED_BONUS_WEIGHT,
        params={"hold_time": HOLD_TIME_S},
    )
    tuck = RewTerm(
        func=mdp.tuck_match,
        weight=TUCK_WEIGHT,
        params={
            "action_name": "joints",
            "front_tuck": TUCK_FRONT_OFFSET,
            "back_tuck": TUCK_BACK_OFFSET,
            "upright_max": 0.90,
            "sigma": 0.15,
        },
    )
    #: one-step penalty for ending the episode by rolling onto the side
    roll_fail = RewTerm(
        func=mdp.roll_fail_penalty,
        weight=ROLL_FAIL_WEIGHT,
        params={"limit_angle": ROLL_LIMIT_RAD},
    )
    #: dense final extension: while upright but still below MIN_HEIGHT, pull the
    #: hips from the tuck (front -0.45 / back +0.45) back to the nominal stance.
    #: Without it the policy stops in a deep squat and never lifts the body.
    leg_extension = RewTerm(
        func=mdp.leg_extension_abs,
        weight=-5.0,
        params={
            "action_name": "joints",
            "upright_min": 0.90,
            "max_height": MIN_HEIGHT,
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_abs,
        weight=-10.0,
        params={"target_height": NOMINAL_HEIGHT},
    )
    #: side falls are out of scope; make rolling onto the side expensive
    roll = RewTerm(func=mdp.roll_abs, weight=ROLL_WEIGHT)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    lin_vel_y = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.3)
    ang_vel_x = RewTerm(func=mdp.ang_vel_x_l2, weight=-0.05)
    ang_vel_y = RewTerm(func=mdp.ang_vel_y_l2, weight=-0.05)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-3.0e-3)
    action_jerk = RewTerm(func=mdp.action_jerk_l2, weight=-0.05, params={"action_name": "joints"})
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
    wheel_spin = RewTerm(func=mdp.wheel_spin_l2, weight=-1.0e-5, params={"action_name": "joints"})
    hip_deviation = RewTerm(
        func=mdp.hip_deviation_abs,
        weight=-0.5,
        params={"action_name": "joints", "tolerance": 0.25},
    )

    #: the five-bar loop must stay closed even while recovering
    closure_residual = RewTerm(
        func=mdp.closure_residual_l2, weight=-1.0e3, params={"action_name": "joints"}
    )


@configclass
class RecoveryTerminationsCfg:
    """Timeout plus upright-and-still success; ordinary fall terminations are absent."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    #: side fall guard: terminate before a stable side-lying pose can be learned
    roll_limit = DoneTerm(
        func=mdp.roll_above,
        params={"limit_angle": ROLL_LIMIT_RAD},
    )
    #: NOTE: there is deliberately NO "recovered" termination.  The episode runs
    #: to its 12 s timeout and the success is a metric + one-shot reward.  Early
    #: termination made "hover just below upright" more profitable than finishing.


@configclass
class RecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Front/back fall recovery with the same 35/6 interface as teleop."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: LocomotionObservationsCfg = LocomotionObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RecoveryRewardsCfg = RecoveryRewardsCfg()
    terminations: RecoveryTerminationsCfg = RecoveryTerminationsCfg()
    commands: RecoveryCommandsCfg = RecoveryCommandsCfg()
    events: RecoveryEventCfg = RecoveryEventCfg()

    joint_action_name: str = "joints"

    def __post_init__(self):
        self.decimation = 4  # 50 Hz control on a 200 Hz physics step
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

        # Recovery has no velocity command: the heading PD must not fight the
        # recovering motion, and the position loop is irrelevant.
        self.actions.joints.heading_control = False
        self.actions.joints.conditional_heading_control = False
        self.actions.joints.position_control = False
        # Front/back recovery only: remove the left/right asymmetry and the wheel
        # differential, which are the actuators that roll the body onto its side.
        self.actions.joints.sagittal_symmetry = True
        # Zero action = the deepest squat (front -0.45 / back +0.45).  The policy
        # starts from the tuck and learns to extend; it does not have to discover
        # the squat by itself.
        self.actions.joints.hip_bias = (
            TUCK_FRONT_OFFSET,
            TUCK_BACK_OFFSET,
            TUCK_FRONT_OFFSET,
            TUCK_BACK_OFFSET,
        )
        #: Recovery is the fall -> stable SQUAT phase.  The hip tuck bias is
        #: CONSTANT here: the validated v5 checkpoint was trained this way.  The
        #: final "extend the legs" phase is the separate `--task stand` controller,
        #: which uses zero hip bias and the teleop checkpoint.
        self.actions.joints.dynamic_hip_bias = False

        print(
            f"[cfg] recovery envelope   : pitch {PITCH_MIN_DEG:.0f}-{PITCH_MAX_DEG:.0f} deg, "
            f"random sign; roll +/-{ROLL_MAX_DEG:.0f} deg; z offset "
            f"{Z_OFFSET_MIN:.2f}-{Z_OFFSET_MAX:.2f} m"
        )
        print(
            f"[cfg] recovery episode    : {EPISODE_LENGTH_S:.1f} s; success = upright "
            f"({UPRIGHT_MIN:.2f}) + height > {MIN_HEIGHT:.2f} m + still "
            f"(<{STOP_LIN_VEL:.2f} m/s, <{STOP_ANG_VEL:.2f} rad/s) for {HOLD_TIME_S:.2f} s"
        )
        print(
            f"[cfg] recovery tuck       : {TUCK_PROB:.0%} of resets use front "
            f"{TUCK_FRONT_OFFSET:+.2f} / back {TUCK_BACK_OFFSET:+.2f} rad "
            f"(L0 0.165 m, wheel under hip); reward weight {TUCK_WEIGHT:+.2f}; "
            "zero action = this squat (constant bias; stand-up is --task stand)"
        )
        print(
            f"[cfg] recovery roll guard : |roll| > {ROLL_LIMIT_RAD:.2f} rad "
            f"({ROLL_LIMIT_RAD * 57.2958:.1f} deg) terminates; per-step weight "
            f"{ROLL_WEIGHT:+.1f}; terminal penalty {ROLL_FAIL_WEIGHT:+.1f}"
        )
        print(
            "[cfg] recovery symmetry   : left/right hips averaged, wheel differential "
            "= 0 (sagittal-plane recovery only)"
        )
        print(
            f"[cfg] recovery upright    : uprightness^{UPRIGHT_POWER:.1f} weight 10, "
            f"near-upright zone (>{UPRIGHT_ZONE_MIN:.2f}) weight 3, "
            f"success bonus {RECOVERED_BONUS_WEIGHT:+.0f}"
        )
        print(
            f"[cfg] recovery extension  : upright + height < {MIN_HEIGHT:.2f} m "
            "-> hip-extension penalty -5 (pushes the tuck back to nominal)"
        )
        print(
            "[cfg] recovery command    : constant zero (the 35-dim observation is unchanged)"
        )

        self.viewer.origin_type = "world"
        self.viewer.eye = (2.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.2)


@configclass
class RecoveryEnvCfg_PLAY(RecoveryEnvCfg):
    """Playback: never auto-reset on recovery, so the final standing pose is visible."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6
        self.observations.policy.enable_corruption = False
        # Show the tuck-then-extend strategy in replay; the training curriculum
        # lowers tuck_prob via INFANTRY_RL_RECOVERY_TUCK_PROB.
        self.events.reset_hips.params["tuck_prob"] = 1.0


# --------------------------------------------------------------------------- #
# Environment class
# --------------------------------------------------------------------------- #
class FiveBarRecoveryEnv(FiveBarEnv):
    """Five-bar wheel-leg with a randomized front/back fallen reset."""

    cfg: RecoveryEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        #: consecutive control steps spent upright and still; maintained by the
        #: upright_still reward term (not by a termination)
        self.recovery_upright_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        #: True if the robot held upright+still for the success hold during this episode
        self.recovery_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # log the per-episode success rate before clearing the flags
        self.extras.setdefault("log", {})["Metrics/recovery/success"] = float(
            self.recovery_success[env_ids].float().mean().item()
        )
        self.recovery_upright_steps[env_ids] = 0
        self.recovery_success[env_ids] = False


__all__ = [
    "EPISODE_LENGTH_S",
    "HOLD_TIME_S",
    "MIN_HEIGHT",
    "PITCH_MAX_DEG",
    "PITCH_MIN_DEG",
    "PITCH_SIGN",
    "ROLL_FAIL_WEIGHT",
    "ROLL_LIMIT_RAD",
    "ROLL_MAX_DEG",
    "ROLL_WEIGHT",
    "STOP_ANG_VEL",
    "STOP_LIN_VEL",
    "FiveBarRecoveryEnv",
    "RecoveryCommandsCfg",
    "RecoveryEnvCfg",
    "RecoveryEnvCfg_PLAY",
    "RecoveryEventCfg",
    "RecoveryRewardsCfg",
    "RECOVERED_BONUS_WEIGHT",
    "RecoveryTerminationsCfg",
    "TUCK_BACK_OFFSET",
    "TUCK_FRONT_OFFSET",
    "TUCK_PROB",
    "TUCK_WEIGHT",
    "UPRIGHT_MIN",
    "UPRIGHT_POWER",
    "UPRIGHT_ZONE_MIN",
    "Z_OFFSET_MAX",
    "Z_OFFSET_MIN",
]
