"""Self-righting (fall-recovery) task for the locked-squat five-bar wheel-leg.

What this task is for
---------------------
``squat_torque`` only ever starts **upright**: pitch +-1.5 deg, hips already at the
squat pose, knees already closing the loop.  The policy it trains therefore never
sees a fallen robot and cannot get up from one -- and the hardware starts on the
ground.

This task starts the robot **on the ground** at a fore/aft pitch and asks for the
same thing the balance task asks for, from the other side: drive the base back into
the balance envelope (pitch < 6 deg, small body rate, standing height, held 0.1 s),
then stop.  The balance policy takes over from there.

The observation and the action are **unchanged** -- 25 dims per frame, 5-frame
history (125), 4 hip position offsets + 2 wheel velocity targets.  That is a hard
requirement: the final ONNX export and the firmware interface stay identical, and a
``squat_torque`` checkpoint loads into this task without any surgery::

    python train.py --task getup_torque --resume --resume_latest --num_envs 1024

The measured ground geometry (why the reset looks the way it does)
------------------------------------------------------------------
``robot_model.fallen_geometry()`` and ``tools/probe_fallen_geometry.py`` measured
the retracted-leg stance against the real collision meshes:

* 0-17 deg   -> the **wheels** touch first;
* 17-20 deg  -> the chassis corner joins the wheels: a stable **tripod**.  This is
  where the robot actually ends up.  Every start at 20-60 deg that we settled under
  physics fell back to |pitch| = 17 deg within 1.2 s;
* >= 20 deg  -> **only** the chassis corner touches and the wheels are in the air
  (4.8 cm at 35 deg).  The CoM projects behind the corner, so gravity rotates the
  robot back down.

So a 30-40 deg start is a *transient*, not a resting state, and the hard part is
escaping the 17 deg tripod.  The reset therefore mixes both populations
(``rest_probability``): the requested 30-40 deg fall and the measured tripod.  Set
``INFANTRY_RL_GETUP_REST_PROB=0`` to train only from 30-40 deg.

Relaxing the failed-episode limits
----------------------------------
The balance task terminates at 0.24 rad pitch because that is just before the
chassis corner touches.  Here the robot *starts* past that, so the limits move out
to a genuine fall (``FELL_FLAT_DEG``) and the episode ends early only on success.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import mdp
from env_base import FiveBarEnv
from env_squat import (
    DR_COM_XY,
    DR_COM_Z,
    DR_FRICTION_MAX,
    DR_FRICTION_MIN,
    DR_GAIN_MAX,
    DR_GAIN_MIN,
    DR_MASS_MAX,
    DR_MASS_MIN,
    DR_MOTOR_MAX,
    DR_MOTOR_MIN,
    DR_PUSH_INTERVAL_MAX,
    DR_PUSH_INTERVAL_MIN,
    DR_PUSH_VEL,
    DR_RESTITUTION_MAX,
    DR_RESTITUTION_MIN,
    RESET_HIP_JITTER,
    SQUAT_BASE_HEIGHT,
    SQUAT_HIP_OFFSETS,
    SquatCommandsCfg,
    SquatEnvCfg,
    SquatObservationsCfg,
    _env_float,
    reset_squat_pose,
)
from robot_model import HIP_ORDER, KNEE_ORDER, WHEEL_ORDER

# --------------------------------------------------------------------------- #
# Fallen-start geometry
# --------------------------------------------------------------------------- #
#: the requested "fallen" pitch: the unstable 30-40 deg transient, chassis corner on
#: the ground and the wheels 2-5 cm in the air (measured).
FALL_PITCH_MIN_DEG = _env_float("INFANTRY_RL_GETUP_FALL_DEG_MIN", 30.0)
FALL_PITCH_MAX_DEG = _env_float("INFANTRY_RL_GETUP_FALL_DEG_MAX", 40.0)
#: the measured resting tripod: wheels + chassis corner on the ground, |pitch| ~17 deg
REST_PITCH_MIN_DEG = _env_float("INFANTRY_RL_GETUP_REST_DEG_MIN", 15.0)
REST_PITCH_MAX_DEG = _env_float("INFANTRY_RL_GETUP_REST_DEG_MAX", 19.0)
#: fraction of episodes that start in the resting tripod instead of the 30-40 deg fall.
#: Default 1.0: the 30-40 deg start is the *unstable* pose (chassis corner on the
#: ground, wheels 2-5 cm in the air), so tipping back onto the wheels makes the robot
#: visibly bounce -- and the hardware never rests there.  The measured resting state,
#: and therefore the state a fallen robot has to be able to leave, is the 17 deg
#: tripod (wheels + chassis corner on the ground).  Set this below 1.0 only if you
#: also want the 30-40 deg fall transient in the mix, or narrow the tripod band with
#: INFANTRY_RL_GETUP_REST_DEG_MIN/MAX (both = 17 for exactly one angle).
REST_PROBABILITY = _env_float("INFANTRY_RL_GETUP_REST_PROB", 1.0)
#: roll is not part of this task; only a small perturbation so the policy is not blind
GETUP_ROLL_DEG = _env_float("INFANTRY_RL_GETUP_ROLL_DEG", 1.5)
#: how far above the measured contact height the root is written (m)
CONTACT_MARGIN = _env_float("INFANTRY_RL_GETUP_CONTACT_MARGIN", 0.0015)

# --------------------------------------------------------------------------- #
# Success / failure envelope
# --------------------------------------------------------------------------- #
EPISODE_LENGTH_S = _env_float("INFANTRY_RL_GETUP_EPISODE", 4.0)
#: the balance policy's envelope is 0.24 rad; succeed well inside it so the hand-off
#: has margin (firmware switches at |pitch| < 10 deg by default)
SUCCESS_PITCH = math.radians(_env_float("INFANTRY_RL_GETUP_SUCCESS_PITCH_DEG", 6.0))
SUCCESS_RATE = _env_float("INFANTRY_RL_GETUP_SUCCESS_RATE", 0.7)
#: the squat height band.  The upper bound is what forces the legs back to the retracted
#: pose: extending them raises the base to ~0.287 m, and the balance policy cannot take
#: over from there.
SUCCESS_HEIGHT_MIN = _env_float("INFANTRY_RL_GETUP_SUCCESS_HEIGHT", 0.155)
SUCCESS_HEIGHT_MAX = _env_float("INFANTRY_RL_GETUP_SUCCESS_HEIGHT_MAX", 0.200)
SUCCESS_HOLD_STEPS = int(_env_float("INFANTRY_RL_GETUP_HOLD_STEPS", 10.0))
#: a real fall: the chassis is flat on the ground and the wheels are up.  Falls outside
#: the reset envelope but inside this are left to the timeout.
FELL_FLAT_LIMIT = math.radians(_env_float("INFANTRY_RL_GETUP_FELL_FLAT_DEG", 65.0))
ROLL_LIMIT = math.radians(_env_float("INFANTRY_RL_GETUP_ROLL_LIMIT_DEG", 35.0))
#: one-shot terminal bonus for reaching the upright envelope
SUCCESS_BONUS = _env_float("INFANTRY_RL_GETUP_SUCCESS_BONUS", 40.0)

#: pushes are off by default: a recovery episode is 4 s, and an interval push would
#: hit mid-manoeuvre.  Enable it only once the policy recovers reliably.
GETUP_PUSH_VEL = _env_float("INFANTRY_RL_GETUP_PUSH_VEL", 0.0)


# --------------------------------------------------------------------------- #
# MDP config
# --------------------------------------------------------------------------- #
@configclass
class GetupRewardsCfg:
    """Get back upright, without thrashing the motors on the way.

    The dominant shaping terms are ``pitch`` and ``upright``: they must beat the
    ``alive`` income of a robot that just lies in the tripod, otherwise "stay down
    and survive" is a local optimum.  Everything else is the same smoothness /
    closure / posture regularisation the balance task uses, so a checkpoint
    fine-tuned here keeps its balance behaviour.
    """

    # -- getting up ------------------------------------------------------
    alive = RewTerm(func=base_mdp.is_alive, weight=0.5)
    #: sum of the squared horizontal gravity components: 0 upright, 1 on its side,
    #: 2 upside down.  Derivable from the 25-D observation (projected_gravity).
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    #: L1 on |pitch|.  At the 17 deg tripod this is -1.8/step, comfortably above the
    #: +0.5/step `alive`, so the policy cannot sit there and farm the survival term.
    pitch = RewTerm(func=mdp.pitch_abs, weight=-6.0)
    roll = RewTerm(func=mdp.roll_abs, weight=-1.0)
    #: weak: at the tripod the root height is only 7 mm below standing (the root sits
    #: almost on the wheel axle), so this is a sanity term, not the shaping signal.
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-4.0, params={"target_height": SQUAT_BASE_HEIGHT}
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_y = RewTerm(func=mdp.ang_vel_y_l2, weight=-0.2)
    ang_vel_x = RewTerm(func=mdp.ang_vel_x_l2, weight=-0.05)
    yaw_rate = RewTerm(func=mdp.yaw_rate_l2, weight=-0.25)
    #: one-shot bonus, paid on the step ``upright_stable`` terminates the episode
    success = RewTerm(func=mdp.recovery_success, weight=SUCCESS_BONUS)

    # -- smooth actuation ------------------------------------------------
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[*HIP_ORDER, *WHEEL_ORDER])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    #: a swing-up needs wheel spin; only the *endless* spin is charged
    wheel_spin = RewTerm(func=mdp.wheel_spin_l2, weight=-3.0e-4, params={"action_name": "joints"})
    wheel_axle = RewTerm(
        func=mdp.leg_axle_skew_abs, weight=-2.0, params={"action_name": "joints"}
    )
    action_jerk = RewTerm(
        func=mdp.action_jerk_l2, weight=-5.0e-3, params={"action_name": "joints"}
    )
    hip_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(HIP_ORDER))},
    )

    # -- keep the closed chain closed and the hips on the reference pose --
    closure_residual = RewTerm(
        func=mdp.closure_residual_l2, weight=-2.0e3, params={"action_name": "joints"}
    )
    hip_lock = RewTerm(
        func=mdp.hip_lock_deviation_abs,
        weight=-0.1,
        params={"action_name": "joints", "tolerance": 0.05},
    )

    # -- the chassis is meant to leave the ground ------------------------
    #: at the tripod the corner carries ~10% of the weight, so this is a -4/step
    #: push to lift it -- softer than the balance task's -20, because here the corner
    #: contact is the *start* state, not a failure.
    base_contact = RewTerm(
        func=base_mdp.undesired_contacts,
        weight=-4.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces")},
    )


@configclass
class GetupTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    #: the episode ends *successfully* here; the bonus is paid by GetupRewardsCfg.success
    upright_enough = DoneTerm(
        func=mdp.upright_stable,
        params={
            "pitch_limit": float(SUCCESS_PITCH),
            "roll_limit": 0.15,
            "height_min": float(SUCCESS_HEIGHT_MIN),
            "height_max": float(SUCCESS_HEIGHT_MAX),
            "rate_limit": float(SUCCESS_RATE),
            "hold_steps": int(SUCCESS_HOLD_STEPS),
        },
    )
    #: a genuine fall: flat on the chassis, wheels in the air
    fell_flat = DoneTerm(func=mdp.pitch_above, params={"limit_angle": float(FELL_FLAT_LIMIT)})
    roll_limit = DoneTerm(func=mdp.roll_above, params={"limit_angle": float(ROLL_LIMIT)})


@configclass
class GetupEventCfg:
    """Start on the ground, plus the same startup randomizations as the balance task."""

    # -- startup domain randomization (identical to ``squat_torque``) -----
    randomize_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (DR_FRICTION_MIN, DR_FRICTION_MAX),
            "dynamic_friction_range": (DR_FRICTION_MIN, DR_FRICTION_MAX),
            "restitution_range": (DR_RESTITUTION_MIN, DR_RESTITUTION_MAX),
            "num_buckets": 64,
            "make_consistent": True,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    randomize_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "mass_distribution_params": (DR_MASS_MIN, DR_MASS_MAX),
            "operation": "add",
            "recompute_inertia": True,
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )
    randomize_base_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "com_range": {
                "x": (-DR_COM_XY, DR_COM_XY),
                "y": (-DR_COM_XY, DR_COM_XY),
                "z": (-DR_COM_Z, DR_COM_Z),
            },
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )
    randomize_knee_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(KNEE_ORDER)),
            "stiffness_distribution_params": (DR_GAIN_MIN, DR_GAIN_MAX),
            "damping_distribution_params": (DR_GAIN_MIN, DR_GAIN_MAX),
            "operation": "scale",
        },
    )
    #: off by default (see GETUP_PUSH_VEL); an interval push mid-manoeuvre is noise
    push_robots = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(DR_PUSH_INTERVAL_MIN, DR_PUSH_INTERVAL_MAX),
        params={
            "velocity_range": {
                "x": (-GETUP_PUSH_VEL, GETUP_PUSH_VEL),
                "y": (-GETUP_PUSH_VEL, GETUP_PUSH_VEL),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # -- per-episode reset ------------------------------------------------
    #: written FIRST: the root pose, at the measured ground contact height
    reset_root = EventTerm(
        func=mdp.reset_root_state_fallen,
        mode="reset",
        params={
            "fall_pitch_range_deg": (FALL_PITCH_MIN_DEG, FALL_PITCH_MAX_DEG),
            "rest_pitch_range_deg": (REST_PITCH_MIN_DEG, REST_PITCH_MAX_DEG),
            "rest_probability": REST_PROBABILITY,
            "roll_range_deg": (-GETUP_ROLL_DEG, GETUP_ROLL_DEG),
            "yaw_range_deg": (-180.0, 180.0),
            "contact_margin": CONTACT_MARGIN,
            "lin_vel_range": (-0.03, 0.03),
            "ang_vel_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    #: written SECOND: hips + closed-chain knees, exactly as in the balance task
    reset_pose = EventTerm(
        func=reset_squat_pose,
        mode="reset",
        params={
            "hip_offsets": SQUAT_HIP_OFFSETS,
            "jitter": RESET_HIP_JITTER,
            "velocity_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class GetupEnvCfg(SquatEnvCfg):
    """Same robot, same action, same observation -- different start and goal."""

    observations: SquatObservationsCfg = SquatObservationsCfg()
    commands: SquatCommandsCfg = SquatCommandsCfg()
    rewards: GetupRewardsCfg = GetupRewardsCfg()
    terminations: GetupTerminationsCfg = GetupTerminationsCfg()
    events: GetupEventCfg = GetupEventCfg()

    def __post_init__(self):
        # keeps the sim / action-term / randomization setup of the balance task
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S
        print(
            f"[cfg] getup reset         : {100.0 * REST_PROBABILITY:.0f}% tripod "
            f"{REST_PITCH_MIN_DEG:.0f}-{REST_PITCH_MAX_DEG:.0f} deg, "
            f"{100.0 * (1.0 - REST_PROBABILITY):.0f}% fallen "
            f"{FALL_PITCH_MIN_DEG:.0f}-{FALL_PITCH_MAX_DEG:.0f} deg, roll "
            f"+-{GETUP_ROLL_DEG:g} deg, on the measured contact height "
            f"(+{1000.0 * CONTACT_MARGIN:.1f} mm)"
        )
        print(
            f"[cfg] getup success       : |pitch| < {float(SUCCESS_PITCH) * 57.29577951308232:.1f} deg, "
            f"|rate| < {SUCCESS_RATE:g} rad/s, base_z in "
            f"[{SUCCESS_HEIGHT_MIN:.3f}, {SUCCESS_HEIGHT_MAX:.3f}] m, "
            f"held {SUCCESS_HOLD_STEPS} steps; bonus {SUCCESS_BONUS:g}"
        )
        print(
            f"[cfg] getup limits        : fell flat at {float(FELL_FLAT_LIMIT) * 57.29577951308232:.0f} deg "
            f"pitch / {float(ROLL_LIMIT) * 57.29577951308232:.0f} deg roll, episode "
            f"{EPISODE_LENGTH_S:.1f} s, push {GETUP_PUSH_VEL:g} m/s"
        )


@configclass
class GetupEnvCfg_PLAY(GetupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6
        self.observations.policy.enable_corruption = False


class FiveBarGetupEnv(FiveBarEnv):
    """Fall recovery with the balance task's observation and action interface."""

    cfg: GetupEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        #: per-episode success flag, set by ``mdp.upright_stable``, read by the reward
        self.getup_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #: consecutive upright steps, see ``mdp.upright_stable``
        self._getup_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        #: the pitch each environment was actually reset at (deg, signed)
        self.reset_pitch_deg = torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # episode_length_buf is zeroed by the base reset, so snapshot it first
        episode_steps = self.episode_length_buf[ids].float().clone()
        success = getattr(self, "getup_success", None)
        success = success[ids].clone() if success is not None else torch.zeros(ids.numel(), dtype=torch.bool)
        pitch = getattr(self, "reset_pitch_deg", None)
        pitch = pitch[ids].clone() if pitch is not None else torch.zeros(ids.numel(), device=self.device)

        super()._reset_idx(env_ids)

        if hasattr(self, "getup_success"):
            self.getup_success[ids] = False
            self._getup_hold[ids] = 0
        if ids.numel() == 0:
            return

        log = self.extras.setdefault("log", {})
        success_f = success.float()
        log["Metrics/recovery/success"] = float(success_f.mean().item())
        log["Metrics/recovery/episode_steps_mean"] = float(episode_steps.mean().item())
        log["Metrics/recovery/reset_pitch_deg"] = float(pitch.abs().mean().item())
        #: fraction of the resets that came from the 30-40 deg fall (the rest: tripod)
        split = 0.5 * (REST_PITCH_MAX_DEG + FALL_PITCH_MIN_DEG)
        log["Metrics/recovery/reset_fallen_pct"] = float(
            (pitch.abs() > math.radians(split)).float().mean().item()
        )
        if bool(success.any()):
            log["Metrics/recovery/steps_to_upright"] = float(episode_steps[success].mean().item())


__all__ = [
    "CONTACT_MARGIN",
    "EPISODE_LENGTH_S",
    "FALL_PITCH_MAX_DEG",
    "FALL_PITCH_MIN_DEG",
    "FiveBarGetupEnv",
    "GETUP_PUSH_VEL",
    "GetupEnvCfg",
    "GetupEnvCfg_PLAY",
    "GetupEventCfg",
    "GetupRewardsCfg",
    "GetupTerminationsCfg",
    "REST_PROBABILITY",
    "REST_PITCH_MAX_DEG",
    "REST_PITCH_MIN_DEG",
    "SUCCESS_BONUS",
    "SUCCESS_PITCH",
    "SUCCESS_HEIGHT_MAX",
    "SUCCESS_HEIGHT_MIN",
]
