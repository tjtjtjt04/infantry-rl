"""Pure-torque squat-pose balance task for the five-bar wheel-leg.

This is the only task.  The 4 hips are torque controlled, the 2 wheels are
position controlled (the action is a relative wheel position increment); the 4
knees stay position-controlled five-bar closed-chain followers (they are passive
on the real robot, but PhysX needs a position target to emulate the mechanical
loop).

The user's training recipe: lock the four hips at the mechanical extreme of the
closed-chain workspace, so the base sits as low as the linkage allows, then learn
to balance on that stance.

Why this pose
-------------
The five-bar leg has two actuated hips per side (``front_hip`` / ``back_hip``) and a
closed chain.  A dense scan over the URDF hip limits, keeping every solved knee
inside its URDF limit with the action term's 0.01 rad safety margin, gives the
minimum leg length

    front_hip = -0.1630 rad, back_hip = +0.1641 rad
    L0        = 0.13637 m          (nominal 0.24977 m)
    base_z    = 0.07637 m          (nominal 0.18977 m)

i.e. 11.3 cm below the nominal stance and 2.9 cm below the existing recovery tuck
(hips +/-0.45, L0 = 0.165 m, base_z ~0.105 m).  Pushing the hips all the way to
their URDF stops only buys another 0.9 mm of height and starts clipping the knee
targets against their own limits, so this is the deepest *feasible closed-chain*
pose.  The closure Jacobian there is well conditioned (condition number 1.37), so
the mechanism is not near a singularity.

The reset starts upright at that height with the hips already at the reference pose
and the knees at the matching closed-chain angles.  Falling is detected by the usual
orientation limit plus a much lower height floor (the old 0.11 m floor would
terminate the squat immediately).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import mdp
from config import HIP_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT
from env_base import (
    ActionsCfg,
    FiveBarEnv,
    NOMINAL_HEIGHT,
    RobotSceneCfg,
)
from robot_model import (
    HIP_ORDER,
    KNEE_ORDER,
    WHEEL_ORDER,
    WHEEL_RADIUS,
    hip_axis_height,
    nominal_pose,
    squat_pose,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        print(f"[cfg] WARNING: {name}={os.environ.get(name)!r} is not a number; using {default}")
        return float(default)


# --------------------------------------------------------------------------- #
# The locked squat pose
# --------------------------------------------------------------------------- #
#: Nominal measured stance (both sides).  ``chassis-rl`` mirrors its right-side
#: joint signs, so the nominal hip values are NOT identical on the two sides.
_NOMINAL = nominal_pose()

#: Deepest feasible reference pose, written by ``tools/measure_chassis_asset.py``
#: to ``robot_data/chassis-rl/squat_pose.json``.
_SQUAT = squat_pose()
_SQUAT_HIPS = _SQUAT["hips"]
#: right-side joint values are ``sign * left`` for the same physical pose
_RIGHT_HIP_SIGN = float(_SQUAT.get("right_hip_sign", -1.0))

#: Absolute hip angles of the deepest feasible squat, left side (rad, URDF zero pose).
SQUAT_FRONT_HIP = _env_float("INFANTRY_RL_SQUAT_FRONT_HIP", float(_SQUAT_HIPS["left"]["front_hip"]))
SQUAT_BACK_HIP = _env_float("INFANTRY_RL_SQUAT_BACK_HIP", float(_SQUAT_HIPS["left"]["back_hip"]))
_FRONT_OVERRIDE = "INFANTRY_RL_SQUAT_FRONT_HIP" in os.environ
_BACK_OVERRIDE = "INFANTRY_RL_SQUAT_BACK_HIP" in os.environ
SQUAT_RIGHT_FRONT_HIP = (
    _RIGHT_HIP_SIGN * SQUAT_FRONT_HIP
    if _FRONT_OVERRIDE
    else float(_SQUAT_HIPS["right"]["front_hip"])
)
SQUAT_RIGHT_BACK_HIP = (
    _RIGHT_HIP_SIGN * SQUAT_BACK_HIP
    if _BACK_OVERRIDE
    else float(_SQUAT_HIPS["right"]["back_hip"])
)

#: Offsets from the nominal stance, in HIP_ORDER = (lf, lb, rf, rb).
SQUAT_FRONT_HIP_OFFSET = SQUAT_FRONT_HIP - float(_NOMINAL["left"]["front_hip"])
SQUAT_BACK_HIP_OFFSET = SQUAT_BACK_HIP - float(_NOMINAL["left"]["back_hip"])
SQUAT_HIP_OFFSETS: tuple[float, float, float, float] = (
    SQUAT_FRONT_HIP_OFFSET,
    SQUAT_BACK_HIP_OFFSET,
    SQUAT_RIGHT_FRONT_HIP - float(_NOMINAL["right"]["front_hip"]),
    SQUAT_RIGHT_BACK_HIP - float(_NOMINAL["right"]["back_hip"]),
)

#: Leg length and root height of that pose.
SQUAT_LEG_LENGTH = _env_float("INFANTRY_RL_SQUAT_LEG_LENGTH", float(_SQUAT["leg_length"]))
SQUAT_BASE_HEIGHT = _env_float(
    "INFANTRY_RL_SQUAT_BASE_HEIGHT",
    float(_SQUAT.get("base_height", WHEEL_RADIUS + SQUAT_LEG_LENGTH - hip_axis_height())),
)
#: Reset offset from the default root height (NOMINAL_HEIGHT) down to the squat.
SQUAT_Z_OFFSET = SQUAT_BASE_HEIGHT - NOMINAL_HEIGHT

EPISODE_LENGTH_S = _env_float("INFANTRY_RL_SQUAT_EPISODE", 10.0)
BASE_FLOOR = _env_float("INFANTRY_RL_SQUAT_BASE_FLOOR", float(_SQUAT.get("base_floor", 0.04)))

# --------------------------------------------------------------------------- #
# Reference wheel_legged joint-space action
# --------------------------------------------------------------------------- #
#: action scales: 0.5 rad per hip action unit, 10 rad/s per wheel action unit
HIP_POS_SCALE = _env_float("INFANTRY_RL_SQUAT_HIP_POS_SCALE", 0.5)
WHEEL_VEL_SCALE = _env_float("INFANTRY_RL_SQUAT_WHEEL_VEL_SCALE", 10.0)
#: explicit low-level PD: tau = kp*(q_ref-q) - kd*qdot ; wheel tau = kd*(v_ref-v)
HIP_KP = _env_float("INFANTRY_RL_SQUAT_HIP_KP", 40.0)
HIP_KD = _env_float("INFANTRY_RL_SQUAT_HIP_KD", 1.0)
WHEEL_KD = _env_float("INFANTRY_RL_SQUAT_WHEEL_KD", 0.5)
TORQUE_HIP_EFFORT_LIMIT = _env_float("INFANTRY_RL_SQUAT_HIP_EFFORT_LIMIT", HIP_EFFORT_LIMIT)
TORQUE_WHEEL_EFFORT_LIMIT = _env_float("INFANTRY_RL_SQUAT_WHEEL_EFFORT_LIMIT", WHEEL_EFFORT_LIMIT)
#: a free-torque hip has no PD to catch the initial pitch, so the first episodes
#: must not start at the corner limit
TORQUE_RESET_PITCH_DEG = _env_float("INFANTRY_RL_SQUAT_TORQUE_RESET_PITCH_DEG", 1.5)
TORQUE_RESET_ROLL_DEG = _env_float("INFANTRY_RL_SQUAT_TORQUE_RESET_ROLL_DEG", 0.75)

# --------------------------------------------------------------------------- #
# Observation noise / joint-velocity estimation
# --------------------------------------------------------------------------- #
#: Uniform additive noise amplitudes, in PHYSICAL units before the observation
#: scale, copied from the reference Wheel-Legged-Gym noise_scales.  The term is
#: scaled afterwards, so e.g. 0.2 rad/s on base_ang_vel becomes +-0.05 in the
#: observation (scale 0.25).
OBS_NOISE_LEVEL = _env_float("INFANTRY_RL_OBS_NOISE_LEVEL", 1.0)
_NOISE_ANG_VEL = _env_float("INFANTRY_RL_OBS_NOISE_ANG_VEL", 0.2) * OBS_NOISE_LEVEL
_NOISE_GRAVITY = _env_float("INFANTRY_RL_OBS_NOISE_GRAVITY", 0.05) * OBS_NOISE_LEVEL
_NOISE_DOF_POS = _env_float("INFANTRY_RL_OBS_NOISE_DOF_POS", 0.01) * OBS_NOISE_LEVEL
_NOISE_DOF_VEL = _env_float("INFANTRY_RL_OBS_NOISE_DOF_VEL", 1.5) * OBS_NOISE_LEVEL

#: Reference-style history encoder: 5 frames of the 25-D observation.
HISTORY_LENGTH = int(_env_float("INFANTRY_RL_HISTORY_LENGTH", 5.0))
FRAME_DIM = 25
#: Per-dimension uniform noise amplitudes in the SCALED observation units
#: (the history term applies them to each new frame before storing it).
NOISE_VEC: tuple[float, ...] = (
    (_NOISE_ANG_VEL * 0.25,) * 3          # base_ang_vel
    + (_NOISE_GRAVITY,) * 3               # projected_gravity
    + (0.0,) * 3                          # command
    + (_NOISE_DOF_POS,) * 4               # hip positions
    + (_NOISE_DOF_VEL * 0.05,) * 6        # hip + wheel velocities
    + (0.0,) * 6                          # last action
)
if len(NOISE_VEC) != FRAME_DIM:
    raise ValueError(f"NOISE_VEC has {len(NOISE_VEC)} entries, expected {FRAME_DIM}")

#: "position_diff" matches the encoder-derived velocity used by the firmware
#: (and by the reference repository); "sim" uses PhysX's exact joint velocity.
VEL_SOURCE = os.environ.get("INFANTRY_RL_VEL_SOURCE", "position_diff").strip().lower()
#: first-order low-pass alpha; 0 disables it.
VEL_FILTER_ALPHA = _env_float("INFANTRY_RL_VEL_FILTER_ALPHA", 0.0)

# --------------------------------------------------------------------------- #
# Domain randomization (reference Wheel-Legged-Gym)
# --------------------------------------------------------------------------- #
#: Master switch.  Values below are sampled per environment at startup.
DR_ENABLED = os.environ.get("INFANTRY_RL_DR", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
DR_FRICTION_MIN = _env_float("INFANTRY_RL_DR_FRICTION_MIN", 0.4)
DR_FRICTION_MAX = _env_float("INFANTRY_RL_DR_FRICTION_MAX", 1.2)
DR_RESTITUTION_MIN = _env_float("INFANTRY_RL_DR_RESTITUTION_MIN", 0.0)
DR_RESTITUTION_MAX = _env_float("INFANTRY_RL_DR_RESTITUTION_MAX", 0.2)
#: added base mass (kg); reference range is [-2, 3]
DR_MASS_MIN = _env_float("INFANTRY_RL_DR_MASS_MIN", -2.0)
DR_MASS_MAX = _env_float("INFANTRY_RL_DR_MASS_MAX", 3.0)
#: added base CoM offset (m); reference range is +/-0.05
DR_COM_XY = _env_float("INFANTRY_RL_DR_COM_XY", 0.05)
DR_COM_Z = _env_float("INFANTRY_RL_DR_COM_Z", 0.03)
#: knee servo gain scale; reference randomizes its PD gains +/-10%
DR_GAIN_MIN = _env_float("INFANTRY_RL_DR_GAIN_MIN", 0.9)
DR_GAIN_MAX = _env_float("INFANTRY_RL_DR_GAIN_MAX", 1.1)
#: per-joint motor torque scale; reference is +/-10%
DR_MOTOR_MIN = _env_float("INFANTRY_RL_DR_MOTOR_MIN", 0.9)
DR_MOTOR_MAX = _env_float("INFANTRY_RL_DR_MOTOR_MAX", 1.1)
#: random push velocity (m/s).  The reference uses 2.0 for locomotion; a squat is
#: much less stable, so the default is much gentler.  Raise it once the policy
#: reliably holds the pose.
DR_PUSH_VEL = _env_float("INFANTRY_RL_DR_PUSH_VEL", 0.3)
DR_PUSH_INTERVAL_MIN = _env_float("INFANTRY_RL_DR_PUSH_INTERVAL_MIN", 5.0)
DR_PUSH_INTERVAL_MAX = _env_float("INFANTRY_RL_DR_PUSH_INTERVAL_MAX", 8.0)
#: per-environment action delay in CONTROL steps (0-1 at 100 Hz = 0-10 ms), matching the reference.
ACTION_DELAY_MAX = int(_env_float("INFANTRY_RL_ACTION_DELAY_MAX", 1.0))
#: reset hip pose jitter (rad, per hip); reference randomizes its default pose by
#: +/-0.05.  Our squat sits at the workspace extreme, so the default is smaller.
RESET_HIP_JITTER = _env_float("INFANTRY_RL_RESET_HIP_JITTER", 0.01)

if not DR_ENABLED:
    DR_FRICTION_MIN = DR_FRICTION_MAX = 1.0
    DR_RESTITUTION_MIN = DR_RESTITUTION_MAX = 0.0
    DR_MASS_MIN = DR_MASS_MAX = 0.0
    DR_COM_XY = DR_COM_Z = 0.0
    DR_GAIN_MIN = DR_GAIN_MAX = 1.0
    DR_MOTOR_MIN = DR_MOTOR_MAX = 1.0
    DR_PUSH_VEL = 0.0
    ACTION_DELAY_MAX = 0
    RESET_HIP_JITTER = 0.0

#: The chassis is only ~7.6 cm off the ground, so a pitch/roll of ~16 deg already
#: puts a base-link corner into the ground.  These limits terminate just before the
#: corner touches; the matching L1 posture rewards pull the equilibrium to level.
PITCH_LIMIT = _env_float("INFANTRY_RL_SQUAT_PITCH_LIMIT", float(_SQUAT.get("pitch_limit", 0.24)))
ROLL_LIMIT = _env_float("INFANTRY_RL_SQUAT_ROLL_LIMIT", float(_SQUAT.get("roll_limit", 0.30)))


# --------------------------------------------------------------------------- #
# Reset event
# --------------------------------------------------------------------------- #
def reset_squat_pose(
    env,
    env_ids: Sequence[int],
    hip_offsets: tuple[float, float, float, float] = SQUAT_HIP_OFFSETS,
    jitter: float = 0.0,
    velocity_range: tuple[float, float] = (-0.05, 0.05),
    action_name: str = "joints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Write the hips and both knee pairs to the fixed squat pose.

    The hips are ``nominal + hip_offsets`` where ``hip_offsets`` is in
    :data:`HIP_ORDER` = (lf, lb, rf, rb).  The right side is NOT necessarily the
    same offset as the left: ``chassis-rl`` mirrors its right-leg joint signs, so
    the same physical pose has opposite joint values.  The knees come from the
    action term's :attr:`reset_knees`, i.e. the solved closed-chain pair for that
    hip pose, so the first physics step does not start with a large loop error.
    """
    asset = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    if ids.numel() == 0:
        return
    n = ids.numel()
    device = asset.device

    offsets = torch.tensor(hip_offsets, device=device, dtype=torch.float32).view(1, -1)
    if jitter > 0.0:
        # independent per-hip noise (the reference randomizes each joint separately)
        offsets = offsets + torch.empty(n, offsets.shape[1], device=device).uniform_(
            -jitter, jitter
        )
    hip_targets = term.nominal_hips[ids] + offsets
    knee_targets = term.reset_knees.unsqueeze(0).expand(n, -1)

    # Resolve by name, never by a hard-coded articulation order.
    joint_names = list(HIP_ORDER) + list(KNEE_ORDER)
    joint_ids, resolved_names = asset.find_joints(joint_names, preserve_order=True)
    values: dict[str, torch.Tensor] = {}
    for i, name in enumerate(HIP_ORDER):
        values[name] = hip_targets[:, i]
    for i, name in enumerate(KNEE_ORDER):
        values[name] = knee_targets[:, i]
    joint_pos = torch.stack([values[name] for name in resolved_names], dim=-1)

    limits = asset.data.soft_joint_pos_limits[ids][:, joint_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

    joint_vel = torch.empty(n, len(joint_ids), device=device).uniform_(*velocity_range)
    vel_limits = asset.data.soft_joint_vel_limits[ids][:, joint_ids]
    joint_vel = joint_vel.clamp(-vel_limits, vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=ids)


# --------------------------------------------------------------------------- #
# MDP config
# --------------------------------------------------------------------------- #
@configclass
class SquatObservationsCfg:
    """History observation for the 5-frame encoder.

    Per frame (25 dims)::

        [base_ang_vel(3)*0.25, projected_gravity(3), command(3),
         hip_pos_rel_locked(4), joint_vel(6)*0.05, last_action(6)]

    The group emits the flattened 5-frame history ``(125,)`` in oldest-to-newest
    order.  The custom actor in ``policy_model.py`` splits it into the current 25
    dims and the 125-dim encoder input, producing a 28-dim actor input.

    Training adds uniform sensor noise to each new frame (amplitudes follow the
    reference Wheel-Legged-Gym ``noise_scales``); commands and the previous action
    are noise-free.  ``SquatEnvCfg_PLAY`` disables the corruption.  ``joint_vel``
    is an encoder-style position difference by default, optionally low-pass
    filtered.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # One term so the history is frame-major [o_{t-4}, ..., o_t].
        history = ObsTerm(
            func=mdp.history_obs,
            params={
                "action_name": "joints",
                "length": HISTORY_LENGTH,
                "noise_vec": NOISE_VEC,
            },
        )

        def __post_init__(self):
            # The term applies noise to each new frame when this is True;
            # SquatEnvCfg_PLAY turns it off.
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SquatRewardsCfg:
    """Balance-only rewards for the locked squat.

    The hips are torque-driven, so the posture terms are the actual holding
    controller.  ``joint_torques`` is restricted to the policy-actuated hips/wheels
    so it does not charge the knee PD load that closes the linkage.
    """

    # -- staying up ------------------------------------------------------
    # Every term here must be derivable from the 25-D observation.  Heading,
    # planar position and base linear velocity are NOT observed any more, so
    # rewarding them only injects advantage noise and encourages the policy to
    # trade pitch stability for quantities it cannot see.
    alive = RewTerm(func=base_mdp.is_alive, weight=0.5)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-10.0, params={"target_height": SQUAT_BASE_HEIGHT}
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_x = RewTerm(func=mdp.ang_vel_x_l2, weight=-0.05)
    # pitch rate is observed (base_ang_vel_y); damp it harder than roll rate.
    ang_vel_y = RewTerm(func=mdp.ang_vel_y_l2, weight=-0.25)
    # L1 posture pull.  flat_orientation_l2 alone is too weak at this very low
    # stance: a 17 deg chassis-corner lean is a stable local optimum otherwise.
    # -10 gives a constant 2.9/step gradient at that lean, dominating `alive`.
    pitch = RewTerm(func=mdp.pitch_abs, weight=-10.0)
    roll = RewTerm(func=mdp.roll_abs, weight=-2.0)
    # yaw rate is observable (base_ang_vel_z); the yaw ANGLE is not.
    yaw_rate = RewTerm(func=mdp.yaw_rate_l2, weight=-0.25)

    # -- smooth actuation ------------------------------------------------
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[*HIP_ORDER, *WHEEL_ORDER])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    wheel_spin = RewTerm(func=mdp.wheel_spin_l2, weight=-1.0e-4, params={"action_name": "joints"})
    wheel_axle = RewTerm(
        func=mdp.leg_axle_skew_abs, weight=-2.0, params={"action_name": "joints"}
    )
    action_jerk = RewTerm(
        func=mdp.action_jerk_l2,
        weight=-5.0e-3,
        params={"action_name": "joints"},
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
    # Weak posture prior, like the reference's nominal_state (-0.1).  A strong
    # lock would fight the policy's hip position targets.
    hip_lock = RewTerm(
        func=mdp.hip_lock_deviation_abs,
        weight=-0.1,
        params={"action_name": "joints", "tolerance": 0.05},
    )

    # -- the chassis must stay off the ground ----------------------------
    # A 17 deg lean lets the base-link corner rest on the floor, which is a
    # stable local optimum with the strict pitch limit removed.  Charge it.
    base_contact = RewTerm(
        func=base_mdp.undesired_contacts,
        weight=-20.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces")},
    )

    # -- falling is bad --------------------------------------------------
    fall_penalty = RewTerm(func=mdp.fall_penalty, weight=-50.0)


@configclass
class SquatCommandsCfg:
    """A constant zero command; the task is balance, not velocity tracking."""

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
class SquatEventCfg:
    """Start upright in the squat pose, plus the reference-style randomizations."""

    # -- startup domain randomization ------------------------------------
    # Robot shape friction/restitution (the ground material stays fixed; the
    # effective contact friction is the product of the two).
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
    # Base mass and CoM.  The reference adds [-2, 3] kg and shifts the CoM by
    # +/-5 cm; inertia is recomputed from the new mass.
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
    # The knees are the only position-controlled joints, so this randomizes the
    # closed-chain servo stiffness/damping by +/-10%.
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
    # Random pushes while standing.  Applied as a root velocity impulse, matching
    # the reference's push_robots.
    push_robots = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(DR_PUSH_INTERVAL_MIN, DR_PUSH_INTERVAL_MAX),
        params={
            "velocity_range": {
                "x": (-DR_PUSH_VEL, DR_PUSH_VEL),
                "y": (-DR_PUSH_VEL, DR_PUSH_VEL),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # -- per-episode reset ------------------------------------------------
    reset_root = EventTerm(
        func=mdp.reset_root_state_standing_squat,
        mode="reset",
        params={
            "pitch_range_deg": (-TORQUE_RESET_PITCH_DEG, TORQUE_RESET_PITCH_DEG),
            "roll_range_deg": (-TORQUE_RESET_ROLL_DEG, TORQUE_RESET_ROLL_DEG),
            "yaw_range_deg": (-180.0, 180.0),
            "z_offset_range": (SQUAT_Z_OFFSET - 0.004, SQUAT_Z_OFFSET + 0.004),
            "lin_vel_range": (-0.05, 0.05),
            "ang_vel_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
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
class SquatTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 1.2})
    # Stop the chassis-corner tripod before the base link touches the ground.
    pitch_limit = DoneTerm(func=mdp.pitch_above, params={"limit_angle": PITCH_LIMIT})
    roll_limit = DoneTerm(func=mdp.roll_above, params={"limit_angle": ROLL_LIMIT})
    # 0.11 m (the normal fall floor) would terminate the squat immediately.
    base_too_low = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": BASE_FLOOR}
    )


@configclass
class SquatEnvCfg(ManagerBasedRLEnvCfg):
    """Hybrid hip-torque / wheel-position balance on a fixed squat stance."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: SquatObservationsCfg = SquatObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: SquatRewardsCfg = SquatRewardsCfg()
    terminations: SquatTerminationsCfg = SquatTerminationsCfg()
    commands: SquatCommandsCfg = SquatCommandsCfg()
    events: SquatEventCfg = SquatEventCfg()

    joint_action_name: str = "joints"

    def __post_init__(self):
        # 100 Hz policy update (sim 200 Hz / decimation 2), matching the
        # reference wheel_legged task.  The 5-frame history is then 50 ms.
        self.decimation = int(_env_float("INFANTRY_RL_DECIMATION", 2.0))
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

        # -- pure torque: 4 hips + 2 wheels, reference squat pose -------------
        joints = self.actions.joints
        joints.hip_bias = SQUAT_HIP_OFFSETS
        joints.hip_pos_scale = HIP_POS_SCALE
        joints.wheel_vel_scale = WHEEL_VEL_SCALE
        joints.hip_kp = HIP_KP
        joints.hip_kd = HIP_KD
        joints.wheel_kd = WHEEL_KD
        joints.hip_effort_limit = TORQUE_HIP_EFFORT_LIMIT
        joints.wheel_effort_limit = TORQUE_WHEEL_EFFORT_LIMIT
        # explicit PD torques; no implicit actuator gains
        wheels = self.scene.robot.actuators["wheels"]
        wheels.stiffness = 0.0
        wheels.damping = 0.0
        hips = self.scene.robot.actuators["hips"]
        hips.stiffness = 0.0
        hips.damping = 0.0
        joints.vel_source = VEL_SOURCE
        joints.vel_filter_alpha = VEL_FILTER_ALPHA
        joints.motor_torque_scale_range = (DR_MOTOR_MIN, DR_MOTOR_MAX)
        joints.action_delay_steps_range = (0, ACTION_DELAY_MAX)

        print(
            f"[cfg] squat pose          : hips L(front {SQUAT_FRONT_HIP:+.4f} / "
            f"back {SQUAT_BACK_HIP:+.4f}) R(front {SQUAT_RIGHT_FRONT_HIP:+.4f} / "
            f"back {SQUAT_RIGHT_BACK_HIP:+.4f}) rad, L0 {SQUAT_LEG_LENGTH:.5f} m, "
            f"base_z ~{SQUAT_BASE_HEIGHT:.4f} m (offset from nominal {SQUAT_Z_OFFSET:+.4f} m)"
        )
        print(
            f"[cfg] squat action        : 4 hip POSITION targets + 2 wheel VELOCITY targets "
            f"(hip {HIP_POS_SCALE:g} rad/unit, wheel {WHEEL_VEL_SCALE:g} rad/s/unit); "
            f"PD hip kp={HIP_KP:g} kd={HIP_KD:g}, wheel kd={WHEEL_KD:g}; "
            f"episode {EPISODE_LENGTH_S:.1f} s; base floor {BASE_FLOOR:.3f} m"
        )
        print(
            f"[cfg] squat_torque limits : hips ±{TORQUE_HIP_EFFORT_LIMIT:g} Nm, "
            f"wheels ±{TORQUE_WHEEL_EFFORT_LIMIT:g} Nm, "
            f"knees position-controlled (stiffness 400/damping 10)"
        )
        print(
            f"[cfg] observation         : {FRAME_DIM} per frame x {HISTORY_LENGTH} frames "
            f"= {FRAME_DIM * HISTORY_LENGTH} history (actor 25 + 3 latent); "
            f"uniform noise level {OBS_NOISE_LEVEL:g} "
            f"(ang_vel ±{_NOISE_ANG_VEL:g} rad/s, gravity ±{_NOISE_GRAVITY:g}, "
            f"hip ±{_NOISE_DOF_POS:g} rad, joint_vel ±{_NOISE_DOF_VEL:g} rad/s); "
            f"joint_vel={VEL_SOURCE}, filter alpha={VEL_FILTER_ALPHA:g}"
        )
        print(
            f"[cfg] domain rand         : {'ON' if DR_ENABLED else 'OFF'}, "
            f"friction {DR_FRICTION_MIN:g}-{DR_FRICTION_MAX:g}, restitution "
            f"{DR_RESTITUTION_MIN:g}-{DR_RESTITUTION_MAX:g}, base mass "
            f"{DR_MASS_MIN:+g}..{DR_MASS_MAX:+g} kg, CoM ±{DR_COM_XY:g} m, "
            f"knee gain {DR_GAIN_MIN:g}-{DR_GAIN_MAX:g}x, motor torque "
            f"{DR_MOTOR_MIN:g}-{DR_MOTOR_MAX:g}x, push {DR_PUSH_VEL:g} m/s "
            f"every {DR_PUSH_INTERVAL_MIN:g}-{DR_PUSH_INTERVAL_MAX:g} s, "
            f"action delay 0-{ACTION_DELAY_MAX} step(s), reset hip jitter "
            f"±{RESET_HIP_JITTER:g} rad"
        )
        self.viewer.origin_type = "world"
        self.viewer.eye = (2.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.1)


@configclass
class SquatEnvCfg_PLAY(SquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6
        self.observations.policy.enable_corruption = False


class FiveBarSquatEnv(FiveBarEnv):
    """Five-bar wheel-leg balancing with hip position PD + wheel velocity PD."""

    cfg: SquatEnvCfg


__all__ = [
    "BASE_FLOOR",
    "EPISODE_LENGTH_S",
    "FiveBarSquatEnv",
    "HIP_KD",
    "HIP_KP",
    "HIP_POS_SCALE",
    "SQUAT_BACK_HIP",
    "SQUAT_BACK_HIP_OFFSET",
    "SQUAT_BASE_HEIGHT",
    "SQUAT_FRONT_HIP",
    "SQUAT_FRONT_HIP_OFFSET",
    "SQUAT_HIP_OFFSETS",
    "SQUAT_LEG_LENGTH",
    "SQUAT_RIGHT_BACK_HIP",
    "SQUAT_RIGHT_FRONT_HIP",
    "SQUAT_Z_OFFSET",
    "SquatCommandsCfg",
    "SquatEnvCfg",
    "SquatEnvCfg_PLAY",
    "SquatEventCfg",
    "SquatObservationsCfg",
    "SquatRewardsCfg",
    "SquatTerminationsCfg",
    "TORQUE_HIP_EFFORT_LIMIT",
    "WHEEL_KD",
    "WHEEL_VEL_SCALE",
    "reset_squat_pose",
]
