"""Squat-pose balance task for the five-bar wheel-leg.

The user's training recipe: lock the four hips at the mechanical extreme of the
closed-chain workspace, so the base sits as low as the linkage allows, then learn
to balance with the wheels alone.  The policy action is therefore **two-dimensional**
(the left/right wheel speed targets); the hips are position-controlled at a fixed
pose and the passive knees are still solved every control step to close the loop.

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

The reset starts upright at that height with the hips already locked and the knees
at the matching closed-chain angles; the policy only has to keep the body over the
wheel axle and stop it drifting / spinning.  Falling is detected by the usual
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
from config import HIP_EFFORT_LIMIT, INFANTRY_RL_TORQUE_CFG, WHEEL_EFFORT_LIMIT
from env_base import (
    ActionsCfg,
    FiveBarEnv,
    NOMINAL_HEIGHT,
    RobotSceneCfg,
    TARGET_HEADING,
    TARGET_X,
    TARGET_Y,
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
#: Nominal measured stance (both sides); the squat pose is expressed as absolute
#: hip angles so the closed-chain solve and the reset event use exactly the same
#: values.  ``chassis-rl`` mirrors its right-side joint signs, so the nominal hip
#: values are NOT identical on the two sides.
_NOMINAL = nominal_pose()

#: Deepest feasible locked pose, written by ``tools/measure_chassis_asset.py`` to
#: ``robot_data/chassis-rl/squat_pose.json``.
_SQUAT = squat_pose()
_SQUAT_HIPS = _SQUAT["hips"]
#: right-side joint values are ``sign * left`` for the same physical pose
#: (-1 for chassis-rl's mirrored right leg)
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
SQUAT_HIP_BIAS = SQUAT_HIP_OFFSETS

#: Leg length and root height of that pose.  These are the values the reward /
#: observation use; the actual settled height is verified by ``--check``.
SQUAT_LEG_LENGTH = _env_float("INFANTRY_RL_SQUAT_LEG_LENGTH", float(_SQUAT["leg_length"]))
SQUAT_BASE_HEIGHT = _env_float(
    "INFANTRY_RL_SQUAT_BASE_HEIGHT",
    float(_SQUAT.get("base_height", WHEEL_RADIUS + SQUAT_LEG_LENGTH - hip_axis_height())),
)
#: Reset offset from the default root height (NOMINAL_HEIGHT) down to the squat.
SQUAT_Z_OFFSET = SQUAT_BASE_HEIGHT - NOMINAL_HEIGHT

EPISODE_LENGTH_S = _env_float("INFANTRY_RL_SQUAT_EPISODE", 10.0)
WHEEL_SCALE = _env_float("INFANTRY_RL_SQUAT_WHEEL_SCALE", 5.0)
HIP_RANGE = _env_float("INFANTRY_RL_SQUAT_HIP_RANGE", 0.70)
BASE_FLOOR = _env_float("INFANTRY_RL_SQUAT_BASE_FLOOR", float(_SQUAT.get("base_floor", 0.04)))

#: pure-torque variant (`squat_torque`): raw action unit -> Nm
HIP_TORQUE_SCALE = _env_float("INFANTRY_RL_SQUAT_HIP_TORQUE_SCALE", 10.0)
WHEEL_TORQUE_SCALE = _env_float("INFANTRY_RL_SQUAT_WHEEL_TORQUE_SCALE", 1.0)
TORQUE_HIP_EFFORT_LIMIT = _env_float("INFANTRY_RL_SQUAT_HIP_EFFORT_LIMIT", HIP_EFFORT_LIMIT)
TORQUE_WHEEL_EFFORT_LIMIT = _env_float("INFANTRY_RL_SQUAT_WHEEL_EFFORT_LIMIT", WHEEL_EFFORT_LIMIT)
#: training-only PD assist on the hips, decayed to zero so the final policy is pure
#: torque.  kp/kd match the original implicit hip servo; 0 disables the curriculum.
HIP_ASSIST_KP = _env_float("INFANTRY_RL_SQUAT_HIP_ASSIST_KP", 200.0)
HIP_ASSIST_KD = _env_float("INFANTRY_RL_SQUAT_HIP_ASSIST_KD", 5.0)
HIP_ASSIST_DECAY_STEPS = int(_env_float("INFANTRY_RL_SQUAT_HIP_ASSIST_DECAY_STEPS", 10000.0))
#: training-only wheel damping (Nm per rad/s), matching the original velocity servo
WHEEL_ASSIST_KD = _env_float("INFANTRY_RL_SQUAT_WHEEL_ASSIST_KD", 2.0)
#: training-only inverted-pendulum pitch assist (common wheel torque, Nm/rad and Nm/(rad/s))
WHEEL_PITCH_ASSIST_KP = _env_float("INFANTRY_RL_SQUAT_WHEEL_PITCH_ASSIST_KP", 25.0)
WHEEL_PITCH_ASSIST_KD = _env_float("INFANTRY_RL_SQUAT_WHEEL_PITCH_ASSIST_KD", 4.0)
#: the torque task starts with a gentler reset: a free-torque hip has no PD to catch
#: the initial pitch, so the first episodes must not start at the corner limit
TORQUE_RESET_PITCH_DEG = _env_float("INFANTRY_RL_SQUAT_TORQUE_RESET_PITCH_DEG", 1.5)
TORQUE_RESET_ROLL_DEG = _env_float("INFANTRY_RL_SQUAT_TORQUE_RESET_ROLL_DEG", 0.75)
#: optional training-only teacher (the existing 2-action squat checkpoint)
TEACHER_CKPT = os.environ.get("INFANTRY_RL_SQUAT_TEACHER_CKPT", "").strip()

#: The chassis is only 7.6 cm off the ground, so a pitch/roll of ~16 deg already
#: puts a base-link corner into the ground.  A policy that learns to "tripod" on
#: that corner (measured: a stable 17 deg nose-up lean at base_z 0.0735 m) is not
#: balancing on its wheels.  These limits terminate just before the corner touches;
#: the matching L1 posture rewards pull the equilibrium towards level.
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
        offsets = offsets + torch.empty(n, device=device).uniform_(-jitter, jitter)
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
    """Observation specifications (30 dims).

    No command term: the task is "hold this pose, do not move".  Heading and planar
    position are fed explicitly for the same reason as the balance task -- without
    them the robot can balance perfectly while slowly spinning or drifting.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # base state
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=0.25)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)

        # where it is / which way it faces (prevents drift and slow spin)
        heading = ObsTerm(func=mdp.heading_sin_cos, params={"target_heading": TARGET_HEADING})
        position = ObsTerm(
            func=mdp.position_error, params={"target_x": TARGET_X, "target_y": TARGET_Y}
        )

        # leg state: the hips are a constant locked offset; the knee term is the
        # closed-chain tracking error and is the early warning of a loop failure.
        hip_pos_rel_locked = ObsTerm(func=mdp.hip_pos_rel_nominal, params={"action_name": "joints"})
        knee_pos_rel_solved = ObsTerm(func=mdp.knee_pos_rel_solved, params={"action_name": "joints"})
        solved_leg_state = ObsTerm(func=mdp.solved_leg_state, params={"action_name": "joints"})

        # actuators
        wheel_vel = ObsTerm(func=mdp.wheel_vel, params={"action_name": "joints"}, scale=0.1)
        last_action = ObsTerm(func=mdp.last_action_raw, params={"action_name": "joints"})

        # height error relative to the squat target (helps detect sinking / a stance
        # collapse before the termination fires)
        base_height = ObsTerm(
            func=mdp.base_height_rel, params={"target_height": SQUAT_BASE_HEIGHT}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SquatTorqueObservationsCfg:
    """Observation specifications for the pure-torque squat (42 dims).

    Torque control needs velocity feedback on the actuated hips and on the leg
    linkage.  ``solved_leg_state`` is replaced by ``measured_leg_state`` because
    the commanded hip pose is only a reference now; the knee tracking error stays.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # base state
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=0.25)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)

        # where it is / which way it faces (prevents drift and slow spin)
        heading = ObsTerm(func=mdp.heading_sin_cos, params={"target_heading": TARGET_HEADING})
        position = ObsTerm(
            func=mdp.position_error, params={"target_x": TARGET_X, "target_y": TARGET_Y}
        )

        # posture / linkage state
        hip_pos_rel_locked = ObsTerm(func=mdp.hip_pos_rel_locked, params={"action_name": "joints"})
        hip_vel = ObsTerm(func=mdp.hip_vel, params={"action_name": "joints"}, scale=0.1)
        knee_pos_rel_solved = ObsTerm(func=mdp.knee_pos_rel_solved, params={"action_name": "joints"})
        measured_leg_state = ObsTerm(func=mdp.measured_leg_state, params={"action_name": "joints"})
        leg_rates = ObsTerm(func=mdp.leg_rates, params={"action_name": "joints"}, scale=0.5)

        # actuators
        wheel_vel = ObsTerm(func=mdp.wheel_vel, params={"action_name": "joints"}, scale=0.1)
        last_action = ObsTerm(func=mdp.last_action_raw, params={"action_name": "joints"})

        # height error relative to the squat target
        base_height = ObsTerm(
            func=mdp.base_height_rel, params={"target_height": SQUAT_BASE_HEIGHT}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SquatRewardsCfg:
    """Balance-only rewards for the locked squat."""

    # -- staying up ------------------------------------------------------
    alive = RewTerm(func=base_mdp.is_alive, weight=0.5)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-10.0, params={"target_height": SQUAT_BASE_HEIGHT}
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_x = RewTerm(func=mdp.ang_vel_x_l2, weight=-0.05)
    ang_vel_y = RewTerm(func=mdp.ang_vel_y_l2, weight=-0.05)
    # L1 posture pull.  flat_orientation_l2 alone is too weak at this very low
    # stance: a 17 deg chassis-corner lean costs only ~0.17/step and is a stable
    # local optimum.  The L1 terms have a constant gradient at every angle.
    pitch = RewTerm(func=mdp.pitch_abs, weight=-4.0)
    roll = RewTerm(func=mdp.roll_abs, weight=-1.0)

    # -- hold the heading / planar position ------------------------------
    heading = RewTerm(
        func=mdp.heading_l2, weight=-0.5, params={"target_heading": TARGET_HEADING}
    )
    yaw_rate = RewTerm(func=mdp.yaw_rate_l2, weight=-0.25)
    heading_alive = RewTerm(
        func=mdp.heading_alive,
        weight=0.25,
        params={"target_heading": TARGET_HEADING, "tolerance": 0.35},
    )
    position = RewTerm(
        func=mdp.position_xy_l2, weight=-4.0, params={"target_x": TARGET_X, "target_y": TARGET_Y}
    )
    lin_vel = RewTerm(func=mdp.lin_vel_xy_l2, weight=-1.0)
    position_alive = RewTerm(
        func=mdp.position_xy_alive,
        weight=0.2,
        params={"target_x": TARGET_X, "target_y": TARGET_Y, "tolerance": 0.2},
    )

    # -- smooth actuation ------------------------------------------------
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    wheel_spin = RewTerm(func=mdp.wheel_spin_l2, weight=-1.0e-4, params={"action_name": "joints"})
    wheel_axle = RewTerm(
        func=mdp.leg_axle_skew_abs, weight=-2.0, params={"action_name": "joints"}
    )

    # -- keep the closed chain closed and the hips locked ----------------
    closure_residual = RewTerm(
        func=mdp.closure_residual_l2, weight=-1.0e3, params={"action_name": "joints"}
    )
    hip_lock = RewTerm(
        func=mdp.hip_lock_deviation_abs,
        weight=-2.0,
        params={"action_name": "joints", "tolerance": 0.05},
    )

    # -- falling is bad --------------------------------------------------
    fall_penalty = RewTerm(func=mdp.fall_penalty, weight=-50.0)


@configclass
class SquatTorqueRewardsCfg(SquatRewardsCfg):
    """Rewards for the pure-torque squat.

    The hips are no longer held by a position servo, so the posture terms are the
    actual holding controller; they are tightened.  ``joint_torques`` is restricted
    to the policy-actuated hips/wheels so it does not charge the knee PD load that
    closes the linkage.
    """

    hip_lock = RewTerm(
        func=mdp.hip_lock_deviation_abs,
        weight=-4.0,
        params={"action_name": "joints", "tolerance": 0.03},
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[*HIP_ORDER, *WHEEL_ORDER])},
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
    closure_residual = RewTerm(
        func=mdp.closure_residual_l2, weight=-2.0e3, params={"action_name": "joints"}
    )


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
    """Start upright, already in the locked squat, with small pose noise."""

    reset_root = EventTerm(
        func=mdp.reset_root_state_standing_squat,
        mode="reset",
        params={
            "pitch_range_deg": (-4.0, 4.0),
            "roll_range_deg": (-2.0, 2.0),
            "yaw_range_deg": (-180.0, 180.0),
            "z_offset_range": (SQUAT_Z_OFFSET - 0.004, SQUAT_Z_OFFSET + 0.004),
            "lin_vel_range": (-0.05, 0.05),
            "ang_vel_range": (-0.10, 0.10),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_pose = EventTerm(
        func=reset_squat_pose,
        mode="reset",
        params={
            "hip_offsets": SQUAT_HIP_OFFSETS,
            "jitter": 0.0,
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
    """Two-wheel balance on a mechanically locked, deepest-possible stance."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: SquatObservationsCfg = SquatObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: SquatRewardsCfg = SquatRewardsCfg()
    terminations: SquatTerminationsCfg = SquatTerminationsCfg()
    commands: SquatCommandsCfg = SquatCommandsCfg()
    events: SquatEventCfg = SquatEventCfg()

    joint_action_name: str = "joints"

    def __post_init__(self):
        self.decimation = 4
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

        # -- lock the four hips and leave only the two wheels to the policy -----
        joints = self.actions.joints
        joints.lock_hips = True
        joints.hip_bias = SQUAT_HIP_OFFSETS
        joints.hip_range = HIP_RANGE
        joints.wheel_scale = WHEEL_SCALE
        joints.limit_margin = 0.01
        joints.solver_iters = 2
        # every optional outer loop / stance bias is off: the hips are fixed
        joints.heading_control = False
        joints.conditional_heading_control = False
        joints.position_control = False
        joints.pitch_control = False
        joints.sagittal_symmetry = False
        joints.hip_symmetry = "off"
        joints.hip_symmetry_gate = False
        joints.stance_randomize = False
        joints.dynamic_hip_bias = False
        joints.hip_bias_time_ramp = False

        print(
            f"[cfg] squat pose          : hips L(front {SQUAT_FRONT_HIP:+.4f} / "
            f"back {SQUAT_BACK_HIP:+.4f}) R(front {SQUAT_RIGHT_FRONT_HIP:+.4f} / "
            f"back {SQUAT_RIGHT_BACK_HIP:+.4f}) rad, L0 {SQUAT_LEG_LENGTH:.5f} m, "
            f"base_z ~{SQUAT_BASE_HEIGHT:.4f} m (offset from nominal {SQUAT_Z_OFFSET:+.4f} m)"
        )
        if joints.torque_control:
            print(
                f"[cfg] squat action        : pure torque, 4 hips + 2 wheels "
                f"(hip {HIP_TORQUE_SCALE:g} Nm/unit, wheel {WHEEL_TORQUE_SCALE:g} Nm/unit); "
                f"episode {EPISODE_LENGTH_S:.1f} s; base floor {BASE_FLOOR:.3f} m"
            )
        else:
            print(
                f"[cfg] squat action        : 2 wheels only (hip lock, wheel_scale={WHEEL_SCALE:g}, "
                f"hip_range={HIP_RANGE:g}); episode {EPISODE_LENGTH_S:.1f} s; "
                f"base floor {BASE_FLOOR:.3f} m"
            )

        self.viewer.origin_type = "world"
        self.viewer.eye = (2.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.1)


@configclass
class SquatTorqueEnvCfg(SquatEnvCfg):
    """Pure-torque squat: 4 hip torques + 2 wheel torques, knees closed-chain followers."""

    observations: SquatTorqueObservationsCfg = SquatTorqueObservationsCfg()
    rewards: SquatTorqueRewardsCfg = SquatTorqueRewardsCfg()

    def __post_init__(self):
        # Set the action mode before super() so its config print is accurate and so
        # the scale/limit values below are not overwritten.
        joints = self.actions.joints
        joints.torque_control = True
        joints.hip_torque_scale = HIP_TORQUE_SCALE
        joints.wheel_torque_scale = WHEEL_TORQUE_SCALE
        joints.hip_effort_limit = TORQUE_HIP_EFFORT_LIMIT
        joints.wheel_effort_limit = TORQUE_WHEEL_EFFORT_LIMIT
        joints.hip_assist_kp = HIP_ASSIST_KP
        joints.hip_assist_kd = HIP_ASSIST_KD
        joints.hip_assist_decay_steps = HIP_ASSIST_DECAY_STEPS
        if TEACHER_CKPT:
            # The teacher servo already contains the -kd*v damping; adding the
            # damping/pitch assists on top would double-count and fight it.
            joints.wheel_assist_kd = 0.0
            joints.wheel_pitch_assist_kp = 0.0
            joints.wheel_pitch_assist_kd = 0.0
        else:
            joints.wheel_assist_kd = WHEEL_ASSIST_KD
            joints.wheel_pitch_assist_kp = WHEEL_PITCH_ASSIST_KP
            joints.wheel_pitch_assist_kd = WHEEL_PITCH_ASSIST_KD
        joints.teacher_checkpoint = TEACHER_CKPT
        joints.teacher_base_height = SQUAT_BASE_HEIGHT
        super().__post_init__()

        # Hips/wheels lose their implicit PD; the knees keep stiffness=400/damping=10.
        self.scene.robot = INFANTRY_RL_TORQUE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # A torque hip cannot catch a large initial pitch, so start closer to level.
        self.events.reset_root.params["pitch_range_deg"] = (
            -TORQUE_RESET_PITCH_DEG, TORQUE_RESET_PITCH_DEG
        )
        self.events.reset_root.params["roll_range_deg"] = (
            -TORQUE_RESET_ROLL_DEG, TORQUE_RESET_ROLL_DEG
        )
        self.events.reset_root.params["ang_vel_range"] = (-0.05, 0.05)
        print(
            f"[cfg] squat_torque       : hips ±{TORQUE_HIP_EFFORT_LIMIT:g} Nm, "
            f"wheels ±{TORQUE_WHEEL_EFFORT_LIMIT:g} Nm, "
            f"knees position-controlled (stiffness 400/damping 10); "
            f"training assist hip kp={HIP_ASSIST_KP:g} kd={HIP_ASSIST_KD:g}, "
            f"wheel kd={WHEEL_ASSIST_KD:g}, pitch kp={WHEEL_PITCH_ASSIST_KP:g} "
            f"kd={WHEEL_PITCH_ASSIST_KD:g}, teacher={TEACHER_CKPT or 'off'}, "
            f"decay={HIP_ASSIST_DECAY_STEPS} steps"
        )


@configclass
class SquatEnvCfg_PLAY(SquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6
        self.observations.policy.enable_corruption = False


@configclass
class SquatTorqueEnvCfg_PLAY(SquatTorqueEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Deployment is pure torque: never re-enable the training assist.  The
        # debug-only switch keeps it for a zero-action stability test.
        _keep_assist = os.environ.get("INFANTRY_RL_SQUAT_PLAY_ASSIST", "0").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if not _keep_assist:
            self.actions.joints.hip_assist_kp = 0.0
            self.actions.joints.hip_assist_kd = 0.0
            self.actions.joints.hip_assist_decay_steps = 0
            self.actions.joints.wheel_assist_kd = 0.0
            self.actions.joints.wheel_pitch_assist_kp = 0.0
            self.actions.joints.wheel_pitch_assist_kd = 0.0
            self.actions.joints.teacher_checkpoint = ""
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6
        self.observations.policy.enable_corruption = False


class FiveBarSquatEnv(FiveBarEnv):
    """Five-bar wheel-leg balancing in a fixed, deepest-possible squat."""

    cfg: SquatEnvCfg


class FiveBarSquatTorqueEnv(FiveBarSquatEnv):
    """Five-bar wheel-leg balancing with pure hip/wheel torque and closed-chain knees."""

    cfg: SquatTorqueEnvCfg


__all__ = [
    "BASE_FLOOR",
    "EPISODE_LENGTH_S",
    "FiveBarSquatEnv",
    "FiveBarSquatTorqueEnv",
    "HIP_ASSIST_DECAY_STEPS",
    "HIP_ASSIST_KD",
    "HIP_ASSIST_KP",
    "HIP_RANGE",
    "HIP_TORQUE_SCALE",
    "SQUAT_BACK_HIP",
    "SQUAT_BACK_HIP_OFFSET",
    "SQUAT_BASE_HEIGHT",
    "SQUAT_FRONT_HIP",
    "SQUAT_FRONT_HIP_OFFSET",
    "SQUAT_HIP_BIAS",
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
    "SquatTorqueEnvCfg",
    "SquatTorqueEnvCfg_PLAY",
    "SquatTorqueObservationsCfg",
    "SquatTorqueRewardsCfg",
    "TORQUE_HIP_EFFORT_LIMIT",
    "TORQUE_WHEEL_EFFORT_LIMIT",
    "WHEEL_ASSIST_KD",
    "WHEEL_PITCH_ASSIST_KD",
    "WHEEL_PITCH_ASSIST_KP",
    "WHEEL_SCALE",
    "WHEEL_TORQUE_SCALE",
    "reset_squat_pose",
]
