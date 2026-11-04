"""Stand-from-squat environment for the five-bar wheel-leg.

This is the missing second half of fall recovery:

1. :mod:`env_recovery` gets the robot from a front/back fall into a stable upright
   squat (hips tucked, base height ~0.10 m);
2. **this** task starts the robot exactly there and teaches it to extend the legs
   into the nominal standing posture and stay balanced.

The task is deliberately easy compared with recovery: the body is already upright,
the feet are on the ground, and the action term's zero hip target is the nominal
stance.  The soft position controller therefore performs the leg extension even
before the policy does anything; the policy mainly has to balance while the base
rises.  That is why it can be warm-started directly from the teleop/balance
checkpoint (``model_11344``) instead of training from scratch.

As with recovery, success does NOT terminate the episode: the robot keeps standing
to the time limit, and ``Metrics/recovery/success`` records whether it reached the
upright + height + still condition.
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


EPISODE_LENGTH_S = _env_float("INFANTRY_RL_STAND_EPISODE", 6.0)
HOLD_TIME_S = _env_float("INFANTRY_RL_STAND_HOLD", 0.5)
STOP_LIN_VEL = _env_float("INFANTRY_RL_STAND_STOP_LIN", 0.20)
STOP_ANG_VEL = _env_float("INFANTRY_RL_STAND_STOP_ANG", 0.50)
UPRIGHT_MIN = _env_float("INFANTRY_RL_STAND_UPRIGHT_MIN", 0.95)
MIN_HEIGHT = _env_float("INFANTRY_RL_STAND_MIN_HEIGHT", 0.16)
UPRIGHT_ZONE_MIN = _env_float("INFANTRY_RL_STAND_ZONE_MIN", 0.95)
UPRIGHT_POWER = _env_float("INFANTRY_RL_STAND_UPRIGHT_POWER", 3.0)
RECOVERED_BONUS_WEIGHT = _env_float("INFANTRY_RL_STAND_BONUS_WEIGHT", 100.0)
ROLL_LIMIT_RAD = _env_float("INFANTRY_RL_STAND_ROLL_LIMIT", 1.05)
ROLL_WEIGHT = _env_float("INFANTRY_RL_STAND_ROLL_WEIGHT", -3.5)
ROLL_FAIL_WEIGHT = _env_float("INFANTRY_RL_STAND_ROLL_FAIL_WEIGHT", -20.0)


@configclass
class StandEventCfg:
    """Start upright, with the hips tucked and the base just above the ground."""

    reset_root = EventTerm(
        func=mdp.reset_root_state_standing_squat,
        mode="reset",
        params={
            "pitch_range_deg": (-5.0, 5.0),
            "roll_range_deg": (-3.0, 3.0),
            "yaw_range_deg": (-180.0, 180.0),
            "z_offset_range": (-0.085, -0.065),
            "lin_vel_range": (-0.05, 0.05),
            "ang_vel_range": (-0.10, 0.10),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_hips = EventTerm(
        func=mdp.reset_hips_tucked,
        mode="reset",
        params={
            "tuck_prob": 1.0,
            "tuck_front_offset": -0.45,
            "tuck_back_offset": 0.45,
            "tuck_jitter": 0.02,
            "random_offset": 0.05,
            "velocity_range": (-0.2, 0.2),
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(HIP_JOINT_REGEX)),
        },
    )


@configclass
class StandCommandsCfg:
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
class StandRewardsCfg:
    """Stand up and stay standing; no tuck/extension shaping needed.

    The action term's zero hip target is nominal, so the position controller starts
    extending the legs immediately.  The reward only needs to make the endpoint
    (height + upright + still) valuable.
    """

    alive = RewTerm(func=base_mdp.is_alive, weight=0.5)
    upright = RewTerm(
        func=mdp.uprightness_power,
        weight=10.0,
        params={"power": UPRIGHT_POWER},
    )
    upright_zone = RewTerm(
        func=mdp.upright_zone,
        weight=3.0,
        params={"min_upright": UPRIGHT_ZONE_MIN, "min_height": MIN_HEIGHT},
    )
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
    recovered_bonus = RewTerm(
        func=mdp.recovered_bonus,
        weight=RECOVERED_BONUS_WEIGHT,
        params={"hold_time": HOLD_TIME_S},
    )
    base_height = RewTerm(
        func=mdp.base_height_abs,
        weight=-10.0,
        params={"target_height": NOMINAL_HEIGHT},
    )
    roll = RewTerm(func=mdp.roll_abs, weight=ROLL_WEIGHT)
    roll_fail = RewTerm(
        func=mdp.roll_fail_penalty,
        weight=ROLL_FAIL_WEIGHT,
        params={"limit_angle": ROLL_LIMIT_RAD},
    )
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
    closure_residual = RewTerm(
        func=mdp.closure_residual_l2, weight=-1.0e3, params={"action_name": "joints"}
    )


@configclass
class StandTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    roll_limit = DoneTerm(
        func=mdp.roll_above,
        params={"limit_angle": ROLL_LIMIT_RAD},
    )
    bad_orientation = DoneTerm(
        func=base_mdp.bad_orientation,
        params={"limit_angle": 1.2},
    )


@configclass
class StandEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: LocomotionObservationsCfg = LocomotionObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: StandRewardsCfg = StandRewardsCfg()
    terminations: StandTerminationsCfg = StandTerminationsCfg()
    commands: StandCommandsCfg = StandCommandsCfg()
    events: StandEventCfg = StandEventCfg()

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

        # Upright squat, zero command.  No tuck bias: zero action targets the
        # nominal stance, so the position controller extends the legs while the
        # policy balances.
        self.actions.joints.heading_control = False
        self.actions.joints.conditional_heading_control = False
        self.actions.joints.position_control = False
        self.actions.joints.sagittal_symmetry = True
        #: The stand phase is deliberately the simplest possible controller: the
        #: reset puts the hips in the deep tuck, and the action term's ZERO target
        #: is the nominal stance.  The soft position controller therefore extends
        #: the legs immediately and raises the base; no tuck bias/ramp is needed.
        #: A zero-action replay then tests the posture/extension directly.
        self.actions.joints.hip_bias = (0.0, 0.0, 0.0, 0.0)
        self.actions.joints.dynamic_hip_bias = False
        self.actions.joints.hip_bias_time_ramp = False

        print(
            f"[cfg] stand episode       : {EPISODE_LENGTH_S:.1f} s; start upright at "
            "base_z ~0.11 m with hips tucked; hold 0.5 s, extend over 1.5 s; "
            "wheel_scale=2.0 (balance only)"
        )
        print(
            f"[cfg] stand success       : upright ({UPRIGHT_MIN:.2f}) + height > "
            f"{MIN_HEIGHT:.2f} m + still (<{STOP_LIN_VEL:.2f} m/s, "
            f"<{STOP_ANG_VEL:.2f} rad/s) for {HOLD_TIME_S:.2f} s; no early termination"
        )

        self.viewer.origin_type = "world"
        self.viewer.eye = (2.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.2)


@configclass
class StandEnvCfg_PLAY(StandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6
        self.observations.policy.enable_corruption = False


class FiveBarStandEnv(FiveBarEnv):
    """Five-bar wheel-leg that starts upright in a squat and stands up."""

    cfg: StandEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.recovery_upright_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.recovery_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.extras.setdefault("log", {})["Metrics/recovery/success"] = float(
            self.recovery_success[env_ids].float().mean().item()
        )
        self.recovery_upright_steps[env_ids] = 0
        self.recovery_success[env_ids] = False


__all__ = [
    "EPISODE_LENGTH_S",
    "FiveBarStandEnv",
    "HOLD_TIME_S",
    "MIN_HEIGHT",
    "StandCommandsCfg",
    "StandEnvCfg",
    "StandEnvCfg_PLAY",
    "StandEventCfg",
    "StandRewardsCfg",
    "StandTerminationsCfg",
    "UPRIGHT_MIN",
]
