"""Balance task (stand still, hold heading / position) -- legacy, kept for parity.

This is the flat-layout equivalent of
``tasks/balance/{balance_env_cfg.py,balance_env.py}``.  It shares the scene, action
term, terminations and loop-closure environment with :mod:`env_locomotion`.

The main task is ``locomotion``; this one is kept because the original workspace
still ships it and the two tasks exercise different observation/reward sets.
Values are copied verbatim from the original workspace.
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import mdp
from env_locomotion import (
    FALL_ANGLE,
    FALL_HEIGHT,
    NOMINAL_HEIGHT,
    TARGET_HEADING,
    TARGET_X,
    TARGET_Y,
    ActionsCfg,
    FiveBarEnv,
    RobotSceneCfg,
    TerminationsCfg,
)

#: |heading error| (rad) inside which the bounded ``heading_alive`` bonus is paid
HEADING_TOLERANCE = 0.35
#: distance (m) inside which the bounded ``position_xy_alive`` bonus is paid
POSITION_TOLERANCE = 0.2


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (33 dims as configured)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations fed to the policy.

        ``heading`` was added after the first training runs: the robot learned to
        hold itself up while slowly spinning on the spot, because neither the
        observation nor the reward mentioned heading at all.  ``position`` was
        added for the same reason (perfect balance while drifting backwards).
        """

        # base state
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=0.25)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)

        # heading relative to the target direction, as (sin, cos) of the wrapped error
        heading = ObsTerm(func=mdp.heading_sin_cos, params={"target_heading": TARGET_HEADING})

        # planar position error relative to the target, in the base yaw frame
        # (forward, left) -- without this the policy cannot cancel a position drift
        position = ObsTerm(
            func=mdp.position_error, params={"target_x": TARGET_X, "target_y": TARGET_Y}
        )

        # leg state, relative to the commanded / solved values
        hip_pos_rel_nominal = ObsTerm(func=mdp.hip_pos_rel_nominal, params={"action_name": "joints"})
        knee_pos_rel_solved = ObsTerm(func=mdp.knee_pos_rel_solved, params={"action_name": "joints"})
        wheel_vel = ObsTerm(func=mdp.wheel_vel, params={"action_name": "joints"}, scale=0.1)

        # last action and the solved leg length / angle
        last_action = ObsTerm(func=mdp.last_action_raw, params={"action_name": "joints"})
        solved_leg_state = ObsTerm(func=mdp.solved_leg_state, params={"action_name": "joints"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the balance task."""

    # -- staying up ------------------------------------------------------
    alive = RewTerm(func=base_mdp.is_alive, weight=0.5)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-10.0, params={"target_height": NOMINAL_HEIGHT}
    )

    # -- hold the heading ------------------------------------------------
    # These three stop the "balanced but slowly rotating" solution.  At 1 rad of
    # heading error the penalty is comparable to the whole alive bonus.
    heading = RewTerm(
        func=mdp.heading_l2, weight=-0.5, params={"target_heading": TARGET_HEADING}
    )
    yaw_rate = RewTerm(func=mdp.yaw_rate_l2, weight=-0.25)
    heading_alive = RewTerm(
        func=mdp.heading_alive,
        weight=0.25,
        params={"target_heading": TARGET_HEADING, "tolerance": HEADING_TOLERANCE},
    )

    # -- hold the planar position ----------------------------------------
    # Same class of hole as the heading: the original reward never mentioned the
    # forward/backward position, so the robot learned to balance while drifting.
    # This must be learned rather than bolted on: forward/backward motion is the
    # common-mode wheel command, the very input the balance loop uses.
    position = RewTerm(
        func=mdp.position_xy_l2, weight=-4.0, params={"target_x": TARGET_X, "target_y": TARGET_Y}
    )
    lin_vel = RewTerm(func=mdp.lin_vel_xy_l2, weight=-1.0)
    position_alive = RewTerm(
        func=mdp.position_xy_alive,
        weight=0.2,
        params={"target_x": TARGET_X, "target_y": TARGET_Y, "tolerance": POSITION_TOLERANCE},
    )

    # -- smooth actuation ------------------------------------------------
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    wheel_spin = RewTerm(func=mdp.wheel_spin_l2, weight=-1.0e-4, params={"action_name": "joints"})

    # -- the closed loop must stay closed --------------------------------
    closure_residual = RewTerm(
        func=mdp.closure_residual_l2, weight=-1.0e3, params={"action_name": "joints"}
    )

    # -- falling is bad --------------------------------------------------
    fall_penalty = RewTerm(func=mdp.fall_penalty, weight=-50.0)


@configclass
class BalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the infantry-rl balance environment."""

    #: 1024 is the largest environment count this machine can run without running the
    #: 8 GB GPU (and PhysX' fabric path) out of memory; ``--num_envs`` overrides it.
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    #: name of the action term that owns the loop-closing knee targets
    joint_action_name: str = "joints"

    def __post_init__(self):
        # general settings
        self.decimation = 4  # 50 Hz control on a 200 Hz physics step
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Isaac Lab's default is 5 * 2**15; the doubled buffer only added GPU pressure.
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

        _iters = os.environ.get("INFANTRY_RL_PHYSX_POS_ITERS")
        if _iters:
            try:
                self.sim.physx.max_position_iteration_count = int(_iters)
            except ValueError:
                pass

        # -------------------------------------------------------------- #
        # Heading hold
        #
        # A) The policy holds the heading itself (observation + heading/yaw_rate
        #    rewards configured below) -- the default.
        # B) Keep an existing legacy policy and close the heading loop outside it:
        #    a differential wheel-speed PD is superimposed on the policy's wheel
        #    commands.  This requires no retraining.
        #
        #   INFANTRY_RL_HEADING_CONTROL=1   enable the outer heading loop (path B)
        #   INFANTRY_RL_NO_HEADING_OBS=1    drop the heading observation so a
        #                                   29-dim checkpoint still loads
        # -------------------------------------------------------------- #
        if os.environ.get("INFANTRY_RL_HEADING_CONTROL", "0").strip().lower() in ("1", "true", "yes", "on"):
            self.actions.joints.heading_control = True
            print(
                "[cfg] heading-hold outer loop ENABLED "
                f"(target={self.actions.joints.target_heading:.2f} rad, "
                f"kp={self.actions.joints.heading_kp}, kd={self.actions.joints.heading_kd})"
            )

        if os.environ.get("INFANTRY_RL_NO_HEADING_OBS", "0").strip().lower() in ("1", "true", "yes", "on"):
            # dropping the term restores the original 29-dim observation layout
            self.observations.policy.heading = None
            print("[cfg] heading observation DROPPED -> 29-dim observations (legacy checkpoint)")

        if os.environ.get("INFANTRY_RL_NO_POSITION_OBS", "0").strip().lower() in ("1", "true", "yes", "on"):
            self.observations.policy.position = None
            print("[cfg] position observation DROPPED")

        # -------------------------------------------------------------- #
        # Position hold (outer loop, no retraining)
        #
        # The same two choices as for the heading: retrain with the ``position``
        # observation + position/velocity rewards, or keep an existing policy and
        # add a COMMON-MODE wheel-speed PD.  Enable with:
        #     set INFANTRY_RL_POSITION_CONTROL=1
        #     set INFANTRY_RL_TARGET_X=1.0      (optional, metres; default 0)
        # -------------------------------------------------------------- #
        if os.environ.get("INFANTRY_RL_POSITION_CONTROL", "0").strip().lower() in ("1", "true", "yes", "on"):
            self.actions.joints.position_control = True
            _tx = os.environ.get("INFANTRY_RL_TARGET_X")
            if _tx:
                try:
                    self.actions.joints.target_position_x = float(_tx)
                except ValueError:
                    pass
            print(
                "[cfg] position-hold outer loop ENABLED "
                f"(target_x={self.actions.joints.target_position_x:.2f} m, "
                f"kp={self.actions.joints.position_kp}, kd={self.actions.joints.position_kd})"
            )
            print(
                "[cfg] WARNING: the position outer loop is EXPERIMENTAL and was measured "
                "to DESTABILISE the robot (it fell over). Forward/backward motion is the "
                "common-mode wheel command, i.e. the same input the balance loop uses, so "
                "it cannot be driven from outside without destroying balance. Prefer "
                "retraining with the 'position' observation + position/lin_vel rewards."
            )

        # viewer (only used in non-headless play)
        self.viewer.eye = (2.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.2)


@configclass
class BalanceEnvCfg_PLAY(BalanceEnvCfg):
    """Smaller, unrandomized variant used by play.py."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
        # no observation corruption at play time
        self.observations.policy.enable_corruption = False


class FiveBarBalanceEnv(FiveBarEnv):
    """Five-bar wheel-leg robot with the legacy stand-still task."""

    cfg: BalanceEnvCfg


__all__ = [
    "ActionsCfg",
    "BalanceEnvCfg",
    "BalanceEnvCfg_PLAY",
    "FALL_ANGLE",
    "FALL_HEIGHT",
    "FiveBarBalanceEnv",
    "HEADING_TOLERANCE",
    "NOMINAL_HEIGHT",
    "ObservationsCfg",
    "POSITION_TOLERANCE",
    "RewardsCfg",
    "RobotSceneCfg",
    "SceneEntityCfg",
    "TARGET_HEADING",
    "TerminationsCfg",
]
