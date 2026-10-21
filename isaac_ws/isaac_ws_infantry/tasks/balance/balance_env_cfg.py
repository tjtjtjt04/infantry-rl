"""Balance task configuration (LEGACY: the main task is now ``locomotion``).

The scene, action term and termination set are shared and live in
:mod:`isaac_ws_infantry.tasks.common.robot_env_cfg`; this file only holds what is
specific to "stand still and hold a heading/position".

Original notes: balance task configuration for the infantry-rl five-bar robot.

Mirrors the shape of Isaac Lab 2.3.2's own manager-based tasks (see
``isaac_ws/docs/API_NOTES.md`` sections 10 and GOTCHAS), with two deliberate
departures forced by this machine and this robot:

* the ground is a **local** USD plane, because ``terrain_type="plane"`` (and hence
  ``sim_utils.spawn_ground_plane``) fetches a Nucleus asset and fails offline;
* the actions are one flat custom term, because the knees must be solved from the
  loop closure instead of being commanded.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaac_ws_infantry.mdp as mdp
from isaac_ws_infantry.tasks.common.robot_env_cfg import (  # noqa: F401
    FALL_ANGLE,
    FALL_HEIGHT,
    NOMINAL_HEIGHT,
    ActionsCfg,
    RobotSceneCfg,
    TARGET_HEADING,
    TARGET_X,
    TARGET_Y,
    TerminationsCfg,
)

#: |heading error| (rad) inside which the bounded ``heading_alive`` bonus is paid
HEADING_TOLERANCE = 0.35
#: distance (m) inside which the bounded ``position_xy_alive`` bonus is paid
POSITION_TOLERANCE = 0.2


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations fed to the policy (31 dims).

        ``heading`` was ADDED after the first training runs: the robot learned to hold
        itself up while slowly spinning on the spot, because neither the observation
        nor the reward mentioned heading at all.  ``base_ang_vel`` carries the yaw
        *rate* (body z) but not the absolute heading, so the policy could not have
        held a direction even if it had wanted to.
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
    # These three are what stop the "balanced but slowly rotating" solution.  The
    # weights are tuned so that at 1 rad of heading error the penalty is comparable
    # to the whole alive bonus (alive = 0.5 * dt), i.e. spinning is never free, yet
    # the balance terms still dominate the gradient near the target.
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
    # forward/backward position, so the robot learned to balance while drifting
    # (measured: -0.245 m of backwards drift in a 10 s episode).
    #
    # Why this must be learned rather than bolted on: heading is a DIFFERENTIAL wheel
    # command and is therefore orthogonal to balance, but forward/backward motion is
    # the COMMON-MODE wheel command -- the very input the balance loop uses.  Driving
    # it from an outer wheel-speed loop tips the robot over (verified: it fell over).
    # So the policy has to learn to lean and translate as one skill.
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
        # Isaac Lab's default is 5 * 2**15; this scene only has a flat ground plane and
        # 1024 small robots (~11 links each), so the doubled buffer bought nothing and
        # only added GPU memory pressure -- the exact resource whose exhaustion shows up
        # as "CUDA error: an illegal memory access" inside the PhysX GPU allocator.
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15

        # ------------------------------------------------------------------ #
        # GPU stability knobs (environment variables, so no code edit is needed)
        #
        # Observed on this machine (RTX 4060 Laptop 8 GB, driver 580.88, WDDM with
        # a display attached): with 4096 environments the run dies after a few
        # minutes with
        #     CUDA error: an illegal instruction was encountered
        # surfaced in the PPO update, with the real fault in
        #     omni.physx.fabric.plugin  DirectGpuHelper.cpp
        # i.e. the PhysX *fabric* direct-GPU buffer path.  The crash dump also
        # showed BAR1 memory exhausted (8164 / 8192 MiB) and SW thermal/power
        # throttling active, so this is a GPU resource/stability limit rather than
        # a bug in the task.
        #
        #   INFANTRY_RL_USE_FABRIC=0   disable the fabric path (slower, more stable)
        #   INFANTRY_RL_PHYSX_POS_ITERS=4   lower the PhysX position iterations
        # ------------------------------------------------------------------ #
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
            except ValueError:
                pass

        # ------------------------------------------------------------------ #
        # Heading hold
        #
        # There are two independent ways to stop the robot spinning; pick ONE:
        #
        # A) The policy holds the heading itself -- retrain with the (31-dim)
        #    observation + the heading/yaw_rate rewards that are configured below.
        #    This is the better long-term solution but needs a fresh policy.
        #
        # B) Keep an existing 29-dim policy and close the heading loop outside it:
        #    a differential wheel-speed PD controller is superimposed on the policy's
        #    wheel commands (see FiveBarJointAction._effective_wheel_targets).  This
        #    requires NO retraining, so it works with the already-trained checkpoint.
        #
        #   INFANTRY_RL_HEADING_CONTROL=1   enable the outer heading loop (path B)
        #   INFANTRY_RL_NO_HEADING_OBS=1    drop the heading observation so a 29-dim
        #                                   checkpoint still loads (path B needs this)
        #
        # The outer loop is OFF during path-A training, otherwise the controller and
        # the policy would fight over yaw.
        # ------------------------------------------------------------------ #
        if _os.environ.get("INFANTRY_RL_HEADING_CONTROL", "0").strip().lower() in ("1", "true", "yes", "on"):
            self.actions.joints.heading_control = True
            print(
                "[cfg] heading-hold outer loop ENABLED "
                f"(target={self.actions.joints.target_heading:.2f} rad, "
                f"kp={self.actions.joints.heading_kp}, kd={self.actions.joints.heading_kd})"
            )

        if _os.environ.get("INFANTRY_RL_NO_HEADING_OBS", "0").strip().lower() in ("1", "true", "yes", "on"):
            # dropping the term restores the original 29-dim observation layout
            self.observations.policy.heading = None
            print("[cfg] heading observation DROPPED -> 29-dim observations (legacy checkpoint)")

        if _os.environ.get("INFANTRY_RL_NO_POSITION_OBS", "0").strip().lower() in ("1", "true", "yes", "on"):
            # dropping the term restores the pre-position observation layout
            self.observations.policy.position = None
            print("[cfg] position observation DROPPED")

        # ------------------------------------------------------------------ #
        # Position hold (outer loop, no retraining)
        #
        # The same two choices as for the heading: retrain with the ``position``
        # observation + position/velocity rewards (the policy holds the target by
        # itself), or keep an existing policy and add a COMMON-MODE wheel-speed PD
        # that cancels the drift through the mean wheel speed.  Enable with:
        #     set INFANTRY_RL_POSITION_CONTROL=1
        #     set INFANTRY_RL_TARGET_X=1.0      (optional, metres; default 0 = stay put)
        # ------------------------------------------------------------------ #
        if _os.environ.get("INFANTRY_RL_POSITION_CONTROL", "0").strip().lower() in ("1", "true", "yes", "on"):
            self.actions.joints.position_control = True
            _tx = _os.environ.get("INFANTRY_RL_TARGET_X")
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
    """Smaller, unrandomized variant used by ``play.py``."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
        # no observation corruption at play time
        self.observations.policy.enable_corruption = False


__all__ = [
    "BalanceEnvCfg",
    "BalanceEnvCfg_PLAY",
    "RobotSceneCfg",
    "ActionsCfg",
    "ObservationsCfg",
    "RewardsCfg",
    "TerminationsCfg",
    "NOMINAL_HEIGHT",
    "FALL_HEIGHT",
    "FALL_ANGLE",
    "TARGET_HEADING",
    "HEADING_TOLERANCE",
    "SceneEntityCfg",
]
