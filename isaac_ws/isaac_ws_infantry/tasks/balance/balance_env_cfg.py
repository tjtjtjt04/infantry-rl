"""Balance task configuration for the infantry-rl five-bar robot.

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
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaac_ws_infantry.mdp as mdp
from isaac_ws_infantry.model import nominal_base_height
from isaac_ws_infantry.robots import INFANTRY_RL_CFG
from isaac_ws_infantry.terrains import ensure_ground_usd

#: root height (m) of the measured nominal stance; the base-height reward holds this
NOMINAL_HEIGHT = nominal_base_height()
#: root height below which the robot counts as fallen (m)
FALL_HEIGHT = 0.11
#: tilt at which the robot counts as fallen (rad)
FALL_ANGLE = 1.2
#: heading (rad) the robot is supposed to hold. All environments spawn with an
#: identity base orientation, so 0 means "same direction as at reset".
TARGET_HEADING = 0.0
#: |heading error| (rad) inside which the bounded ``heading_alive`` bonus is paid
HEADING_TOLERANCE = 0.35
#: planar position (m) the robot is supposed to hold, in the ENVIRONMENT frame
#: (``env_origins`` is the spawn point, so (0, 0) = "do not translate").  To command a
#: displacement, change these two numbers -- or see the outer-loop option below.
TARGET_X = 0.0
TARGET_Y = 0.0
#: distance (m) inside which the bounded ``position_xy_alive`` bonus is paid
POSITION_TOLERANCE = 0.2


@configclass
class BalanceSceneCfg(InteractiveSceneCfg):
    """Flat ground + the robot. Everything is local, nothing is fetched."""

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
        hip_scale=0.25,
        wheel_scale=5.0,
        hip_range=0.45,
        limit_margin=0.01,
        # 2 is provably enough (worst-case residual 3.6 um vs 750 um achievable in sim);
        # each extra iteration costs ~150 CUDA kernel launches per control step.
        solver_iters=2,
        # outer heading loop; enabled via __post_init__ / INFANTRY_RL_HEADING_CONTROL
        heading_control=False,
        heading_mode="override",
        target_heading=TARGET_HEADING,
        heading_kp=3.0,
        heading_kd=0.35,
        heading_max_yaw_rate=3.0,
        wheel_radius=0.068,
        track_width=0.4379,
        # outer position loop; enabled via __post_init__ / INFANTRY_RL_POSITION_CONTROL
        position_control=False,
        target_position_x=TARGET_X,
        target_position_y=TARGET_Y,
        position_kp=1.5,
        position_kd=0.6,
        position_max_speed=0.6,
    )


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
class TerminationsCfg:
    """Termination terms for the balance task."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": FALL_ANGLE})
    base_too_low = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": FALL_HEIGHT}
    )


@configclass
class BalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the infantry-rl balance environment."""

    #: 1024 is the largest environment count this machine can run without running the
    #: 8 GB GPU (and PhysX' fabric path) out of memory; ``--num_envs`` overrides it.
    scene: BalanceSceneCfg = BalanceSceneCfg(num_envs=1024, env_spacing=2.5)
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
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

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
    "BalanceSceneCfg",
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
