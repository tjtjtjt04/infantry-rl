"""Task-shared environment configuration for the infantry-rl five-bar robot.

Everything here is common to the tasks in :mod:`isaac_ws_infantry.tasks` -- the flat
offline ground plus the robot (:class:`RobotSceneCfg`), the single flat six-dimensional
action term (:class:`ActionsCfg`) and the fall conditions (:class:`TerminationsCfg`), plus
the measured stance constants they depend on.

Task-specific pieces (commands, observations, rewards, the env class) live in each task's
own directory:

* :mod:`isaac_ws_infantry.tasks.locomotion` -- the main task: track a forward velocity
  command while keeping pitch/roll at zero;
* :mod:`isaac_ws_infantry.tasks.balance` -- the earlier task: stand still and hold a
  heading/position (kept as a reference).
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaac_ws_infantry.mdp as mdp
from isaac_ws_infantry.robot_model import nominal_base_height
from isaac_ws_infantry.robots import INFANTRY_RL_CFG
from isaac_ws_infantry.terrains import ensure_ground_usd

#: root height (m) of the measured nominal stance; the base-height reward holds this
NOMINAL_HEIGHT = nominal_base_height()
#: root height below which the robot counts as fallen (m)
FALL_HEIGHT = 0.11
#: tilt at which the robot counts as fallen (rad)
FALL_ANGLE = 1.2

#: heading (rad) the robot is supposed to hold when the action term's outer heading loop
#: is enabled.  All envs spawn with an identity base orientation, so 0 = "as at reset".
TARGET_HEADING = 0.0
#: planar position (m) the robot is supposed to hold, in the ENVIRONMENT frame
#: (``env_origins`` is the spawn point, so (0, 0) = "do not translate").
TARGET_X = 0.0
TARGET_Y = 0.0


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat ground + the robot, shared by every task. Everything is local, nothing is fetched."""

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
class TerminationsCfg:
    """Termination terms for the balance task."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": FALL_ANGLE})
    base_too_low = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": FALL_HEIGHT}
    )


__all__ = [
    "ActionsCfg",
    "FALL_ANGLE",
    "FALL_HEIGHT",
    "NOMINAL_HEIGHT",
    "RobotSceneCfg",
    "TARGET_HEADING",
    "TARGET_X",
    "TARGET_Y",
    "TerminationsCfg",
]
